"""
Comprehensive performance benchmarking suite.

Measures:
- Model inference latency
- Memory usage
- XAI explanation time
- Throughput
- Scalability
"""

import torch
import time
import psutil
import json
from pathlib import Path
from datetime import datetime


class BenchmarkSuite:
    def __init__(self):
        self.results = {}
        self.timestamps = {}
        
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 2)
    
    def measure_latency(self, func, num_iterations=10):
        """Measure function latency"""
        times = []
        for _ in range(num_iterations):
            start = time.time()
            func()
            times.append((time.time() - start) * 1000)  # Convert to ms
        
        return {
            "min": min(times),
            "max": max(times),
            "mean": sum(times) / len(times),
            "median": sorted(times)[len(times)//2],
            "p95": sorted(times)[int(len(times)*0.95)]
        }
    
    def benchmark_model_loading(self):
        """Benchmark model loading time"""
        from src.model import ETGT_FRD
        from src.utils import get_device, load_checkpoint
        import yaml
        
        print("\n📦 **Model Loading Benchmark:**")
        
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        
        # Time model instantiation
        start = time.time()
        device = get_device(cfg)
        model = ETGT_FRD.from_config(cfg).to(device)
        create_time = (time.time() - start) * 1000
        
        # Time checkpoint loading
        ckpt_path = Path("outputs/checkpoints/best_model.pt")
        start = time.time()
        load_checkpoint(model, ckpt_path, device=str(device))
        load_time = (time.time() - start) * 1000
        
        model.eval()
        
        print(f"✅ Model creation: {create_time:.1f}ms")
        print(f"✅ Checkpoint loading: {load_time:.1f}ms")
        print(f"✅ Total: {create_time + load_time:.1f}ms")
        
        return {
            "creation_ms": create_time,
            "loading_ms": load_time,
            "total_ms": create_time + load_time
        }, model, cfg, device
    
    def benchmark_data_loading(self, cfg):
        """Benchmark data loading"""
        from src.data_loader import EllipticDataLoader
        
        print("\n📊 **Data Loading Benchmark:**")
        
        start = time.time()
        loader = EllipticDataLoader(cfg)
        data, splits = loader.load()
        load_time = (time.time() - start) * 1000
        
        print(f"✅ Data loading: {load_time:.1f}ms")
        print(f"   Nodes: {data.x.shape[0]:,}")
        print(f"   Edges: {data.edge_index.shape[1]:,}")
        print(f"   Features: {data.x.shape[1]}")
        
        return {"data_loading_ms": load_time}, data
    
    def benchmark_forward_pass(self, model, cfg, device, data):
        """Benchmark model forward pass"""
        print("\n🚀 **Forward Pass Latency:**")
        
        def forward_func():
            with torch.no_grad():
                model(
                    data.x.to(device),
                    data.edge_index.to(device),
                    data.edge_attr.to(device)
                )
        
        latencies = self.measure_latency(forward_func, num_iterations=5)
        
        for key, val in latencies.items():
            print(f"✅ {key}: {val:.1f}ms")
        
        return {"forward_pass": latencies}
    
    def benchmark_mc_dropout(self, model, cfg, device, data):
        """Benchmark MC-Dropout uncertainty"""
        print("\n🔀 **MC-Dropout Uncertainty (5 passes):**")
        
        def mc_func():
            model.predict_with_uncertainty(
                data.x.to(device),
                data.edge_index.to(device),
                data.edge_attr.to(device),
                num_forward_passes=5
            )
        
        latencies = self.measure_latency(mc_func, num_iterations=3)
        
        for key, val in latencies.items():
            print(f"✅ {key}: {val:.1f}ms")
        
        return {"mc_dropout": latencies}
    
    def benchmark_xai_pipeline(self, model, cfg, device, data):
        """Benchmark XAI explanation time"""
        from src.explain import XAIPipeline
        
        print("\n💡 **XAI Pipeline Latency:**")
        
        xai = XAIPipeline(model, cfg, device, use_llm=False)
        
        def xai_func():
            xai.explain(0, data.x, data.edge_index, data.edge_attr)
        
        latencies = self.measure_latency(xai_func, num_iterations=2)
        
        for key, val in latencies.items():
            print(f"✅ {key}: {val:.1f}ms")
        
        return {"xai_pipeline": latencies}
    
    def benchmark_memory(self, model, cfg, device, data):
        """Benchmark memory usage"""
        print("\n💾 **Memory Usage:**")
        
        import gc
        gc.collect()
        baseline = self.get_memory_usage()
        
        # Load full batch
        with torch.no_grad():
            logits, probs = model(
                data.x.to(device),
                data.edge_index.to(device),
                data.edge_attr.to(device)
            )
        
        peak = self.get_memory_usage()
        
        print(f"✅ Baseline: {baseline:.1f}MB")
        print(f"✅ Peak: {peak:.1f}MB")
        print(f"✅ Inference memory: {peak - baseline:.1f}MB")
        
        return {
            "baseline_mb": baseline,
            "peak_mb": peak,
            "inference_mb": peak - baseline
        }
    
    def benchmark_throughput(self, model, cfg, device, data, batch_size=100):
        """Benchmark prediction throughput"""
        print(f"\n📈 **Throughput (batch size {batch_size}):**")
        
        num_batches = 10
        total_time = 0
        total_samples = 0
        
        for i in range(num_batches):
            start = time.time()
            with torch.no_grad():
                model(
                    data.x[:batch_size].to(device),
                    data.edge_index.to(device),
                    data.edge_attr.to(device)
                )
            total_time += (time.time() - start)
            total_samples += batch_size
        
        throughput = total_samples / total_time
        
        print(f"✅ Samples per second: {throughput:.1f}")
        print(f"✅ Latency per batch: {(total_time/num_batches)*1000:.1f}ms")
        
        return {"throughput_sps": throughput}
    
    def save_results(self, results):
        """Save benchmark results to JSON"""
        output_file = Path("benchmarks_results.json")
        results["timestamp"] = datetime.now().isoformat()
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📝 Results saved to {output_file}")
    
    def run_full_benchmark(self):
        """Run complete benchmarking suite"""
        print("=" * 70)
        print("ETGT-FRD v2.0 - Performance Benchmarks")
        print("=" * 70)
        
        results = {}
        
        # 1. Model loading
        loading_results, model, cfg, device = self.benchmark_model_loading()
        results["model_loading"] = loading_results
        
        # 2. Data loading
        data_results, data = self.benchmark_data_loading(cfg)
        results["data_loading"] = data_results
        
        # 3. Forward pass
        fwd_results = self.benchmark_forward_pass(model, cfg, device, data)
        results["forward_pass"] = fwd_results
        
        # 4. MC-Dropout
        mc_results = self.benchmark_mc_dropout(model, cfg, device, data)
        results["mc_dropout"] = mc_results
        
        # 5. XAI Pipeline
        xai_results = self.benchmark_xai_pipeline(model, cfg, device, data)
        results["xai_pipeline"] = xai_results
        
        # 6. Memory
        mem_results = self.benchmark_memory(model, cfg, device, data)
        results["memory"] = mem_results
        
        # 7. Throughput
        tp_results = self.benchmark_throughput(model, cfg, device, data)
        results["throughput"] = tp_results
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        print("\n⏱️  **Latency Summary:**")
        print(f"Model loading: {loading_results['total_ms']:.0f}ms")
        print(f"Forward pass (mean): {fwd_results['forward_pass']['mean']:.0f}ms")
        print(f"MC-Dropout (5 pass): {mc_results['mc_dropout']['mean']:.0f}ms")
        print(f"XAI Pipeline: {xai_results['xai_pipeline']['mean']:.0f}ms")
        
        print("\n💾 **Memory Usage:**")
        print(f"Baseline: {mem_results['baseline_mb']:.0f}MB")
        print(f"Peak: {mem_results['peak_mb']:.0f}MB")
        
        print("\n📈 **Throughput:**")
        print(f"{tp_results['throughput_sps']:.1f} samples/second")
        
        print("\n" + "=" * 70)
        
        self.save_results(results)
        return results


def main():
    try:
        suite = BenchmarkSuite()
        results = suite.run_full_benchmark()
        return 0
    except Exception as e:
        print(f"\n❌ Benchmarking failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
