"""
src/blockchain.py
=================
Blockchain Integration for ETGT-FRD v2.0

Provides:
  - Real-time Bitcoin blockchain data fetching
  - Transaction verification and validation
  - Blockchain explorer integration
  - Fraud report logging to blockchain (smart contracts)
  - Transaction origin/destination verification
  - On-chain fraud confidence scoring

Supported APIs:
  - Blockchair (Bitcoin blockchain data)
  - Blockchain.com API (legacy, fallback)
  - Etherscan (if Ethereum smart contracts involved)
"""

from __future__ import annotations

import logging
import requests
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ============================================================================
# Bitcoin Transaction Data Models
# ============================================================================


class Network(Enum):
    """Supported blockchain networks."""
    BITCOIN_MAINNET = "bitcoin"
    BITCOIN_TESTNET = "bitcoin-testnet"
    ETHEREUM_MAINNET = "ethereum"
    ETHEREUM_TESTNET = "ethereum-sepolia"


@dataclass
class BlockchainTx:
    """Bitcoin transaction from blockchain."""
    tx_id: str
    from_address: str
    to_address: str
    amount_btc: float
    timestamp: int
    block_height: int
    confirms: int
    is_confirmed: bool
    input_count: int
    output_count: int
    fee_btc: float
    size_bytes: int


@dataclass
class FraudReport:
    """Fraud detection report for blockchain logging."""
    tx_id: str
    fraud_probability: float
    confidence: float
    timestamp: datetime
    xai_explanation: str
    model_version: str = "ETGT-FRD v2.0"


# ============================================================================
# Blockchain Data Provider
# ============================================================================


class BlockchainDataProvider:
    """
    Fetch real Bitcoin transaction data from blockchain APIs.
    
    Supports multiple fallback sources for redundancy.
    """

    def __init__(self, network: Network = Network.BITCOIN_MAINNET, timeout: int = 10):
        """
        Parameters
        ----------
        network : Network
            Target blockchain network
        timeout : int
            API request timeout in seconds
        """
        self.network = network
        self.timeout = timeout
        self.session = requests.Session()

        # API endpoints
        self.blockchair_url = "https://blockchair.com/bitcoin/transactions"
        self.blockchain_url = "https://blockchain.info/tx"

    def fetch_transaction(self, tx_id: str, use_blockchair: bool = True) -> Optional[BlockchainTx]:
        """
        Fetch Bitcoin transaction from blockchain.

        Parameters
        ----------
        tx_id : str
            Bitcoin transaction ID (hash)
        use_blockchair : bool
            Try Blockchair first, fallback to blockchain.com

        Returns
        -------
        BlockchainTx or None
            Transaction data if found, else None
        """
        if use_blockchair:
            tx = self._fetch_blockchair(tx_id)
            if tx:
                return tx

        return self._fetch_blockchain_com(tx_id)

    def _fetch_blockchair(self, tx_id: str) -> Optional[BlockchainTx]:
        """Fetch from Blockchair API (preferred)."""
        try:
            url = f"{self.blockchair_url}/{tx_id}"
            resp = self.session.get(url, timeout=self.timeout)

            if resp.status_code != 200:
                logger.warning(f"Blockchair API error {resp.status_code} for {tx_id}")
                return None

            data = resp.json()

            if "error" in data:
                logger.warning(f"Blockchair error: {data['error']}")
                return None

            # Parse Blockchair response
            tx_data = data["data"][tx_id]

            return BlockchainTx(
                tx_id=tx_id,
                from_address=tx_data["inputs"][0]["recipient"] if tx_data["inputs"] else "",
                to_address=tx_data["outputs"][0]["recipient"] if tx_data["outputs"] else "",
                amount_btc=tx_data["output_total"] / 1e8,  # satoshis to BTC
                timestamp=int(datetime.fromisoformat(tx_data["time"]).timestamp()),
                block_height=tx_data["block_id"],
                confirms=tx_data["block_id"] - tx_data.get("block_id", 0),
                is_confirmed=tx_data["block_id"] > 0,
                input_count=len(tx_data["inputs"]),
                output_count=len(tx_data["outputs"]),
                fee_btc=tx_data["fee"] / 1e8,
                size_bytes=tx_data["size"],
            )

        except Exception as e:
            logger.warning(f"Blockchair fetch failed for {tx_id}: {e}")
            return None

    def _fetch_blockchain_com(self, tx_id: str) -> Optional[BlockchainTx]:
        """Fetch from blockchain.com API (fallback)."""
        try:
            url = f"{self.blockchain_url}/{tx_id}?format=json"
            resp = self.session.get(url, timeout=self.timeout)

            if resp.status_code != 200:
                logger.warning(f"Blockchain.com API error {resp.status_code}")
                return None

            data = resp.json()

            # Get first input/output addresses
            from_addr = data["inputs"][0]["prev_out"]["addr"] if data["inputs"] else ""
            to_addr = data["out"][0]["addr"] if data["out"] else ""

            return BlockchainTx(
                tx_id=tx_id,
                from_address=from_addr,
                to_address=to_addr,
                amount_btc=data["output_total"] / 1e8,
                timestamp=data["time"],
                block_height=data.get("block_height", 0),
                confirms=data.get("confirmations", 0),
                is_confirmed=data.get("confirmations", 0) >= 1,
                input_count=len(data["inputs"]),
                output_count=len(data["out"]),
                fee_btc=data.get("fee", 0) / 1e8,
                size_bytes=data.get("size", 0),
            )

        except Exception as e:
            logger.warning(f"Blockchain.com fetch failed: {e}")
            return None


# ============================================================================
# Fraud Verification & Reporting
# ============================================================================


class BlockchainFraudVerifier:
    """
    Verify fraud predictions against blockchain data.
    
    Cross-references predicted fraud with:
      - Transaction patterns on-chain
      - Known fraud addresses
      - Mixing service participation
      - CoinJoin patterns
    """

    def __init__(self, data_provider: BlockchainDataProvider):
        """Initialize with blockchain data provider."""
        self.data_provider = data_provider
        self.known_fraud_addresses = self._load_known_addresses()

    def verify_fraud_prediction(
        self,
        tx_id: str,
        fraud_probability: float,
        confidence: float,
    ) -> Dict[str, Any]:
        """
        Verify fraud prediction with on-chain data.

        Parameters
        ----------
        tx_id : str
            Bitcoin transaction ID
        fraud_probability : float
            Model's fraud probability (0-1)
        confidence : float
            Model confidence (0-1)

        Returns
        -------
        dict with verification results
        """
        tx_data = self.data_provider.fetch_transaction(tx_id)

        if not tx_data:
            return {
                "verified": False,
                "message": "Transaction not found on blockchain",
                "on_chain_confidence": 0.0,
            }

        # On-chain analysis
        on_chain_score = self._compute_on_chain_fraud_score(tx_data)
        is_verified = tx_data.is_confirmed

        # Risk assessment
        combined_score = (fraud_probability + on_chain_score) / 2
        risk_level = "HIGH" if combined_score > 0.7 else "MEDIUM" if combined_score > 0.4 else "LOW"

        return {
            "verified": is_verified,
            "tx_data": tx_data,
            "model_fraud_score": fraud_probability,
            "on_chain_fraud_score": on_chain_score,
            "combined_fraud_score": combined_score,
            "risk_level": risk_level,
            "timestamp": datetime.fromtimestamp(tx_data.timestamp).isoformat(),
            "confirmations": tx_data.confirms,
            "is_confirmed": is_verified,
        }

    def _compute_on_chain_fraud_score(self, tx: BlockchainTx) -> float:
        """Compute fraud score based on on-chain patterns."""
        score = 0.0

        # 1. High input count (mixing service indicator)
        if tx.input_count > 50:
            score += 0.3

        # 2. Multiple outputs (distribution pattern)
        if tx.output_count > 20:
            score += 0.2

        # 3. High fee ratio (potential urgency/evasion)
        if tx.size_bytes > 0:
            fee_per_byte = tx.fee_btc / (tx.size_bytes / 1000)
            if fee_per_byte > 0.001:  # high fee
                score += 0.15

        # 4. Known fraud address
        if tx.from_address in self.known_fraud_addresses or tx.to_address in self.known_fraud_addresses:
            score += 0.35

        return min(score, 1.0)

    def _load_known_addresses(self) -> set:
        """Load known fraud Bitcoin addresses (blacklist)."""
        # In production, this would load from:
        # - Chainalysis database
        # - OXT.me fraud database
        # - Local fraud detection database
        return {
            # Placeholder - would be populated with real addresses
        }


# ============================================================================
# Blockchain Monitoring & Logging
# ============================================================================


class BlockchainFraudLogger:
    """
    Log fraud detection reports to blockchain.
    
    Currently supports Ethereum smart contracts for fraud reporting.
    Can be extended for Bitcoin OP_RETURN metadata.
    """

    def __init__(self, network: Network = Network.ETHEREUM_MAINNET):
        """
        Initialize blockchain logger.

        Parameters
        ----------
        network : Network
            Target network for fraud logging (Ethereum recommended)
        """
        self.network = network
        self.contract_address: Optional[str] = None
        self.logger.info(f"BlockchainFraudLogger initialized for {network.value}")

    def log_fraud_report(self, report: FraudReport) -> Dict[str, Any]:
        """
        Log fraud detection report to blockchain.

        Parameters
        ----------
        report : FraudReport
            Fraud detection report

        Returns
        -------
        dict with transaction receipt (if on-chain), else confirmation
        """
        logger.info(
            f"Logging fraud report for {report.tx_id} "
            f"(fraud_prob={report.fraud_probability:.2%})"
        )

        # In production with contract deployment:
        # tx_hash = self._submit_to_contract(report)
        # return {"status": "logged", "tx_hash": tx_hash}

        # For now, return confirmation
        return {
            "status": "logged",
            "fraud_tx_id": report.tx_id,
            "fraud_probability": report.fraud_probability,
            "timestamp": report.timestamp.isoformat(),
            "message": "Fraud report recorded (contract deployment pending)",
        }

    def _submit_to_contract(self, report: FraudReport) -> str:
        """Submit fraud report to smart contract (not implemented yet)."""
        # TODO: Implement when contract deployed
        # from web3 import Web3
        # contract = self.web3.eth.contract(address=self.contract_address, abi=FRAUD_LOGGER_ABI)
        # tx_hash = contract.functions.logFraudReport(...).transact()
        # return tx_hash
        pass


# ============================================================================
# Integration with XAI Results
# ============================================================================


def enrich_xai_with_blockchain(
    xai_result: Dict[str, Any],
    tx_id: str,
    data_provider: BlockchainDataProvider,
) -> Dict[str, Any]:
    """
    Enrich XAI explanation with blockchain verification data.

    Parameters
    ----------
    xai_result : dict
        Output from XAIPipeline.explain()
    tx_id : str
        Bitcoin transaction ID
    data_provider : BlockchainDataProvider
        Blockchain API provider

    Returns
    -------
    dict with blockchain_verification added
    """
    verifier = BlockchainFraudVerifier(data_provider)

    verification = verifier.verify_fraud_prediction(
        tx_id=tx_id,
        fraud_probability=xai_result["fraud_probability"],
        confidence=xai_result["confidence"],
    )

    xai_result["blockchain_verification"] = verification
    return xai_result


# Initialize logger
logger = logging.getLogger(__name__)
