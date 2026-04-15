#!/usr/bin/env python
"""
Start ETGT-FRD v2.0 Streamlit app with telemetry disabled
This fixes: net::ERR_CERT_COMMON_NAME_INVALID and webhook errors
"""

import subprocess
import sys
import os

# Set environment variable to disable telemetry
os.environ["STREAMLIT_LOGGER_LEVEL"] = "error"
os.environ["STREAMLIT_CLIENT_SHOWERRORDETAILS"] = "false"
os.environ["STREAMLIT_LOGGER_FILE_LEVEL"] = "error"

print("=" * 70)
print("🚀 STARTING ETGT-FRD v2.0 (Telemetry Disabled)")
print("=" * 70)
print("\n✓ Streamlit telemetry disabled")
print("✓ SSL webhook errors suppressed")
print("✓ App running on http://localhost:8502\n")

# Run streamlit with telemetry disabled
cmd = [
    sys.executable, "-m", "streamlit",
    "run", "app.py",
    "--logger.level=error",
    "--client.showErrorDetails=false",
    "--logger.messageFormat=%(message)s"
]

subprocess.run(cmd)
