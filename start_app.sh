#!/bin/bash
# Starts the Critical Care Flow clinical agent, accessible on all network interfaces.
# Access from any device on the same Tailscale network:
#   http://<your-mac-tailscale-ip>:8501

cd "/Users/erikgary/Documents/Agentic Workflows/Critical Care Flow"

python3 -m streamlit run 3_clinical_agent.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true \
  --browser.gatherUsageStats=false
