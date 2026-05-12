#!/bin/bash
# Start the Forward & Backward pass demo
cd "$(dirname "$0")"
echo "============================================"
echo "  MLP Training — Forward & Backward Pass"
echo "  Open: http://127.0.0.1:8080"
echo "  Stop: Ctrl+C"
echo "============================================"
conda run -n python3 python app.py
