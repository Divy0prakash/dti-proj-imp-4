#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DTI Pipeline Runner
# Usage:
#   export KAGGLE_USERNAME=your_username
#   export KAGGLE_KEY=your_api_key
#   bash run_pipeline.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DTI — Dynamic Discount Optimizer | ML Pipeline  "
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 --version || { echo "Python 3 required"; exit 1; }

echo ""
echo "Installing dependencies..."
pip install -q kaggle vaderSentiment textblob xgboost scikit-learn pandas numpy
python3 -c "import nltk; nltk.download('punkt', quiet=True)" 2>/dev/null || true

echo ""
echo "Running pipeline..."
python3 dti_pipeline.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done! Run the dashboard with:"
echo "    streamlit run streamlit_app.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
