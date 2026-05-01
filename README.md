# 🎯 DTI — Dynamic Discount Optimizer

> ML-powered discount optimization dashboard built with XGBoost, VADER sentiment analysis, and Streamlit.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 📁 Repo Structure

```
dti-discount-optimizer/
├── streamlit_app.py          # Main dashboard (entry point)
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Theme & server config
├── .gitignore
└── README.md
```

> **Optional data files** (place in repo root or use Streamlit Secrets for cloud storage):
> - `unified_dataset.csv`
> - `top_recommendations.csv`
> - `xgboost_model.json`
> - `scaler.pkl`
>
> If files are missing the dashboard runs in **Demo Mode** with synthetic data.

---

## 🚀 Deploy to Streamlit Cloud

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "feat: initial DTI dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/dti-discount-optimizer.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"New app"**
3. Set:
   - **Repository:** `YOUR_USERNAME/dti-discount-optimizer`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
4. Click **"Deploy!"**

---

## 📦 Including Large Files (Model + CSVs)

Streamlit Cloud has a **1 GB repo limit**. For large model/data files, choose one:

### Option A — Commit directly (files < 100 MB each)
```bash
git add unified_dataset.csv top_recommendations.csv xgboost_model.json scaler.pkl
git commit -m "add data and model files"
git push
```

### Option B — Git LFS (files 100 MB–1 GB)
```bash
git lfs install
git lfs track "*.csv" "*.pkl" "*.json"
git add .gitattributes
git add unified_dataset.csv top_recommendations.csv xgboost_model.json scaler.pkl
git commit -m "add large files via LFS"
git push
```

### Option C — Streamlit Secrets + Cloud Storage (recommended for production)
Store files in **Google Drive / S3 / GCS** and load via URL. Add credentials to Streamlit Secrets:

1. In Streamlit Cloud dashboard → your app → **⋮ menu → Settings → Secrets**
2. Add:
```toml
[gcs]
bucket = "your-bucket-name"
credentials = "{ ... your service account JSON ... }"
```

---

## 💻 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/dti-discount-optimizer.git
cd dti-discount-optimizer

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

streamlit run streamlit_app.py
```

Open **http://localhost:8501**

---

## 🧠 ML Pipeline

| Step | Detail |
|------|--------|
| **Data Sources** | ecommerce, retail, bigmart, reviews |
| **Sentiment** | VADER + TextBlob (averaged) |
| **Models Trained** | Linear Regression, Random Forest, XGBoost |
| **Winner** | XGBoost (lowest RMSE) |
| **Scaler** | StandardScaler (sklearn) |
| **Output** | Optimised discount % per product |

---

## 📊 Dashboard Features

- **KPI Row** — Products, Sentiment, Discount, Effective Price, Revenue
- **Styled Table** — Gradient coloring, search, CSV export
- **Core Charts** — Discount bar, Price vs Sentiment scatter
- **Advanced Analytics** — Seasonal trends, Festival analysis, Revenue simulation, Correlation heatmap
- **Model Explanation** — XGBoost feature importance

---

*Built with ❤️ using Streamlit · XGBoost · Plotly · VADER*
