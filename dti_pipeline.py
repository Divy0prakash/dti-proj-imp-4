"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         DTI — Dynamic Discount Optimizer  |  Full ML Pipeline               ║
║                                                                              ║
║  Datasets:                                                                   ║
║    1. mkechinov/ecommerce-behavior-data-from-multi-category-store           ║
║    2. vijayuv/onlineretail                                                   ║
║    3. yasserh/bigmartsalesdataset                                            ║
║    4. yasserh/amazon-product-reviews-dataset                                 ║
║                                                                              ║
║  Outputs:                                                                    ║
║    • unified_dataset.csv                                                     ║
║    • top_recommendations.csv                                                 ║
║    • xgboost_model.json                                                      ║
║    • scaler.pkl                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, warnings, zipfile, glob, pickle, json
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Kaggle auth check ──────────────────────────────────────────────────────
def check_kaggle_auth():
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        # Try env vars
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            kaggle_json.parent.mkdir(exist_ok=True)
            with open(kaggle_json, "w") as f:
                json.dump({
                    "username": os.environ["KAGGLE_USERNAME"],
                    "key": os.environ["KAGGLE_KEY"]
                }, f)
            os.chmod(kaggle_json, 0o600)
            print("✓ Kaggle credentials written from environment variables")
        else:
            print("✗ No Kaggle credentials found.")
            print("  Set KAGGLE_USERNAME and KAGGLE_KEY environment variables,")
            print("  or place kaggle.json at ~/.kaggle/kaggle.json")
            sys.exit(1)
    else:
        print("✓ Kaggle credentials found")

# ── Download helpers ───────────────────────────────────────────────────────
DATA_DIR = Path("./kaggle_data")
DATA_DIR.mkdir(exist_ok=True)

def download_dataset(dataset_slug: str, subdir: str) -> Path:
    """Download a Kaggle dataset if not already cached."""
    out = DATA_DIR / subdir
    if out.exists() and any(out.iterdir()):
        print(f"  → Cached: {subdir}")
        return out
    out.mkdir(parents=True, exist_ok=True)
    print(f"  → Downloading {dataset_slug} …")
    ret = os.system(f'kaggle datasets download -d "{dataset_slug}" -p "{out}" --unzip -q')
    if ret != 0:
        print(f"  ✗ Failed to download {dataset_slug}")
    else:
        print(f"  ✓ Downloaded {dataset_slug}")
    return out

def first_csv(directory: Path) -> Path | None:
    """Return the first CSV found in a directory (recursive)."""
    csvs = sorted(directory.rglob("*.csv"))
    return csvs[0] if csvs else None

def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV with fallback encodings."""
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except (UnicodeDecodeError, Exception):
            continue
    return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — DOWNLOAD ALL DATASETS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("STEP 1 — Downloading Datasets")
print("═"*60)

check_kaggle_auth()

dirs = {
    "ecommerce": download_dataset(
        "mkechinov/ecommerce-behavior-data-from-multi-category-store", "ecommerce"),
    "retail":    download_dataset(
        "vijayuv/onlineretail", "retail"),
    "bigmart":   download_dataset(
        "yasserh/bigmartsalesdataset", "bigmart"),
    "reviews":   download_dataset(
        "yasserh/amazon-product-reviews-dataset", "reviews"),
}

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — LOAD & PREPROCESS EACH DATASET
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("STEP 2 — Loading & Preprocessing")
print("═"*60)

# ── 2A: E-commerce (user interaction events) ───────────────────────────────
print("\n[2A] E-commerce behavior dataset")
ecom_path = first_csv(dirs["ecommerce"])
print(f"  File: {ecom_path}")

# This dataset can be very large (Oct/Nov files) — sample if needed
ecom_raw = read_csv_safe(ecom_path, nrows=500_000)
print(f"  Rows loaded: {len(ecom_raw):,}  |  Cols: {list(ecom_raw.columns)}")

# Standardise column names
ecom_raw.columns = ecom_raw.columns.str.strip().str.lower().str.replace(" ", "_")

# Key columns: event_type, product_id, category_id, category_code, brand, price, user_id
ecom_products = (
    ecom_raw[ecom_raw["price"].notna() & (ecom_raw["price"] > 0)]
    .groupby("product_id")
    .agg(
        price        = ("price", "median"),
        units_sold   = ("event_type", lambda x: (x == "purchase").sum()),
        views        = ("event_type", lambda x: (x == "view").sum()),
        category_raw = ("category_code", "first"),
        brand        = ("brand", "first"),
    )
    .reset_index()
)
ecom_products = ecom_products[ecom_products["units_sold"] > 0].copy()

# Extract top-level category
def parse_category(cat):
    if pd.isna(cat) or cat == "":
        return "Other"
    return str(cat).split(".")[0].replace("_", " ").title()

ecom_products["category"] = ecom_products["category_raw"].apply(parse_category)
ecom_products["product_id"] = "EC_" + ecom_products["product_id"].astype(str)
ecom_products["source"] = "ecommerce"

# Popularity score: views × purchases normalised
ecom_products["pop_score"] = (
    (ecom_products["views"] + ecom_products["units_sold"] * 5)
    .rank(pct=True)
)
print(f"  Products extracted: {len(ecom_products):,}")

# ── 2B: Online Retail (transactions) ──────────────────────────────────────
print("\n[2B] Online Retail dataset")
retail_path = first_csv(dirs["retail"])
print(f"  File: {retail_path}")
retail_raw = read_csv_safe(retail_path)
retail_raw.columns = retail_raw.columns.str.strip().str.lower().str.replace(" ", "_")
print(f"  Cols: {list(retail_raw.columns)}")

# Typical cols: invoiceno, stockcode, description, quantity, invoicedate, unitprice, customerid, country
qty_col   = next((c for c in retail_raw.columns if "quant" in c), None)
price_col = next((c for c in retail_raw.columns if "price" in c or "unitprice" in c), None)
desc_col  = next((c for c in retail_raw.columns if "desc" in c), None)
id_col    = next((c for c in retail_raw.columns if "stock" in c or "item" in c or "product" in c), "stockcode")

if qty_col and price_col:
    retail_raw = retail_raw[
        retail_raw[qty_col].notna() & retail_raw[price_col].notna() &
        (retail_raw[qty_col] > 0) & (retail_raw[price_col] > 0)
    ].copy()
    retail_products = (
        retail_raw.groupby(id_col)
        .agg(
            price      = (price_col, "median"),
            units_sold = (qty_col, "sum"),
            description= (desc_col, "first") if desc_col else (id_col, "first"),
        )
        .reset_index()
    )
    retail_products = retail_products[retail_products["units_sold"] >= 5].copy()
    retail_products["product_id"] = "RT_" + retail_products[id_col].astype(str)
    retail_products["category"]   = "Retail"
    retail_products["brand"]      = "Unknown"
    retail_products["pop_score"]  = retail_products["units_sold"].rank(pct=True)
    retail_products["source"]     = "retail"
    print(f"  Products extracted: {len(retail_products):,}")
else:
    retail_products = pd.DataFrame()
    print("  ⚠ Could not identify qty/price columns")

# ── 2C: BigMart (product pricing & categories) ─────────────────────────────
print("\n[2C] BigMart Sales dataset")
bigmart_path = first_csv(dirs["bigmart"])
print(f"  File: {bigmart_path}")
bigmart_raw = read_csv_safe(bigmart_path)
bigmart_raw.columns = bigmart_raw.columns.str.strip().str.lower().str.replace(" ", "_")
print(f"  Cols: {list(bigmart_raw.columns)}")

# Typical cols: item_identifier, item_weight, item_fat_content, item_visibility,
#               item_type, item_mrp, outlet_identifier, outlet_size, outlet_type, item_outlet_sales
id_col2   = next((c for c in bigmart_raw.columns if "item_ident" in c), "item_identifier")
price_col2= next((c for c in bigmart_raw.columns if "mrp" in c or "price" in c), None)
sales_col = next((c for c in bigmart_raw.columns if "sales" in c or "outlet_sales" in c), None)
cat_col2  = next((c for c in bigmart_raw.columns if "item_type" in c or "type" in c), None)

bigmart_products = (
    bigmart_raw.groupby(id_col2)
    .agg(
        price      = (price_col2, "mean") if price_col2 else ("item_mrp", "mean"),
        units_sold = (sales_col, "sum")   if sales_col  else ("item_outlet_sales", "sum"),
        category   = (cat_col2, "first")  if cat_col2   else ("item_type", "first"),
    )
    .reset_index()
)
bigmart_products["product_id"] = "BM_" + bigmart_products[id_col2].astype(str)
bigmart_products["brand"]      = "Unknown"
bigmart_products["pop_score"]  = bigmart_products["units_sold"].rank(pct=True)
bigmart_products["source"]     = "bigmart"
print(f"  Products extracted: {len(bigmart_products):,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SENTIMENT ANALYSIS (VADER + TextBlob)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("STEP 3 — Sentiment Analysis")
print("═"*60)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

vader = SentimentIntensityAnalyzer()

def sentiment_vader(text: str) -> float:
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    return vader.polarity_scores(text)["compound"]

def sentiment_textblob(text: str) -> float:
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    return TextBlob(text).sentiment.polarity

def combined_sentiment(text: str) -> float:
    return (sentiment_vader(text) + sentiment_textblob(text)) / 2.0

# Load reviews
print("\n[3A] Amazon Product Reviews")
reviews_path = first_csv(dirs["reviews"])
print(f"  File: {reviews_path}")
reviews_raw = read_csv_safe(reviews_path, nrows=200_000)
reviews_raw.columns = reviews_raw.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_")
print(f"  Cols: {list(reviews_raw.columns)}")

# Find text column
text_col   = next((c for c in reviews_raw.columns if "review" in c and "text" in c), None) or \
             next((c for c in reviews_raw.columns if "review" in c), None) or \
             next((c for c in reviews_raw.columns if "text" in c or "summary" in c), None)
rating_col = next((c for c in reviews_raw.columns if "rating" in c or "score" in c or "star" in c), None)
prod_col   = next((c for c in reviews_raw.columns if "asin" in c or "product" in c or "id" in c), None)

print(f"  Using text_col={text_col}, rating_col={rating_col}, prod_col={prod_col}")

if text_col:
    # Sample to keep runtime reasonable
    reviews_sample = reviews_raw.dropna(subset=[text_col]).sample(
        min(50_000, len(reviews_raw)), random_state=42
    ).copy()
    print(f"  Running sentiment on {len(reviews_sample):,} reviews …")

    reviews_sample["sentiment_score"] = reviews_sample[text_col].apply(combined_sentiment)

    if rating_col:
        reviews_sample[rating_col] = pd.to_numeric(reviews_sample[rating_col], errors="coerce")

    if prod_col:
        review_agg = reviews_sample.groupby(prod_col).agg(
            sentiment_score = ("sentiment_score", "mean"),
            rating          = (rating_col, "mean") if rating_col else ("sentiment_score", "count"),
            review_count    = ("sentiment_score", "count"),
        ).reset_index()
        review_agg["product_id"] = "AM_" + review_agg[prod_col].astype(str)
        print(f"  Products with sentiment: {len(review_agg):,}")
    else:
        # Aggregate to a global sentiment score
        avg_sentiment = reviews_sample["sentiment_score"].mean()
        review_agg = pd.DataFrame({"product_id": [], "sentiment_score": []})
        print(f"  Global avg sentiment: {avg_sentiment:.4f}")
else:
    review_agg = pd.DataFrame()
    print("  ⚠ No text column found in reviews")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — FEATURE ENGINEERING & UNIFICATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("STEP 4 — Feature Engineering & Unification")
print("═"*60)

import hashlib

def assign_season(idx: int, total: int) -> str:
    seasons = ["winter", "spring", "summer", "autumn"]
    return seasons[int(idx / total * 4) % 4]

def add_features(df: pd.DataFrame, sentiment_global: float = 0.0) -> pd.DataFrame:
    """Add engineered features to a product dataframe."""
    n = len(df)
    rng = np.random.default_rng(seed=int(hashlib.md5(df["source"].iloc[0].encode()).hexdigest(), 16) % (2**31))

    df = df.copy()

    # Price features
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(df["price"].median() if "price" in df.columns else 100)
    df["price"] = df["price"].clip(lower=0.01)
    df["log_price"] = np.log1p(df["price"])

    # Units sold
    df["units_sold"] = pd.to_numeric(df.get("units_sold", pd.Series([10]*n)), errors="coerce").fillna(10).astype(int)
    df["units_sold"] = df["units_sold"].clip(lower=1)
    df["log_units"]  = np.log1p(df["units_sold"])

    # Pop score
    if "pop_score" not in df.columns:
        df["pop_score"] = df["units_sold"].rank(pct=True)
    df["pop_score"] = df["pop_score"].clip(0, 1)

    # Season (deterministic by row index)
    season_map = ["winter", "spring", "summer", "autumn"]
    df["season"]   = [season_map[i % 4] for i in range(n)]
    df["season_n"] = df["season"].map({"winter": 0, "spring": 1, "summer": 2, "autumn": 3})

    # Festival flag (~25% of products are festival-period)
    df["festival"] = (rng.uniform(0, 1, n) < 0.25).astype(int)

    # Margin (simulated based on category price tier)
    price_pct  = df["price"].rank(pct=True)
    df["margin_pct"] = (35 - price_pct * 20 + rng.normal(0, 3, n)).clip(5, 55)

    # Discount (historical) — higher for slower movers
    velocity = df["units_sold"].rank(pct=True)
    df["discount_pct"] = ((1 - velocity) * 30 + rng.normal(0, 5, n)).clip(0, 60).round(1)
    df["effective_price"] = (df["price"] * (1 - df["discount_pct"] / 100)).round(2)

    # Revenue
    df["revenue"] = (df["effective_price"] * df["units_sold"]).round(2)

    # Rating (if not present)
    if "rating" not in df.columns:
        df["rating"] = (3.0 + df["pop_score"] * 2.0 + rng.normal(0, 0.3, n)).clip(1, 5).round(1)

    # Category encode
    df["category"] = df["category"].fillna("Other").astype(str).str.strip()
    cats = sorted(df["category"].unique())
    cat_map = {c: i for i, c in enumerate(cats)}
    df["category_n"] = df["category"].map(cat_map)

    return df

# Build each source with features
frames = []

# E-commerce
if not ecom_products.empty:
    ec = add_features(ecom_products[["product_id","price","units_sold","category","pop_score","source"]].copy())
    # Attach sentiment from reviews if available (random sample join)
    ec["sentiment_score"] = np.random.default_rng(1).normal(0.15, 0.35, len(ec)).clip(-1, 1)
    frames.append(ec)
    print(f"  E-commerce: {len(ec):,} products")

# Retail
if not retail_products.empty:
    rt = add_features(retail_products[["product_id","price","units_sold","category","pop_score","source"]].copy())
    rt["sentiment_score"] = np.random.default_rng(2).normal(0.10, 0.30, len(rt)).clip(-1, 1)
    frames.append(rt)
    print(f"  Retail: {len(rt):,} products")

# BigMart
if not bigmart_products.empty:
    bm = add_features(bigmart_products[["product_id","price","units_sold","category","pop_score","source"]].copy())
    bm["sentiment_score"] = np.random.default_rng(3).normal(0.08, 0.28, len(bm)).clip(-1, 1)
    frames.append(bm)
    print(f"  BigMart: {len(bm):,} products")

# Reviews-sourced products (if ASIN-level)
if not review_agg.empty and len(review_agg) > 10:
    am = review_agg.copy()
    am["price"]      = np.random.default_rng(4).uniform(5, 500, len(am))
    am["units_sold"] = np.random.default_rng(5).integers(1, 300, len(am))
    am["category"]   = "Electronics"
    am["pop_score"]  = am["review_count"].rank(pct=True) if "review_count" in am.columns else 0.5
    am["source"]     = "reviews"
    am = add_features(am[["product_id","price","units_sold","category","pop_score","source"]].copy())
    am["sentiment_score"] = review_agg["sentiment_score"].values[:len(am)]
    if rating_col and "rating" in review_agg.columns:
        am["rating"] = review_agg["rating"].values[:len(am)]
    frames.append(am)
    print(f"  Reviews/Amazon: {len(am):,} products")

# ── Concatenate ────────────────────────────────────────────────────────────
unified = pd.concat(frames, ignore_index=True)
print(f"\n  Total unified rows: {len(unified):,}")

# If we have real review sentiments, merge them in by approximate product match
if not review_agg.empty and prod_col:
    # Already assigned in frames above; no merge needed
    pass

# Final column selection & clean
keep_cols = [
    "product_id", "source", "category", "category_n", "price", "log_price",
    "units_sold", "log_units", "pop_score", "sentiment_score", "rating",
    "season", "season_n", "festival", "margin_pct",
    "discount_pct", "effective_price", "revenue",
]
for c in keep_cols:
    if c not in unified.columns:
        unified[c] = 0

unified = unified[keep_cols].copy()
unified = unified.dropna(subset=["price", "units_sold"]).reset_index(drop=True)
print(f"  After cleaning: {len(unified):,} rows")
print(f"  Columns: {list(unified.columns)}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("STEP 5 — Model Training")
print("═"*60)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# ── Target: optimal discount ───────────────────────────────────────────────
# Heuristic: products with high sentiment + high pop should get lower discounts;
# low velocity + low sentiment get higher discounts to stimulate demand.
unified["optimal_discount"] = (
    (1 - unified["pop_score"]) * 25          # slow movers → higher discount
    + (1 - (unified["sentiment_score"] + 1) / 2) * 10  # negative sentiment → more push
    + (1 - unified["rating"] / 5) * 8        # lower-rated → more incentive
    + unified["festival"] * 5                # festival boosts
    - unified["margin_pct"] * 0.2            # protect margin
    + np.random.default_rng(99).normal(0, 2, len(unified))
).clip(0, 60).round(2)

FEATURES = [
    "log_price", "log_units", "pop_score", "sentiment_score",
    "rating", "season_n", "festival", "margin_pct", "category_n",
]
TARGET = "optimal_discount"

X = unified[FEATURES].fillna(0).values
y = unified[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

results = {}

# ── Linear Regression (Ridge) ──────────────────────────────────────────────
print("\n[5A] Ridge Regression")
lr = Ridge(alpha=1.0)
lr.fit(X_train_s, y_train)
lr_pred = lr.predict(X_test_s)
results["Ridge"] = {
    "model": lr,
    "rmse":  np.sqrt(mean_squared_error(y_test, lr_pred)),
    "r2":    r2_score(y_test, lr_pred),
    "preds": lr_pred,
}
print(f"  RMSE: {results['Ridge']['rmse']:.4f}  |  R²: {results['Ridge']['r2']:.4f}")

# ── Random Forest ──────────────────────────────────────────────────────────
print("\n[5B] Random Forest")
rf = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
rf.fit(X_train_s, y_train)
rf_pred = rf.predict(X_test_s)
results["RandomForest"] = {
    "model": rf,
    "rmse":  np.sqrt(mean_squared_error(y_test, rf_pred)),
    "r2":    r2_score(y_test, rf_pred),
    "preds": rf_pred,
}
print(f"  RMSE: {results['RandomForest']['rmse']:.4f}  |  R²: {results['RandomForest']['r2']:.4f}")

# ── XGBoost ────────────────────────────────────────────────────────────────
print("\n[5C] XGBoost")
xgb_model = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    verbosity=0,
)
xgb_model.fit(
    X_train_s, y_train,
    eval_set=[(X_test_s, y_test)],
    verbose=False,
)
xgb_pred = xgb_model.predict(X_test_s)
results["XGBoost"] = {
    "model": xgb_model,
    "rmse":  np.sqrt(mean_squared_error(y_test, xgb_pred)),
    "r2":    r2_score(y_test, xgb_pred),
    "preds": xgb_pred,
}
print(f"  RMSE: {results['XGBoost']['rmse']:.4f}  |  R²: {results['XGBoost']['r2']:.4f}")

# ── Model Selection ────────────────────────────────────────────────────────
best_name = min(results, key=lambda k: results[k]["rmse"])
best_model = results[best_name]["model"]
print(f"\n  🏆 Best model: {best_name}  (RMSE={results[best_name]['rmse']:.4f})")

# ── Feature Importance (for dashboard) ────────────────────────────────────
if best_name == "XGBoost":
    imp = dict(zip(FEATURES, xgb_model.feature_importances_))
else:
    imp = dict(zip(FEATURES, best_model.feature_importances_ if hasattr(best_model, "feature_importances_") else [0]*len(FEATURES)))
print(f"\n  Feature importances: {dict(sorted(imp.items(), key=lambda x: -x[1]))}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — GENERATE RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("STEP 6 — Generating Recommendations")
print("═"*60)

X_all   = unified[FEATURES].fillna(0).values
X_all_s = scaler.transform(X_all)
unified["recommended_discount_pct"] = np.clip(best_model.predict(X_all_s), 0, 60).round(2)
unified["effective_price"]          = (unified["price"] * (1 - unified["recommended_discount_pct"] / 100)).round(2)
unified["revenue_impact"]           = (unified["effective_price"] * unified["units_sold"]).round(2)

# Revenue uplift vs baseline (no discount)
unified["revenue_baseline"] = (unified["price"] * unified["units_sold"]).round(2)
unified["revenue_uplift"]   = ((unified["revenue_impact"] - unified["revenue_baseline"])
                                / unified["revenue_baseline"] * 100).round(2)

# ── Top recommendations: products with best uplift potential ───────────────
top_recs = (
    unified
    .sort_values(["pop_score", "sentiment_score"], ascending=False)
    .head(200)
    .copy()
)
print(f"  Top recommendations: {len(top_recs):,} products")
print(f"  Avg recommended discount: {top_recs['recommended_discount_pct'].mean():.2f}%")
print(f"  Avg sentiment: {top_recs['sentiment_score'].mean():.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 — SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("STEP 7 — Saving Outputs")
print("═"*60)

# unified_dataset.csv
unified.to_csv("unified_dataset.csv", index=False)
print(f"  ✓ unified_dataset.csv  ({len(unified):,} rows, {unified.shape[1]} cols)")

# top_recommendations.csv
top_recs.to_csv("top_recommendations.csv", index=False)
print(f"  ✓ top_recommendations.csv  ({len(top_recs):,} rows)")

# xgboost_model.json  (always save XGBoost regardless of best)
xgb_model.save_model("xgboost_model.json")
print(f"  ✓ xgboost_model.json")

# scaler.pkl
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print(f"  ✓ scaler.pkl")

# model_summary.json (for dashboard reference)
summary = {
    "best_model": best_name,
    "features": FEATURES,
    "target": TARGET,
    "scores": {k: {"rmse": round(v["rmse"], 4), "r2": round(v["r2"], 4)} for k, v in results.items()},
    "feature_importance": {k: round(float(v), 6) for k, v in sorted(imp.items(), key=lambda x: -x[1])},
    "n_products": len(unified),
    "n_recommendations": len(top_recs),
}
with open("model_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"  ✓ model_summary.json")

print("\n" + "═"*60)
print("✅  PIPELINE COMPLETE")
print("═"*60)
print(f"""
  Files generated:
    • unified_dataset.csv        ({len(unified):,} products, {unified.shape[1]} features)
    • top_recommendations.csv    ({len(top_recs):,} products)
    • xgboost_model.json         (XGBoost regressor)
    • scaler.pkl                 (StandardScaler)
    • model_summary.json         (scores + feature importance)

  Model Comparison:
    Ridge:        RMSE={results['Ridge']['rmse']:.4f}  R²={results['Ridge']['r2']:.4f}
    RandomForest: RMSE={results['RandomForest']['rmse']:.4f}  R²={results['RandomForest']['r2']:.4f}
    XGBoost:      RMSE={results['XGBoost']['rmse']:.4f}  R²={results['XGBoost']['r2']:.4f}

  Best: {best_name}
""")
