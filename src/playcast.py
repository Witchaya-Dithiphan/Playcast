# ======================================
# owners_predict_mlr_fast.py
# Supports Tags, Languages, Publishers, and Categories
# ======================================
import ast
import math
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from time import perf_counter

MODEL_TYPE = "ols"
SHOW_TIMING = True


# --------------------------------------
# Utility Functions
# --------------------------------------
def parse_range_midpoint(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    if " - " in s:
        lo, hi = s.split(" - ", 1)
        lo = int(lo.replace(",", "").strip())
        hi = int(hi.replace(",", "").strip())
        return (lo + hi) / 2
    try:
        return float(s.replace(",", ""))
    except:
        return None


def _to_list_from_any(x):
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return list(x.keys())
    if pd.isna(x):
        return []
    s = str(x).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            return list(v.keys())
        return [s] if s else []
    except:
        for sep in [",", ";", "|", "/"]:
            if sep in s:
                return [t.strip() for t in s.split(sep)]
        return [s] if s else []


def ensure_list_str(x) -> List[str]:
    items = _to_list_from_any(x)
    out = []
    for it in items:
        si = str(it).strip()
        if si and si.lower() not in {"nan", "none", "null", "0"}:
            out.append(si)
    return out


def safe_num(x, default=0.0):
    try:
        return float(str(x).replace(",", "").strip())
    except:
        return default


def safe_int(x, default=0):
    try:
        return int(str(x).replace(",", "").strip())
    except:
        return default


def extract_year(s):
    if pd.isna(s):
        return None
    for tok in str(s).replace(",", " ").split():
        if tok.isdigit() and len(tok) == 4:
            return int(tok)
    return None


# --------------------------------------
# Normalize columns (case-insensitive)
# --------------------------------------
def normalize_columns(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}

    def find_col(name_options):
        for opt in name_options:
            if opt.lower() in cols:
                return cols[opt.lower()]
        return None

    owners_col = find_col(["estimated owners", "owners"])
    price_col = find_col(["price"])
    play_col = find_col(["average playtime forever", "average playtime", "playtime"])
    date_col = find_col(["release date", "release"])
    tags_col = find_col(["tags"])
    langs_col = find_col(["supported languages", "languages"])
    pos_col = find_col(["positive", "positive reviews"])
    neg_col = find_col(["negative", "negative reviews"])
    pub_col = find_col(["publishers", "publisher"])
    cat_col = find_col(["categories", "category"]) 

    nd = pd.DataFrame(index=df.index)
    nd["owners"] = df[owners_col].apply(parse_range_midpoint)
    nd["price_num"] = df[price_col].apply(safe_num)
    nd["avg_playtime"] = df[play_col].fillna(0).apply(safe_num) if play_col else 0
    nd["release_year"] = df[date_col].apply(extract_year).fillna(0).astype(int) if date_col else 0
    nd["tags_list"] = df[tags_col].apply(ensure_list_str) if tags_col else [[] for _ in range(len(df))]
    nd["supported_languages"] = df[langs_col].apply(ensure_list_str) if langs_col else [[] for _ in range(len(df))]
    nd["publishers_list"] = df[pub_col].apply(ensure_list_str) if pub_col else [[] for _ in range(len(df))]
    nd["categories_list"] = df[cat_col].apply(ensure_list_str) if cat_col else [[] for _ in range(len(df))]
    nd["pos_reviews"] = df[pos_col].apply(safe_int) if pos_col else 0
    nd["neg_reviews"] = df[neg_col].apply(safe_int) if neg_col else 0

    for nm in ["Name", "name", "title"]:
        if nm in df.columns:
            nd["name"] = df[nm]
            break

    return nd


# --------------------------------------
# Feature Builder (now includes categories)
# --------------------------------------
def build_features(df_norm: pd.DataFrame, fit: bool, encoders=None):
    for col in ["tags_list", "supported_languages", "publishers_list", "categories_list"]:
        df_norm[col] = df_norm[col].apply(ensure_list_str)

    num = df_norm[["price_num", "avg_playtime", "release_year"]].astype(float).copy()
    pos = df_norm.get("pos_reviews", 0)
    neg = df_norm.get("neg_reviews", 0)
    num["pos_log1p"] = np.log1p(pos)
    num["neg_log1p"] = np.log1p(neg)
    total = (pos + neg).replace(0, np.nan)
    num["pos_ratio"] = (pos / total).fillna(0.0)

    if fit:
        enc = {
            "tag": MultiLabelBinarizer(),
            "lang": MultiLabelBinarizer(),
            "pub": MultiLabelBinarizer(),
            "cat": MultiLabelBinarizer(),  # ✅ NEW
        }

        tag_encoded = pd.DataFrame(
            enc["tag"].fit_transform(df_norm["tags_list"]),
            columns=[f"tag_{t}" for t in enc["tag"].classes_],
            index=df_norm.index,
        )
        lang_encoded = pd.DataFrame(
            enc["lang"].fit_transform(df_norm["supported_languages"]),
            columns=[f"lang_{l}" for l in enc["lang"].classes_],
            index=df_norm.index,
        )
        pub_encoded = pd.DataFrame(
            enc["pub"].fit_transform(df_norm["publishers_list"]),
            columns=[f"pub_{p}" for p in enc["pub"].classes_],
            index=df_norm.index,
        )
        cat_encoded = pd.DataFrame(
            enc["cat"].fit_transform(df_norm["categories_list"]),
            columns=[f"cat_{c}" for c in enc["cat"].classes_],
            index=df_norm.index,
        )
    else:
        enc = encoders
        tag_encoded = pd.DataFrame(
            enc["tag"].transform(df_norm["tags_list"]),
            columns=[f"tag_{t}" for t in enc["tag"].classes_],
            index=df_norm.index,
        )
        lang_encoded = pd.DataFrame(
            enc["lang"].transform(df_norm["supported_languages"]),
            columns=[f"lang_{l}" for l in enc["lang"].classes_],
            index=df_norm.index,
        )
        pub_encoded = pd.DataFrame(
            enc["pub"].transform(df_norm["publishers_list"]),
            columns=[f"pub_{p}" for p in enc["pub"].classes_],
            index=df_norm.index,
        )
        cat_encoded = pd.DataFrame(
            enc["cat"].transform(df_norm["categories_list"]),
            columns=[f"cat_{c}" for c in enc["cat"].classes_],
            index=df_norm.index,
        )

    X = pd.concat([num, tag_encoded, lang_encoded, pub_encoded, cat_encoded], axis=1).fillna(0.0)
    return X, enc


# --------------------------------------
# Model Builder
# --------------------------------------
def build_linear_model(model_type: str):
    if model_type == "ols":
        return LinearRegression()
    if model_type == "ridge":
        return Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("ridge", RidgeCV(alphas=np.logspace(-2, 2, 7)))
        ])
    if model_type == "elasticnet":
        return Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("enet", ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], cv=5, max_iter=10000))
        ])
    raise ValueError("Unknown model type")


# --------------------------------------
# Train & Save
# --------------------------------------
def train_and_save(csv_path: Path, model_path: Path):
    t0 = perf_counter()
    
    # 1. โหลดข้อมูล 
    df = pd.read_csv(csv_path)

    # 2. จัดระเบียบข้อมูล (Normalization) 
    df_norm = normalize_columns(df)
    df_norm = df_norm[pd.notna(df_norm["owners"])].copy()

    # 3. สร้างฟีเจอร์และแปลงข้อมูลเป้าหมาย 
    X, enc = build_features(df_norm, fit=True)
    y = df_norm["owners"].astype(float).apply(math.log1p)

    # 4. สร้างและฝึกสอนโมเดล 
    model = build_linear_model(MODEL_TYPE)
    model.fit(X, y)

    # 5. บรรจุและบันทึกโมเดล
    payload = {"model": model, "encoders": enc, "feature_columns": list(X.columns)}
    joblib.dump(payload, model_path)
    print(f"✅ Saved → {model_path}")
    print(f"⏱ {perf_counter() - t0:.2f}s total")


# --------------------------------------
# Predict (now includes categories)
# --------------------------------------
def predict_new_game(model_bundle_path: Path, price_num: float, release_year: int,
                     tags: List[str], languages: List[str],
                     publishers: Optional[List[str]] = None,
                     categories: Optional[List[str]] = None,
                     avg_playtime: float = 0.0, positive: int = 0, negative: int = 0):
    bundle = joblib.load(model_bundle_path)
    model, enc = bundle["model"], bundle["encoders"]

    df_norm = pd.DataFrame([{
        "price_num": price_num,
        "avg_playtime": avg_playtime,
        "release_year": release_year,
        "tags_list": ensure_list_str(tags),
        "supported_languages": ensure_list_str(languages),
        "publishers_list": ensure_list_str(publishers or []),
        "categories_list": ensure_list_str(categories or []),
        "pos_reviews": positive,
        "neg_reviews": negative,
    }])

    X, _ = build_features(df_norm, fit=False, encoders=enc)
    y_pred = model.predict(X)[0]
    return float(np.expm1(y_pred))


# --------------------------------------
# Find Similar Games (by tags + languages + categories)
# --------------------------------------
def find_similar_games(model_bundle_path: Path, dataset_csv_path: Path,
                       tags: List[str], languages: List[str],
                       categories: Optional[List[str]] = None,
                       top_k: int = 5) -> pd.DataFrame:
    """
    Finds top_k most similar games based on cosine similarity of tags, languages, and categories.
    """
    bundle = joblib.load(model_bundle_path)
    enc = bundle["encoders"]

    raw = pd.read_csv(dataset_csv_path)
    df_norm = normalize_columns(raw)

    if "categories" in [c.lower() for c in raw.columns]:
        df_norm["categories_list"] = raw[[c for c in raw.columns if c.lower() == "categories"][0]].apply(ensure_list_str)
    else:
        df_norm["categories_list"] = [[] for _ in range(len(df_norm))]

    tag_enc, lang_enc, cat_enc = enc["tag"], enc["lang"], enc["cat"]

    tag_features = pd.DataFrame(tag_enc.transform(df_norm["tags_list"]),
                                columns=[f"tag_{t}" for t in tag_enc.classes_], index=df_norm.index)
    lang_features = pd.DataFrame(lang_enc.transform(df_norm["supported_languages"]),
                                 columns=[f"lang_{l}" for l in lang_enc.classes_], index=df_norm.index)
    cat_features = pd.DataFrame(cat_enc.transform(df_norm["categories_list"]),
                                columns=[f"cat_{c}" for c in cat_enc.classes_], index=df_norm.index)

    X_dataset = pd.concat([tag_features, lang_features, cat_features], axis=1)

    sample_df = pd.DataFrame([{
        "tags_list": ensure_list_str(tags),
        "supported_languages": ensure_list_str(languages),
        "categories_list": ensure_list_str(categories or []),
    }])
    tag_sample = pd.DataFrame(tag_enc.transform(sample_df["tags_list"]),
                              columns=[f"tag_{t}" for t in tag_enc.classes_])
    lang_sample = pd.DataFrame(lang_enc.transform(sample_df["supported_languages"]),
                               columns=[f"lang_{l}" for l in lang_enc.classes_])
    cat_sample = pd.DataFrame(cat_enc.transform(sample_df["categories_list"]),
                              columns=[f"cat_{c}" for c in cat_enc.classes_])
    X_sample = pd.concat([tag_sample, lang_sample, cat_sample], axis=1)

    sims = cosine_similarity(X_dataset.values, X_sample.values.reshape(1, -1)).ravel()
    df_norm["_similarity"] = np.round(sims, 5)

    cols = ["name"] if "name" in df_norm.columns else []
    for cand in ["AppID", "appid", "app_id", "index", "id"]:
        if cand in raw.columns:
            df_norm["_id"] = raw[cand]
            cols.insert(0, "_id")
            break
    cols.append("_similarity")
    return df_norm[cols].sort_values("_similarity", ascending=False).head(top_k).reset_index(drop=True)


# --------------------------------------
# Main: ฝึกสอนและบันทึกโมเดลจาก data/games_clean.csv
#   วิธีใช้:  python src/playcast.py
#            python src/playcast.py <csv_path> <model_path>
# --------------------------------------
if __name__ == "__main__":
    import sys

    # คอนโซล Windows (cp1252) เข้ารหัส emoji ไม่ได้ → บังคับเป็น utf-8 กัน crash
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    BASE_DIR = Path(__file__).resolve().parents[1]   # โฟลเดอร์ราก playcast/
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "data" / "games_clean.csv"
    model_path = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE_DIR / "models" / "owners_model_mlr.joblib"

    if not csv_path.exists():
        raise SystemExit(f"❌ ไม่พบไฟล์ข้อมูล: {csv_path}")

    print(f"📂 โหลดข้อมูลจาก: {csv_path}")
    print(f"🧮 โมเดล: {MODEL_TYPE}")
    train_and_save(csv_path, model_path)
