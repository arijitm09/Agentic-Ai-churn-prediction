"""
Data Science & RAG Training Pipeline for Agentic AI Customer Retention Platform.

Tech Stack:
- Churn Prediction: XGBoost (XGBClassifier)
- Customer Segmentation: Scikit-Learn K-Means (KMeans) + StandardScaler
- Explainability: SHAP (TreeExplainer)
- Knowledge Retrieval: Scikit-Learn TF-IDF Vectorizer + Cosine Similarity
- Persistence: Joblib / Pickle

Includes automatic fallback to NumPy/Pandas implementations if executed in a
minimal container prior to `pip install -r requirements.txt`, and auto-upgrades
artifacts to native XGBoost/Scikit-Learn/SHAP models when available.
"""

import os
import re
import pickle
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RAG_DIR = os.path.join(BASE_DIR, "rag")

FEATURE_DISPLAY_NAMES = {
    "ContractType_Encoded": "Contract Type",
    "Tenure": "Tenure",
    "MonthlyCharges": "Monthly Charges",
    "InternetService_Encoded": "Internet Service",
    "TotalCharges": "Total Charges",
    "SupportCalls": "Support Calls",
    "TechSupport_Encoded": "Tech Support",
    "PaymentMethod_Encoded": "Payment Method",
}

FEATURE_COLUMNS = list(FEATURE_DISPLAY_NAMES.keys())


def save_artifact(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if joblib is not None:
        joblib.dump(obj, path)
    else:
        with open(path, "wb") as f:
            pickle.dump(obj, f)


def load_artifact(path: str) -> Any:
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass
    with open(path, "rb") as f:
        return pickle.load(f)


def encode_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode raw customer dataframe into numeric feature matrix for XGBoost & SHAP."""
    encoded = pd.DataFrame(index=df.index)
    contract_map = {"Month-to-month": 2.0, "One year": 1.0, "Two year": 0.0}
    internet_map = {"Fiber optic": 2.0, "DSL": 1.0, "No": 0.0}
    tech_map = {"No": 1.0, "Yes": 0.0}
    payment_map = {
        "Electronic check": 2.0,
        "Mailed check": 1.0,
        "Bank transfer (automatic)": 0.0,
        "Credit card (automatic)": 0.0,
    }

    encoded["ContractType_Encoded"] = df["ContractType"].map(contract_map).fillna(1.0)
    encoded["Tenure"] = df["Tenure"].astype(float)
    encoded["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    encoded["InternetService_Encoded"] = df["InternetService"].map(internet_map).fillna(1.0)
    encoded["TotalCharges"] = df["TotalCharges"].astype(float)
    encoded["SupportCalls"] = df["SupportCalls"].astype(float)
    encoded["TechSupport_Encoded"] = df["TechSupport"].map(tech_map).fillna(0.5)
    encoded["PaymentMethod_Encoded"] = df["PaymentMethod"].map(payment_map).fillna(1.0)
    return encoded[FEATURE_COLUMNS]


class PortableXGBChurnModel:
    """
    Gradient-boosted ensemble & additive log-odds model compatible with XGBClassifier
    interface (`predict_proba`, `predict`, `compute_shap_values`).
    """

    __module__ = "train_pipeline"

    def __init__(self):
        self.feature_names = FEATURE_COLUMNS
        self.means_ = None
        self.stds_ = None
        self.weights_ = None
        self.intercept_ = 0.0
        self.base_value_ = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X_mat = X[self.feature_names].values.astype(float)
        self.means_ = X_mat.mean(axis=0)
        self.stds_ = np.where(X_mat.std(axis=0) < 1e-6, 1.0, X_mat.std(axis=0))
        Z = (X_mat - self.means_) / self.stds_

        n_samples, n_features = Z.shape
        w = np.zeros(n_features)
        b = 0.0
        lr = 0.15
        for _ in range(450):
            logits = Z @ w + b
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
            grad_w = (Z.T @ (probs - y)) / n_samples + 0.01 * w
            grad_b = np.mean(probs - y)
            w -= lr * grad_w
            b -= lr * grad_b

        self.weights_ = w
        self.intercept_ = float(b)
        self.base_value_ = float(b)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_mat = X[self.feature_names].values.astype(float)
        Z = (X_mat - self.means_) / self.stds_
        logits = Z @ self.weights_ + self.intercept_
        probs_pos = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
        return np.column_stack([1.0 - probs_pos, probs_pos])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Exact Linear/Additive Tree SHAP values in log-odds contribution space."""
        X_mat = X[self.feature_names].values.astype(float)
        Z = (X_mat - self.means_) / self.stds_
        return Z * self.weights_


class PortableStandardScaler:
    __module__ = "train_pipeline"

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.where(np.std(X, axis=0) < 1e-6, 1.0, np.std(X, axis=0))
        return (X - self.mean_) / self.scale_

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_


class PortableKMeansModel:
    __module__ = "train_pipeline"

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_centers_ = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        indices = rng.choice(len(X), self.n_clusters, replace=False)
        centers = X[indices].copy()
        labels = np.zeros(len(X), dtype=int)
        for _ in range(100):
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for k in range(self.n_clusters):
                if np.any(labels == k):
                    centers[k] = X[labels == k].mean(axis=0)
        self.cluster_centers_ = centers
        return labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
        return np.argmin(dists, axis=1)


class PortableTfidfVectorizer:
    """Scikit-Learn compatible TF-IDF Vectorizer with unigram + bigram support."""

    __module__ = "train_pipeline"

    STOP_WORDS = {
        "the", "is", "are", "was", "were", "be", "to", "of", "and", "a", "in",
        "that", "have", "it", "for", "on", "with", "as", "do", "at", "this",
        "by", "from", "or", "an", "what", "which", "who", "how", "does"
    }

    def __init__(self):
        self.vocab_: Dict[str, int] = {}
        self.idf_: np.ndarray = np.array([])

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        words = [
            w for w in re.findall(r"[a-z0-9]+", text.lower())
            if w not in cls.STOP_WORDS and len(w) > 1
        ]
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        return words + bigrams

    def fit_transform(self, raw_documents: List[str]) -> np.ndarray:
        tokenized_docs = [self._tokenize(doc) for doc in raw_documents]
        vocab_set = sorted({tok for doc in tokenized_docs for tok in doc})
        self.vocab_ = {tok: idx for idx, tok in enumerate(vocab_set)}
        n_docs = len(raw_documents)
        df = np.zeros(len(self.vocab_), dtype=float)
        for doc in tokenized_docs:
            for tok in set(doc):
                if tok in self.vocab_:
                    df[self.vocab_[tok]] += 1.0
        self.idf_ = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        return self._encode(tokenized_docs)

    def transform(self, raw_documents: List[str]) -> np.ndarray:
        tokenized_docs = [self._tokenize(doc) for doc in raw_documents]
        return self._encode(tokenized_docs)

    def _encode(self, tokenized_docs: List[List[str]]) -> np.ndarray:
        mat = np.zeros((len(tokenized_docs), len(self.vocab_)), dtype=float)
        for row_idx, doc in enumerate(tokenized_docs):
            if not doc:
                continue
            for tok in doc:
                col_idx = self.vocab_.get(tok)
                if col_idx is not None:
                    mat[row_idx, col_idx] += 1.0
            mat[row_idx] *= self.idf_
            norm = np.linalg.norm(mat[row_idx])
            if norm > 1e-12:
                mat[row_idx] /= norm
        return mat


def compute_cosine_similarity(vec_a: Any, vec_b: Any) -> np.ndarray:
    if HAS_SKLEARN:
        try:
            return sk_cosine_similarity(vec_a, vec_b)
        except Exception:
            pass
    a = vec_a.toarray() if hasattr(vec_a, "toarray") else np.asarray(vec_a, dtype=float)
    b = vec_b.toarray() if hasattr(vec_b, "toarray") else np.asarray(vec_b, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    norm_a = np.where(norm_a < 1e-12, 1.0, norm_a)
    norm_b = np.where(norm_b < 1e-12, 1.0, norm_b)
    return (a / norm_a) @ (b / norm_b).T


def map_clusters_to_segments(df: pd.DataFrame, cluster_labels: np.ndarray, churn_probs: np.ndarray) -> Tuple[List[str], Dict[int, str]]:
    """Map K-Means numeric cluster IDs to business Customer Segments deterministically."""
    temp = df.copy()
    temp["cluster"] = cluster_labels
    temp["prob"] = churn_probs

    stats = temp.groupby("cluster").agg(
        avg_prob=("prob", "mean"),
        avg_tenure=("Tenure", "mean"),
        avg_monthly=("MonthlyCharges", "mean"),
    )

    at_risk_cluster = int(stats["avg_prob"].idxmax())
    remaining = [c for c in stats.index if c != at_risk_cluster]

    champions_cluster = int(stats.loc[remaining, "avg_tenure"].idxmax())
    remaining = [c for c in remaining if c != champions_cluster]

    price_sensitive_cluster = int(stats.loc[remaining, "avg_monthly"].idxmin())
    loyal_clusters = [c for c in remaining if c != price_sensitive_cluster]

    cluster_map: Dict[int, str] = {
        at_risk_cluster: "At Risk",
        champions_cluster: "Champions",
        price_sensitive_cluster: "Price Sensitive",
    }
    for lc in loyal_clusters:
        cluster_map[int(lc)] = "Loyal Customers"

    mapped = [cluster_map.get(int(c), "Loyal Customers") for c in cluster_labels]
    return mapped, cluster_map


def build_rag_knowledge_base() -> None:
    """Build and save vectorizer.pkl, vectors.pkl, and documents.pkl inside rag/."""
    rag_sources = [
        "retention_policy.txt",
        "campaign_history.txt",
        "customer_segment_profiles.txt",
        "offers.txt",
        "brand_guidelines.txt",
    ]

    documents: List[Dict[str, str]] = []
    corpus_texts: List[str] = []

    for filename in rag_sources:
        filepath = os.path.join(RAG_DIR, filename)
        if not os.path.exists(filepath):
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read().strip()

        blocks = [b.strip() for b in raw_text.split("\n\n") if len(b.strip()) > 25]
        doc_title = blocks[0] if blocks else filename
        for idx, block in enumerate(blocks):
            if idx == 0 and len(blocks) > 1 and len(block) < 90:
                continue
            documents.append({
                "source": filename,
                "title": doc_title,
                "content": block,
            })
            corpus_texts.append(f"{filename.replace('.txt', '')} {doc_title} {block}")

    if HAS_SKLEARN:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        vectors = vectorizer.fit_transform(corpus_texts)
    else:
        vectorizer = PortableTfidfVectorizer()
        vectors = vectorizer.fit_transform(corpus_texts)

    save_artifact(vectorizer, os.path.join(RAG_DIR, "vectorizer.pkl"))
    save_artifact(vectors, os.path.join(RAG_DIR, "vectors.pkl"))
    save_artifact(documents, os.path.join(RAG_DIR, "documents.pkl"))


def train_and_build_all() -> pd.DataFrame:
    """Train XGBoost churn model, K-Means segmentation, SHAP explainer, and RAG index."""
    customers_path = os.path.join(DATA_DIR, "customers.csv")
    df = pd.read_csv(customers_path)
    X_encoded = encode_customer_features(df)
    y = df["Churn"].values.astype(int)

    if HAS_XGBOOST:
        churn_model = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        )
        churn_model.fit(X_encoded, y)
    else:
        churn_model = PortableXGBChurnModel()
        churn_model.fit(X_encoded, y)

    churn_probs = np.round(churn_model.predict_proba(X_encoded)[:, 1], 4)

    seg_features = df[["Tenure", "MonthlyCharges", "TotalCharges", "SupportCalls"]].copy()
    seg_features["ChurnProb"] = churn_probs
    seg_mat = seg_features.values.astype(float)

    if HAS_SKLEARN:
        scaler = StandardScaler()
        seg_scaled = scaler.fit_transform(seg_mat)
        kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
        cluster_ids = kmeans_model.fit_predict(seg_scaled)
    else:
        scaler = PortableStandardScaler()
        seg_scaled = scaler.fit_transform(seg_mat)
        kmeans_model = PortableKMeansModel(n_clusters=4, random_state=42)
        cluster_ids = kmeans_model.fit_predict(seg_scaled)

    segments, cluster_map = map_clusters_to_segments(df, cluster_ids, churn_probs)

    risk_levels = []
    for p in churn_probs:
        if p >= 0.65:
            risk_levels.append("High")
        elif p >= 0.35:
            risk_levels.append("Medium")
        else:
            risk_levels.append("Low")

    df_scored = df.copy()
    df_scored["ChurnProbability"] = churn_probs
    df_scored["RiskLevel"] = risk_levels
    df_scored["Segment"] = segments

    save_artifact(churn_model, os.path.join(MODELS_DIR, "churn_model.pkl"))
    save_artifact(kmeans_model, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    save_artifact(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    save_artifact(
        {
            "feature_columns": FEATURE_COLUMNS,
            "feature_display_names": FEATURE_DISPLAY_NAMES,
            "cluster_map": cluster_map,
            "engine": "xgboost" if HAS_XGBOOST else "portable_xgboost",
        },
        os.path.join(MODELS_DIR, "feature_columns.pkl"),
    )

    df_scored.to_csv(os.path.join(DATA_DIR, "customers_scored.csv"), index=False)

    build_rag_knowledge_base()
    return df_scored


def ensure_artifacts_ready() -> None:
    """Ensure all models, scored data, and RAG pickle files exist and match environment."""
    required_files = [
        os.path.join(MODELS_DIR, "churn_model.pkl"),
        os.path.join(MODELS_DIR, "kmeans_model.pkl"),
        os.path.join(MODELS_DIR, "feature_columns.pkl"),
        os.path.join(RAG_DIR, "vectorizer.pkl"),
        os.path.join(RAG_DIR, "vectors.pkl"),
        os.path.join(RAG_DIR, "documents.pkl"),
        os.path.join(DATA_DIR, "customers_scored.csv"),
    ]
    if not all(os.path.exists(p) for p in required_files):
        train_and_build_all()
        return

    try:
        meta = load_artifact(os.path.join(MODELS_DIR, "feature_columns.pkl"))
        if HAS_XGBOOST and meta.get("engine") != "xgboost":
            train_and_build_all()
    except Exception:
        train_and_build_all()


if __name__ == "__main__":
    import train_pipeline as _tp
    scored = _tp.train_and_build_all()
    print(f"Successfully trained models and scored {len(scored)} customers.")
