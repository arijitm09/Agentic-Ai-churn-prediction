"""
Explainability Agent (SHAP Explainable AI)
Computes feature-level SHAP values for a selected customer, identifies Top Churn Drivers,
and produces the SHAP Table and SHAP Feature Importance Bar Chart data.
"""

import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from train_pipeline import (
    MODELS_DIR,
    FEATURE_COLUMNS,
    FEATURE_DISPLAY_NAMES,
    encode_customer_features,
    load_artifact,
    HAS_SHAP,
)

if HAS_SHAP:
    import shap


class ExplainabilityAgent:
    """Specialized Agent for XGBoost Churn Prediction & SHAP Explainability."""

    RAW_FEATURE_KEY_MAP = {
        "ContractType_Encoded": "ContractType",
        "Tenure": "Tenure",
        "MonthlyCharges": "MonthlyCharges",
        "InternetService_Encoded": "InternetService",
        "TotalCharges": "TotalCharges",
        "SupportCalls": "SupportCalls",
        "TechSupport_Encoded": "TechSupport",
        "PaymentMethod_Encoded": "PaymentMethod",
    }

    def __init__(self) -> None:
        self.model_path = os.path.join(MODELS_DIR, "churn_model.pkl")
        self.churn_model = load_artifact(self.model_path)
        self._explainer = None
        if HAS_SHAP and not hasattr(self.churn_model, "compute_shap_values"):
            try:
                self._explainer = shap.TreeExplainer(self.churn_model)
            except Exception:
                self._explainer = None

    def _format_customer_value(self, raw_col: str, val: Any) -> str:
        if raw_col in ("MonthlyCharges", "TotalCharges"):
            return f"${float(val):,.2f}"
        if raw_col == "Tenure":
            return f"{int(val)} months"
        if raw_col == "SupportCalls":
            return f"{int(val)} calls"
        return str(val)

    def explain_customer(self, customer_row: pd.Series) -> Dict[str, Any]:
        """
        Compute SHAP feature attributions and return:
        - Churn Probability
        - Risk Level
        - Segment
        - Top Churn Drivers
        - SHAP Table (list of dicts / DataFrame-ready)
        """
        single_df = pd.DataFrame([customer_row])
        X_encoded = encode_customer_features(single_df)

        # Predict probability
        churn_prob = float(self.churn_model.predict_proba(X_encoded)[0, 1])
        if "ChurnProbability" in customer_row and pd.notna(customer_row["ChurnProbability"]):
            churn_prob = float(customer_row["ChurnProbability"])

        if churn_prob >= 0.65:
            risk_level = "High"
        elif churn_prob >= 0.35:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        segment = str(customer_row.get("Segment", "At Risk"))

        # Compute SHAP values
        shap_vals = None
        if self._explainer is not None:
            try:
                raw_shap = self._explainer.shap_values(X_encoded)
                if isinstance(raw_shap, list):
                    shap_vals = np.asarray(raw_shap[1])[0]
                else:
                    shap_vals = np.asarray(raw_shap)[0]
            except Exception:
                shap_vals = None

        if shap_vals is None:
            if hasattr(self.churn_model, "compute_shap_values"):
                shap_vals = self.churn_model.compute_shap_values(X_encoded)[0]
            else:
                shap_vals = np.zeros(len(FEATURE_COLUMNS), dtype=float)

        shap_records: List[Dict[str, Any]] = []
        for idx, feat_col in enumerate(FEATURE_COLUMNS):
            display_name = FEATURE_DISPLAY_NAMES[feat_col]
            raw_col = self.RAW_FEATURE_KEY_MAP[feat_col]
            raw_val = customer_row.get(raw_col, "N/A")
            formatted_val = self._format_customer_value(raw_col, raw_val)
            s_val = round(float(shap_vals[idx]), 4)
            abs_val = round(abs(s_val), 4)
            direction = "Increases Churn Risk (+)" if s_val >= 0 else "Reduces Churn Risk (-)"

            shap_records.append({
                "Feature": display_name,
                "CustomerValue": formatted_val,
                "SHAPValue": s_val,
                "AbsoluteImportance": abs_val,
                "ImpactDirection": direction,
            })

        # Sort by absolute SHAP importance descending
        shap_records.sort(key=lambda r: r["AbsoluteImportance"], reverse=True)

        # Identify top positive churn drivers (or top overall drivers if low risk)
        positive_drivers = [r for r in shap_records if r["SHAPValue"] > 0]
        top_drivers_source = positive_drivers[:5] if len(positive_drivers) >= 3 else shap_records[:5]
        top_churn_drivers = [
            {
                "feature": r["Feature"],
                "value": r["CustomerValue"],
                "shap_value": r["SHAPValue"],
                "direction": r["ImpactDirection"],
                "summary": f"{r['Feature']} ({r['CustomerValue']}) -> SHAP {r['SHAPValue']:+.3f} ({r['ImpactDirection']})",
            }
            for r in top_drivers_source
        ]

        return {
            "customer_id": str(customer_row.get("CustomerID", "")),
            "customer_name": str(customer_row.get("CustomerName", "")),
            "churn_probability": round(churn_prob, 4),
            "churn_probability_pct": round(churn_prob * 100, 1),
            "risk_level": risk_level,
            "segment": segment,
            "top_churn_drivers": top_churn_drivers,
            "shap_table": shap_records,
        }
