"""
Campaign Simulator Agent
Simulates retention campaign outcomes using historical campaign data stored in
`data/campaign_history.csv` (and root `campaign_history.csv`).

Columns:
- CampaignID
- CampaignName
- Segment
- Offer
- Channel
- ResponseRate
- RetentionImprovement
"""

import os
from typing import Dict, Any, List
import pandas as pd

from train_pipeline import BASE_DIR, DATA_DIR


class CampaignSimulator:
    """Specialized Agent for Campaign Outcome Simulation & Historical Benchmarking."""

    def __init__(self) -> None:
        primary_path = os.path.join(DATA_DIR, "campaign_history.csv")
        fallback_path = os.path.join(BASE_DIR, "campaign_history.csv")
        self.csv_path = primary_path if os.path.exists(primary_path) else fallback_path
        self.df_history = pd.read_csv(self.csv_path)

    def get_available_segments(self) -> List[str]:
        return list(self.df_history["Segment"].unique())

    def get_available_offers(self) -> List[str]:
        return list(self.df_history["Offer"].unique())

    def get_all_campaigns(self) -> pd.DataFrame:
        return self.df_history.copy()

    def simulate(self, customer_segment: str, offer: str) -> Dict[str, Any]:
        """
        User inputs:
          - Customer Segment
          - Offer
        System outputs:
          - Matching Historical Campaign
          - Retention Improvement
          - Response Rate
          - Recommended Channel
          - Historical Campaign Insight
        """
        df = self.df_history
        exact_match = df[(df["Segment"] == customer_segment) & (df["Offer"] == offer)]

        if not exact_match.empty:
            best_row = exact_match.sort_values(by="RetentionImprovement", ascending=False).iloc[0]
            match_type = "Exact Historical Match"
        else:
            seg_matches = df[df["Segment"] == customer_segment]
            offer_matches = df[df["Offer"] == offer]
            if not seg_matches.empty:
                best_row = seg_matches.sort_values(by="RetentionImprovement", ascending=False).iloc[0]
                match_type = "Segment Best-Performer Benchmark"
            elif not offer_matches.empty:
                best_row = offer_matches.sort_values(by="RetentionImprovement", ascending=False).iloc[0]
                match_type = "Offer Cross-Segment Benchmark"
            else:
                best_row = df.sort_values(by="RetentionImprovement", ascending=False).iloc[0]
                match_type = "Global Top Benchmark"

        campaign_id = str(best_row["CampaignID"])
        campaign_name = str(best_row["CampaignName"])
        channel = str(best_row["Channel"])
        response_rate = float(best_row["ResponseRate"])
        retention_improvement = float(best_row["RetentionImprovement"])

        seg_df = df[df["Segment"] == customer_segment]
        avg_seg_ret = float(seg_df["RetentionImprovement"].mean()) if not seg_df.empty else retention_improvement
        avg_seg_resp = float(seg_df["ResponseRate"].mean()) if not seg_df.empty else response_rate
        delta_ret = round(retention_improvement - avg_seg_ret, 1)
        delta_resp = round(response_rate - avg_seg_resp, 1)

        insight = (
            f"Deploying **{offer}** to the **{customer_segment}** segment matches historical campaign "
            f"**{campaign_name} ({campaign_id})** via **{channel}**. Historically, this configuration achieved a "
            f"**{response_rate:.1f}% Response Rate** ({delta_resp:+.1f}% vs. segment average) and a "
            f"**{retention_improvement:.1f}% Retention Improvement** ({delta_ret:+.1f}% vs. segment average)."
        )

        return {
            "customer_segment": customer_segment,
            "selected_offer": offer,
            "campaign_id": campaign_id,
            "matching_campaign": f"{campaign_name} ({campaign_id})",
            "campaign_name": campaign_name,
            "match_type": match_type,
            "retention_improvement": retention_improvement,
            "response_rate": response_rate,
            "recommended_channel": channel,
            "delta_retention_vs_segment": delta_ret,
            "delta_response_vs_segment": delta_resp,
            "historical_insight": insight,
            "segment_benchmarks": seg_df.to_dict(orient="records"),
        }
