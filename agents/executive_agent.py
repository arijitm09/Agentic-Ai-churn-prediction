"""
Executive Agent (Centralized Multi-Agent Orchestrator)

Architecture Mandate:
- The Executive Agent must call:
  1. Explainability Agent
  2. Retention Agent
  3. Marketing Agent
  4. Knowledge Agent
  (Plus Campaign Simulator and Executive Report Agent)
- The Streamlit UI interacts ONLY with Executive Agent.
- Other agents are never called directly from the UI.
- Future LLM replacement in Knowledge Agent (Gemini API, TinyLlama, Llama 3, Azure OpenAI)
  works seamlessly through Executive Agent without changing the UI.
"""

import os
from typing import Dict, Any, List, Optional
import pandas as pd

from train_pipeline import DATA_DIR, ensure_artifacts_ready
from agents.explainability_agent import ExplainabilityAgent
from agents.retention_agent import RetentionAgent
from agents.marketing_agent import MarketingAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.campaign_simulator import CampaignSimulator
from agents.executive_report_agent import ExecutiveReportAgent


class ExecutiveAgent:
    """
    Centralized Orchestrator Agent for the Agentic AI Customer Retention Platform.
    Coordinates the end-to-end agentic pipeline:
    Predict -> Explain -> Segment -> Recommend -> Generate Campaign ->
    Retrieve Knowledge -> Simulate Campaign -> Generate Executive Report
    """

    def __init__(self, llm_provider: Optional[str] = None) -> None:
        ensure_artifacts_ready()
        self.customers_path = os.path.join(DATA_DIR, "customers_scored.csv")
        self.df_customers = pd.read_csv(self.customers_path)

        # Initialize the specialized subordinate agents
        self.explainability_agent = ExplainabilityAgent()
        self.retention_agent = RetentionAgent()
        self.marketing_agent = MarketingAgent()
        self.knowledge_agent = KnowledgeAgent(llm_provider=llm_provider)
        self.campaign_simulator = CampaignSimulator()
        self.report_agent = ExecutiveReportAgent()

    def set_knowledge_llm_provider(self, provider_key: str) -> None:
        """Allow seamless switching between TF-IDF, Gemini API, TinyLlama, Llama 3, and Azure OpenAI."""
        self.knowledge_agent.set_llm_provider(provider_key)

    def get_supported_llm_providers(self) -> Dict[str, str]:
        return self.knowledge_agent.SUPPORTED_PROVIDERS

    # =========================================================================
    # TAB 1: EXECUTIVE DASHBOARD DATA & METRICS
    # =========================================================================
    def get_executive_dashboard_data(self) -> Dict[str, Any]:
        """
        Compute KPIs and dataset slices for Tab 1 (Executive Dashboard):
        - Total Customers
        - High Risk Customers
        - Medium Risk Customers
        - Low Risk Customers
        - Revenue At Risk
        - Risk Distribution, Segment Distribution, Churn Probability Distribution, Top High-Risk Table
        """
        df = self.df_customers.copy()
        total_customers = int(len(df))
        high_risk_df = df[df["RiskLevel"] == "High"]
        medium_risk_df = df[df["RiskLevel"] == "Medium"]
        low_risk_df = df[df["RiskLevel"] == "Low"]

        high_count = int(len(high_risk_df))
        medium_count = int(len(medium_risk_df))
        low_count = int(len(low_risk_df))

        monthly_rev_at_risk = float(high_risk_df["MonthlyCharges"].sum())
        annual_rev_at_risk = round(monthly_rev_at_risk * 12.0, 2)

        top_high_risk = (
            high_risk_df.sort_values(by=["ChurnProbability", "MonthlyCharges"], ascending=[False, False])
            .head(25)[
                [
                    "CustomerID",
                    "CustomerName",
                    "Segment",
                    "ContractType",
                    "Tenure",
                    "MonthlyCharges",
                    "TotalCharges",
                    "InternetService",
                    "SupportCalls",
                    "ChurnProbability",
                    "RiskLevel",
                ]
            ]
            .copy()
        )

        return {
            "total_customers": total_customers,
            "high_risk_customers": high_count,
            "medium_risk_customers": medium_count,
            "low_risk_customers": low_count,
            "monthly_revenue_at_risk": round(monthly_rev_at_risk, 2),
            "revenue_at_risk": annual_rev_at_risk,
            "customers_df": df,
            "top_high_risk_df": top_high_risk,
        }

    # =========================================================================
    # TAB 2: AI DECISION CENTER (END-TO-END MULTI-AGENT ORCHESTRATION)
    # =========================================================================
    def get_customer_list(self) -> List[Dict[str, Any]]:
        """Return list of customers sorted with High-Risk accounts first for intuitive selection."""
        df = self.df_customers.sort_values(by="ChurnProbability", ascending=False)
        return df.to_dict(orient="records")

    def execute_customer_workflow(self, customer_id: str) -> Dict[str, Any]:
        """
        Execute the full multi-agent workflow for a selected customer:
        Customer Selected
          -> Executive Agent
          -> 1. Explainability Agent (SHAP)
          -> 2. Retention Agent
          -> 3. Marketing Agent
          -> 4. Knowledge Agent (RAG)
          -> 5. Campaign Simulator
          -> 6. Executive Report Generator & Final Executive Recommendation
        """
        matches = self.df_customers[self.df_customers["CustomerID"] == customer_id]
        if matches.empty:
            customer_row = self.df_customers.iloc[0]
        else:
            customer_row = matches.iloc[0]

        customer_profile = customer_row.to_dict()

        # Step 1: Call Explainability Agent
        explain_res = self.explainability_agent.explain_customer(customer_row)

        # Step 2: Call Retention Agent
        retention_res = self.retention_agent.recommend_retention_strategy(
            risk_level=explain_res["risk_level"],
            customer_segment=explain_res["segment"],
            top_drivers=explain_res["top_churn_drivers"],
            monthly_charges=float(customer_row["MonthlyCharges"]),
        )

        # Step 3: Call Marketing Agent
        marketing_res = self.marketing_agent.generate_campaign(
            customer_segment=explain_res["segment"],
            recommended_offer=retention_res["recommended_offer"],
            customer_name=str(customer_row["CustomerName"]),
            recommended_channel=retention_res["recommended_channel"],
            tenure_months=int(customer_row["Tenure"]),
            monthly_charges=float(customer_row["MonthlyCharges"]),
        )

        # Step 4: Call Knowledge Agent (RAG)
        knowledge_res = self.knowledge_agent.get_executive_insights(
            customer_segment=explain_res["segment"],
            risk_level=explain_res["risk_level"],
            recommended_offer=retention_res["recommended_offer"],
        )

        # Step 5: Call Campaign Simulator for expected outcome benchmarks
        simulation_res = self.campaign_simulator.simulate(
            customer_segment=explain_res["segment"],
            offer=retention_res["recommended_offer"],
        )

        # Step 6: Synthesize Final Executive Recommendation
        annual_val = float(customer_row["MonthlyCharges"]) * 12.0
        expected_saved_val = annual_val * (simulation_res["retention_improvement"] / 100.0)
        top_driver_names = ", ".join([d["feature"] for d in explain_res["top_churn_drivers"][:3]])

        executive_recommendation = (
            f"APPROVE IMMEDIATE '{retention_res['priority'].upper()}' PRIORITY INTERVENTION: "
            f"Deploy the '{retention_res['recommended_offer']}' offer to {customer_row['CustomerName']} "
            f"({customer_id}) via {retention_res['recommended_channel']} within 24 hours. "
            f"This action directly counteracts the customer's primary SHAP churn drivers ({top_driver_names}) "
            f"and aligns with historical campaign '{simulation_res['matching_campaign']}', which is projected "
            f"to yield a {simulation_res['response_rate']:.1f}% response probability and a "
            f"+{simulation_res['retention_improvement']:.1f}% retention lift—protecting ${annual_val:,.2f} "
            f"in annualized recurring revenue (expected incremental retained value: ${expected_saved_val:,.2f}/yr)."
        )

        # Step 7: Call Executive Report Agent (UI Report + DOCX Binary)
        report_content = self.report_agent.generate_report_content(
            customer_profile=customer_profile,
            explainability_output=explain_res,
            retention_output=retention_res,
            marketing_output=marketing_res,
            knowledge_output=knowledge_res,
            simulation_output=simulation_res,
            executive_recommendation=executive_recommendation,
        )
        docx_bytes = self.report_agent.build_docx_bytes(report_content)

        return {
            "customer_profile": customer_profile,
            "explainability": explain_res,
            "retention": retention_res,
            "marketing": marketing_res,
            "knowledge": knowledge_res,
            "simulation": simulation_res,
            "executive_recommendation": executive_recommendation,
            "executive_report": report_content,
            "docx_bytes": docx_bytes,
            "docx_filename": f"Customer_Retention_Executive_Report_{customer_id}.docx",
            "workflow_trace": [
                {"step": "1. Predict & Segment", "agent": "Executive Agent (XGBoost + K-Means)", "status": f"Churn Prob: {explain_res['churn_probability_pct']}% | Segment: {explain_res['segment']}"},
                {"step": "2. Explain Drivers", "agent": "Explainability Agent (SHAP)", "status": f"Top Driver: {explain_res['top_churn_drivers'][0]['feature']} ({explain_res['top_churn_drivers'][0]['shap_value']:+.3f})"},
                {"step": "3. Recommend Action", "agent": "Retention Agent", "status": f"Offer: {retention_res['recommended_offer']} | Priority: {retention_res['priority']}"},
                {"step": "4. Generate Campaign", "agent": "Marketing Agent", "status": f"Channel: {retention_res['recommended_channel']}"},
                {"step": "5. Retrieve Knowledge", "agent": "Knowledge Agent (TF-IDF RAG)", "status": f"Retrieved {len(knowledge_res['supporting_sources'])} policy & benchmark sources"},
                {"step": "6. Simulate Outcome", "agent": "Campaign Simulator", "status": f"Resp Rate: {simulation_res['response_rate']}% | Retention Lift: +{simulation_res['retention_improvement']}%"},
                {"step": "7. Executive Report", "agent": "Executive Report Agent (python-docx)", "status": "7-Section Report & DOCX Generated"},
            ],
        }

    # =========================================================================
    # TAB 3: KNOWLEDGE ASSISTANT (RAG Q&A)
    # =========================================================================
    def query_knowledge_assistant(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Route RAG Q&A queries through the Executive Agent to the Knowledge Agent."""
        return self.knowledge_agent.answer_question(question, top_k=top_k)

    # =========================================================================
    # TAB 4: CAMPAIGN SIMULATOR
    # =========================================================================
    def get_simulator_options(self) -> Dict[str, Any]:
        """Return available segments, offers, and historical dataset for Tab 4."""
        return {
            "segments": self.campaign_simulator.get_available_segments(),
            "offers": self.campaign_simulator.get_available_offers(),
            "history_df": self.campaign_simulator.get_all_campaigns(),
        }

    def run_campaign_simulation(self, customer_segment: str, offer: str) -> Dict[str, Any]:
        """Route campaign simulation requests through the Executive Agent."""
        return self.campaign_simulator.simulate(customer_segment=customer_segment, offer=offer)
