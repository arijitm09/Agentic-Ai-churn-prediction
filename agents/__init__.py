"""
Agentic AI Customer Retention Platform - Modular Multi-Agent Package.
The Streamlit UI interacts exclusively with `ExecutiveAgent`, which orchestrates:
1. ExplainabilityAgent
2. RetentionAgent
3. MarketingAgent
4. KnowledgeAgent
5. CampaignSimulator
6. ExecutiveReportAgent
"""

from agents.executive_agent import ExecutiveAgent

__all__ = ["ExecutiveAgent"]
