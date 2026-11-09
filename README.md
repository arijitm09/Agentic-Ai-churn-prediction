# 🛡️ Agentic AI Customer Retention Platform

An enterprise-grade, end-to-end **Agentic AI Customer Retention Platform** built with **Python**, **Streamlit**, **XGBoost**, **K-Means**, **SHAP Explainable AI**, **TF-IDF + Cosine Similarity RAG**, **Plotly**, and **`python-docx`**.

---

## 🎯 Business Objective & End-to-End Agentic Workflow

The platform predicts customer churn, explains feature-level churn drivers, segments customers, recommends targeted retention actions, generates personalized marketing campaigns, retrieves enterprise business knowledge via RAG, simulates campaign outcomes, and exports 7-section executive reports in `.docx` format.

```text
Predict (XGBoost)
  ↓
Explain (SHAP Explainable AI)
  ↓
Segment (K-Means Clustering)
  ↓
Recommend (Retention Agent)
  ↓
Generate Campaign (Marketing Agent)
  ↓
Retrieve Knowledge (Knowledge Agent - TF-IDF RAG)
  ↓
Simulate Campaign (Campaign Simulator)
  ↓
Generate Executive Report (Executive Report Agent - python-docx)
```

### Centralized Multi-Agent Architecture
- **`ExecutiveAgent` (`agents/executive_agent.py`)** is the single orchestrator called by the Streamlit UI (`app.py`).
- Subordinate specialized agents (`ExplainabilityAgent`, `RetentionAgent`, `MarketingAgent`, `KnowledgeAgent`, `CampaignSimulator`, `ExecutiveReportAgent`) are orchestrated strictly by `ExecutiveAgent` and are never invoked directly by the UI.
- **Customer Intelligence** and **AI Agents** are unified inside the centralized **AI Decision Center** workspace.

---

## 🖥️ Four-Tab Application Structure

| Tab | Name | Core Capabilities |
| :--- | :--- | :--- |
| **Tab 1** | **Executive Dashboard** | Displays 5 KPIs (`Total Customers`, `High Risk Customers`, `Medium Risk Customers`, `Low Risk Customers`, `Revenue At Risk`) and 4 interactive Plotly visualizations (`Risk Distribution Pie Chart`, `Customer Segment Distribution`, `Churn Probability Distribution`, `Top High-Risk Customers Table`). |
| **Tab 2** | **AI Decision Center** | Centralized agentic workspace. Select any customer to trigger `ExecutiveAgent` -> `ExplainabilityAgent` (SHAP Table & SHAP Feature Importance Bar Chart) -> `RetentionAgent` (Offer, Channel, Priority, Justification) -> `MarketingAgent` (Personalized Email/SMS/Call Script) -> `KnowledgeAgent` (RAG Insights) -> `ExecutiveReportAgent` (7-Section UI Report + 1-click `.docx` download). |
| **Tab 3** | **Knowledge Assistant (RAG)** | Dedicated Q&A interface backed by `vectorizer.pkl`, `vectors.pkl`, and `documents.pkl` across 5 enterprise knowledge sources (`retention_policy.txt`, `campaign_history.txt`, `customer_segment_profiles.txt`, `offers.txt`, `brand_guidelines.txt`). Displays synthesized `Answer` and ranked `Supporting Sources`. |
| **Tab 4** | **Campaign Simulator** | Interactive simulator powered by `campaign_history.csv`. Select `Customer Segment` and `Offer` to view `Matching Historical Campaign`, `Retention Improvement`, `Response Rate`, `Recommended Channel`, `Historical Campaign Insight`, and the Plotly `Campaign Performance Chart`. |

---

## 📁 Project Folder Structure

```text
agentic_ai_customer_retention_platform/
├── app.py
├── train_pipeline.py
├── requirements.txt
├── Dockerfile
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── agents/
│   ├── __init__.py
│   ├── executive_agent.py
│   ├── retention_agent.py
│   ├── marketing_agent.py
│   ├── explainability_agent.py
│   ├── knowledge_agent.py
│   ├── campaign_simulator.py
│   └── executive_report_agent.py
├── data/
│   ├── customers.csv
│   ├── customers_scored.csv
│   └── campaign_history.csv
├── models/
│   ├── churn_model.pkl
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
├── rag/
│   ├── retention_policy.txt
│   ├── campaign_history.txt
│   ├── customer_segment_profiles.txt
│   ├── offers.txt
│   ├── brand_guidelines.txt
│   ├── vectorizer.pkl
│   ├── vectors.pkl
│   └── documents.pkl
└── reports/
    └── Customer_Retention_Executive_Report_CUST-1004.docx
```

---

## 🔌 Future Enhancement: Pluggable LLM Integration

`agents/knowledge_agent.py` implements the `BaseLLMProvider` strategy pattern (`TFIDFExtractiveProvider` + `PluggableLLMProvider`), allowing seamless replacement or augmentation of the RAG synthesis layer with:
- **Gemini API** (`KNOWLEDGE_AGENT_PROVIDER=gemini`)
- **TinyLlama** (`KNOWLEDGE_AGENT_PROVIDER=tinyllama`)
- **Llama 3** (`KNOWLEDGE_AGENT_PROVIDER=llama3`)
- **Azure OpenAI** (`KNOWLEDGE_AGENT_PROVIDER=azure_openai`)

Switching providers requires **zero changes to `app.py` or `ExecutiveAgent`**.

---

## 🚀 Quickstart & Deployment Guide

### 1. Local Execution
```bash
pip install -r requirements.txt
python train_pipeline.py
streamlit run app.py
```

### 2. Deploy to GitHub + Streamlit Community Cloud
```bash
git init
git add .
git commit -m "Initial commit: Agentic AI Customer Retention Platform"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
Then visit [share.streamlit.io](https://share.streamlit.io), select your repository, and set the main file path to `app.py`.

### 3. Docker Deployment
```bash
docker build -t agentic-retention-platform .
docker run -p 8501:8501 agentic-retention-platform
```
