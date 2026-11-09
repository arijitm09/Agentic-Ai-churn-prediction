"""
Agentic AI Customer Retention Platform
Enterprise Streamlit Dashboard UI

Architecture Constraint:
- Contains ONLY FOUR TABS:
  1. Executive Dashboard
  2. AI Decision Center
  3. Knowledge Assistant (RAG)
  4. Campaign Simulator
- Customer Intelligence & AI Agents are unified into the centralized AI Decision Center.
- The Streamlit UI interacts EXCLUSIVELY with `ExecutiveAgent`.
- Clean Executive UI: No left sidebar, no developer/code context strings under components,
  and full-width responsive KPI cards with zero text truncation.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agents.executive_agent import ExecutiveAgent

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & CLEAN FULL-WIDTH ENTERPRISE STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic AI Customer Retention Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Hide left sidebar and collapse toggle completely for a clean customer-facing UI */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    .main {
        background-color: #F8FAFC;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }
    .enterprise-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 60%, #0D9488 100%);
        color: #FFFFFF;
        padding: 1.5rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.14);
    }
    .enterprise-header h1 {
        margin: 0;
        font-size: 1.95rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF;
    }
    .enterprise-header p {
        margin: 0.35rem 0 0 0;
        font-size: 0.98rem;
        color: #E2E8F0;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #1E3A8A;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
        height: 100%;
    }
    .kpi-card.high-risk {
        border-top-color: #DC2626;
    }
    .kpi-card.medium-risk {
        border-top-color: #D97706;
    }
    .kpi-card.low-risk {
        border-top-color: #059669;
    }
    .kpi-card.revenue-risk {
        border-top-color: #7C3AED;
    }
    .kpi-card.teal-accent {
        border-top-color: #0D9488;
    }
    .kpi-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
    }
    .kpi-value-text {
        font-size: 1.12rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.35;
        word-wrap: break-word;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #475569;
        margin-top: 0.35rem;
        font-weight: 500;
    }
    .kpi-sub.positive {
        color: #059669;
        font-weight: 600;
    }
    .agent-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #1E3A8A;
        border-radius: 10px;
        padding: 1.15rem 1.35rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    .agent-card.teal {
        border-left-color: #0D9488;
    }
    .workflow-pill-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        align-items: center;
        background: #F1F5F9;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border: 1px solid #CBD5E1;
        margin-bottom: 1.15rem;
        font-size: 0.84rem;
        font-weight: 600;
        color: #1E293B;
    }
    .workflow-step {
        background: #FFFFFF;
        border: 1px solid #94A3B8;
        color: #0F172A;
        padding: 0.3rem 0.65rem;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_executive_agent() -> ExecutiveAgent:
    """Initialize and cache the centralized ExecutiveAgent orchestrator."""
    return ExecutiveAgent()


executive_agent = get_executive_agent()

# -----------------------------------------------------------------------------
# TOP ENTERPRISE BANNER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="enterprise-header">
        <h1>🛡️ Agentic AI Customer Retention Platform</h1>
        <p>Enterprise Customer Churn Intelligence, Explainable AI Diagnostics, Automated Retention Strategy &amp; Campaign Outcome Simulation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# STRICTLY FOUR TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 1. Executive Dashboard",
        "🧠 2. AI Decision Center",
        "📚 3. Knowledge Assistant (RAG)",
        "🎯 4. Campaign Simulator",
    ]
)

# =============================================================================
# TAB 1: EXECUTIVE DASHBOARD
# =============================================================================
with tab1:
    dash_data = executive_agent.get_executive_dashboard_data()
    df_all = dash_data["customers_df"]
    df_top_risk = dash_data["top_high_risk_df"]

    total_cust = dash_data["total_customers"]
    high_cust = dash_data["high_risk_customers"]
    med_cust = dash_data["medium_risk_customers"]
    low_cust = dash_data["low_risk_customers"]
    rev_at_risk = dash_data["revenue_at_risk"]
    monthly_rev_risk = dash_data["monthly_revenue_at_risk"]

    # 5 Required KPI Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Customers</div>
                <div class="kpi-value">{total_cust:,}</div>
                <div class="kpi-sub">Active Customer Base</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-card high-risk">
                <div class="kpi-label">High Risk Customers</div>
                <div class="kpi-value">{high_cust:,}</div>
                <div class="kpi-sub">{high_cust / total_cust * 100:.1f}% of Total Base</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card medium-risk">
                <div class="kpi-label">Medium Risk Customers</div>
                <div class="kpi-value">{med_cust:,}</div>
                <div class="kpi-sub">{med_cust / total_cust * 100:.1f}% of Total Base</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-card low-risk">
                <div class="kpi-label">Low Risk Customers</div>
                <div class="kpi-value">{low_cust:,}</div>
                <div class="kpi-sub">{low_cust / total_cust * 100:.1f}% of Total Base</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"""
            <div class="kpi-card revenue-risk">
                <div class="kpi-label">Revenue At Risk</div>
                <div class="kpi-value">${rev_at_risk:,.0f}</div>
                <div class="kpi-sub">${monthly_rev_risk:,.2f}/mo High-Risk MRR</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1 Charts: Risk Distribution Pie Chart & Customer Segment Distribution
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        risk_counts = (
            df_all["RiskLevel"]
            .value_counts()
            .reindex(["High", "Medium", "Low"])
            .reset_index()
        )
        risk_counts.columns = ["RiskLevel", "Customers"]
        fig_pie = px.pie(
            risk_counts,
            names="RiskLevel",
            values="Customers",
            hole=0.48,
            title="Risk Distribution Pie Chart",
            color="RiskLevel",
            color_discrete_map={
                "High": "#DC2626",
                "Medium": "#D97706",
                "Low": "#059669",
            },
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label+value")
        fig_pie.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        seg_counts = (
            df_all.groupby(["Segment", "RiskLevel"])
            .size()
            .reset_index(name="Count")
        )
        fig_seg = px.bar(
            seg_counts,
            x="Segment",
            y="Count",
            color="RiskLevel",
            barmode="group",
            title="Customer Segment Distribution",
            color_discrete_map={
                "High": "#DC2626",
                "Medium": "#D97706",
                "Low": "#059669",
            },
            text_auto=True,
        )
        fig_seg.update_layout(
            height=380,
            xaxis_title="Customer Segment",
            yaxis_title="Number of Customers",
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    # Row 2: Churn Probability Distribution & Top High-Risk Customers Table
    col_chart3, col_table = st.columns([1.05, 1.15])

    with col_chart3:
        fig_hist = px.histogram(
            df_all,
            x="ChurnProbability",
            color="RiskLevel",
            nbins=25,
            title="Churn Probability Distribution",
            color_discrete_map={
                "High": "#DC2626",
                "Medium": "#D97706",
                "Low": "#059669",
            },
        )
        fig_hist.add_vline(
            x=0.65,
            line_dash="dash",
            line_color="#DC2626",
            annotation_text="High Risk (0.65)",
        )
        fig_hist.add_vline(
            x=0.35,
            line_dash="dot",
            line_color="#D97706",
            annotation_text="Medium Risk (0.35)",
        )
        fig_hist.update_layout(
            height=410,
            xaxis_title="Predicted Churn Probability",
            yaxis_title="Customer Count",
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_table:
        st.markdown("##### 🚨 Top High-Risk Customers Table")
        display_top = df_top_risk.copy()
        display_top["ChurnProbability"] = (display_top["ChurnProbability"] * 100).round(1).astype(str) + "%"
        display_top["MonthlyCharges"] = display_top["MonthlyCharges"].map(lambda v: f"${v:,.2f}")
        display_top["TotalCharges"] = display_top["TotalCharges"].map(lambda v: f"${v:,.2f}")
        st.dataframe(
            display_top,
            use_container_width=True,
            height=365,
            hide_index=True,
        )


# =============================================================================
# TAB 2: AI DECISION CENTER (CENTRALIZED AGENTIC WORKSPACE)
# =============================================================================
with tab2:
    st.markdown(
        """
        <div class="workflow-pill-bar">
            <span>🔄 <b>Decision Workflow:</b></span>
            <span class="workflow-step">Customer Selected</span> &rarr;
            <span class="workflow-step">Executive Agent</span> &rarr;
            <span class="workflow-step">Explainability Agent</span> &rarr;
            <span class="workflow-step">Retention Agent</span> &rarr;
            <span class="workflow-step">Marketing Agent</span> &rarr;
            <span class="workflow-step">Knowledge Agent</span> &rarr;
            <span class="workflow-step">Executive Recommendation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    customers_list = executive_agent.get_customer_list()

    col_sel1, col_sel2 = st.columns([1.1, 2.9])
    with col_sel1:
        risk_filter = st.selectbox(
            "Filter by Risk Level",
            options=["All Risk Levels", "High", "Medium", "Low"],
            index=0,
        )
    if risk_filter != "All Risk Levels":
        filtered_customers = [c for c in customers_list if c["RiskLevel"] == risk_filter]
    else:
        filtered_customers = customers_list

    with col_sel2:
        selected_cust = st.selectbox(
            "Select Customer Account",
            options=filtered_customers,
            format_func=lambda c: (
                f"{c['CustomerID']} — {c['CustomerName']} | "
                f"Risk: {c['RiskLevel']} ({c['ChurnProbability']*100:.1f}%) | "
                f"Segment: {c['Segment']} | ${c['MonthlyCharges']:.2f}/mo ({c['ContractType']})"
            ),
        )

    selected_customer_id = selected_cust["CustomerID"]

    # Execute the entire multi-agent workflow strictly through ExecutiveAgent
    workflow_result = executive_agent.execute_customer_workflow(selected_customer_id)

    profile = workflow_result["customer_profile"]
    explain = workflow_result["explainability"]
    retention = workflow_result["retention"]
    marketing = workflow_result["marketing"]
    knowledge = workflow_result["knowledge"]
    sim = workflow_result["simulation"]
    report_obj = workflow_result["executive_report"]

    # Top Customer Telemetry Summary Cards (no truncation)
    risk_card_class = (
        "high-risk" if explain["risk_level"] == "High"
        else ("medium-risk" if explain["risk_level"] == "Medium" else "low-risk")
    )
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="kpi-card {risk_card_class}">
                <div class="kpi-label">Churn Probability</div>
                <div class="kpi-value">{explain['churn_probability_pct']:.1f}%</div>
                <div class="kpi-sub">Predicted Attrition Risk</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="kpi-card {risk_card_class}">
                <div class="kpi-label">Risk Level</div>
                <div class="kpi-value">{explain['risk_level']}</div>
                <div class="kpi-sub">Priority Classification</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="kpi-card teal-accent">
                <div class="kpi-label">Customer Segment</div>
                <div class="kpi-value-text" style="font-size:1.35rem;">{explain['segment']}</div>
                <div class="kpi-sub">Behavioral Cluster</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Contract &amp; Monthly Charges</div>
                <div class="kpi-value">${profile['MonthlyCharges']:,.2f}<span style="font-size:0.95rem;font-weight:500;">/mo</span></div>
                <div class="kpi-sub">{profile['ContractType']} &bull; {profile['Tenure']} Months Tenure</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # AGENT 1: EXPLAINABILITY AGENT (SHAP EXPLAINABLE AI)
    # -------------------------------------------------------------------------
    st.markdown("### 🔍 Explainability Agent (SHAP Analysis)")
    col_shap_left, col_shap_right = st.columns([1.05, 1.25])

    with col_shap_left:
        st.markdown("#### Top Churn Drivers")
        for driver in explain["top_churn_drivers"]:
            badge_color = "🔴" if driver["shap_value"] > 0 else "🟢"
            st.markdown(
                f"- {badge_color} **{driver['feature']}** ({driver['value']}): "
                f"SHAP `{driver['shap_value']:+.4f}` — *{driver['direction']}*"
            )

        st.markdown("#### SHAP Table")
        shap_df = pd.DataFrame(explain["shap_table"])[
            ["Feature", "CustomerValue", "SHAPValue", "ImpactDirection"]
        ]
        shap_df.columns = ["Feature", "Customer Value", "SHAP Value", "Impact Direction"]
        st.dataframe(shap_df, use_container_width=True, hide_index=True)

    with col_shap_right:
        chart_df = pd.DataFrame(explain["shap_table"]).sort_values(
            by="AbsoluteImportance", ascending=True
        )
        chart_df["Impact"] = chart_df["SHAPValue"].apply(
            lambda v: "Increases Churn Risk (+)" if v >= 0 else "Reduces Churn Risk (-)"
        )
        fig_shap = px.bar(
            chart_df,
            x="SHAPValue",
            y="Feature",
            orientation="h",
            color="Impact",
            title=f"SHAP Feature Importance Bar Chart — {selected_customer_id}",
            color_discrete_map={
                "Increases Churn Risk (+)": "#DC2626",
                "Reduces Churn Risk (-)": "#059669",
            },
            text=chart_df["SHAPValue"].map(lambda v: f"{v:+.3f}"),
        )
        fig_shap.add_vline(x=0.0, line_width=1.5, line_color="#334155")
        fig_shap.update_layout(
            height=410,
            xaxis_title="SHAP Value (Impact on Churn Risk)",
            yaxis_title="Feature",
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # AGENT 2 & 3: RETENTION AGENT & MARKETING AGENT
    # -------------------------------------------------------------------------
    col_ret, col_mkt = st.columns([1.0, 1.2])

    with col_ret:
        st.markdown("### 🛡️ Retention Agent")
        st.markdown(
            f"""
            <div class="agent-card">
                <p><b>Customer Risk Level:</b> {retention['risk_level']} &nbsp;|&nbsp; <b>Customer Segment:</b> {retention['customer_segment']}</p>
                <hr style="margin:0.6rem 0; border-color:#E2E8F0;">
                <p>🎯 <b>Recommended Offer:</b> <span style="color:#1E3A8A; font-weight:700;">{retention['recommended_offer']}</span></p>
                <p>📡 <b>Recommended Retention Channel:</b> <span style="font-weight:600;">{retention['recommended_channel']}</span></p>
                <p>⚡ <b>Priority:</b> <span style="color:#DC2626; font-weight:700;">{retention['priority']}</span></p>
                <p>📋 <b>Business Justification:</b><br>{retention['business_justification']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📈 Projected Campaign Performance")
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.markdown(
                f"""
                <div class="kpi-card teal-accent">
                    <div class="kpi-label">Projected Response Rate</div>
                    <div class="kpi-value">{sim['response_rate']:.1f}%</div>
                    <div class="kpi-sub">{sim['campaign_name']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s_col2:
            st.markdown(
                f"""
                <div class="kpi-card low-risk">
                    <div class="kpi-label">Retention Improvement</div>
                    <div class="kpi-value">+{sim['retention_improvement']:.1f}%</div>
                    <div class="kpi-sub">Expected Churn Reduction</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_mkt:
        st.markdown("### ✉️ Marketing Agent")
        st.info(f"**Email Subject:** {marketing['email_subject']}")
        st.text_area(
            "Personalized Retention Campaign Content",
            value=marketing["email_body"],
            height=240,
        )
        st.markdown(f"**SMS / In-App Notification:**\n> {marketing['sms_copy']}")
        with st.expander("📞 View Outbound Retention Specialist Call Script"):
            st.markdown(marketing["call_script"])

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # AGENT 4: KNOWLEDGE AGENT (SUPPORTING BUSINESS INSIGHTS)
    # -------------------------------------------------------------------------
    st.markdown("### 📚 Knowledge Agent — Supporting Business Insights")
    for b in knowledge["bullet_insights"]:
        st.markdown(
            f"""
            <div class="agent-card teal" style="padding:0.85rem 1.15rem; margin-bottom:0.65rem;">
                <b>Policy &amp; Benchmark Insight (Relevance Score: {b['similarity']:.2f}):</b><br>
                {b['insight']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # EXECUTIVE RECOMMENDATION & 7-SECTION REPORT GENERATOR (WITH DOCX EXPORT)
    # -------------------------------------------------------------------------
    st.markdown("### 🏛️ Executive Recommendation & Executive Report")
    st.success(f"**Executive Recommendation:** {workflow_result['executive_recommendation']}")

    col_rep_header, col_rep_dl = st.columns([2.2, 1.0])
    with col_rep_header:
        st.markdown("#### 📄 Customer Retention Executive Report")
    with col_rep_dl:
        st.download_button(
            label="📥 Download DOCX Report",
            data=workflow_result["docx_bytes"],
            file_name=workflow_result["docx_filename"],
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            type="primary",
        )

    with st.expander("📖 View Full Customer Retention Executive Report", expanded=True):
        st.markdown(report_obj["markdown_report"])


# =============================================================================
# TAB 3: KNOWLEDGE ASSISTANT (RAG Q&A INTERFACE)
# =============================================================================
with tab3:
    st.markdown("### 📚 Enterprise Knowledge Assistant")

    preset_questions = [
        "Which campaign works best for At Risk customers?",
        "What is the maximum discount allowed?",
        "What retention strategy works best for Loyal Customers?",
        "Which channel performs best for Champions?",
    ]

    st.markdown("#### 💡 Recommended Executive Questions")
    q_cols = st.columns(2)
    if "active_rag_question" not in st.session_state:
        st.session_state["active_rag_question"] = preset_questions[0]

    for idx, pq in enumerate(preset_questions):
        with q_cols[idx % 2]:
            if st.button(f"🔎 {pq}", key=f"preset_q_{idx}", use_container_width=True):
                st.session_state["active_rag_question"] = pq

    custom_question = st.text_input(
        "Ask a question about retention strategies, discount policies, customer segments, or campaign performance:",
        value=st.session_state["active_rag_question"],
    )

    if custom_question:
        rag_response = executive_agent.query_knowledge_assistant(custom_question, top_k=3)

        st.markdown("#### ✅ Answer")
        st.markdown(
            f"""
            <div class="agent-card teal">
                <div>{rag_response['answer']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📂 Supporting Sources")
        for rank, src in enumerate(rag_response["supporting_sources"], start=1):
            clean_title = src["title"].title()
            with st.expander(
                f"Supporting Reference #{rank}: {clean_title} (Relevance Score: {src['score']:.2f})",
                expanded=(rank == 1),
            ):
                st.markdown(src["content"])


# =============================================================================
# TAB 4: CAMPAIGN SIMULATOR & ANALYTICS
# =============================================================================
with tab4:
    st.markdown("### 🎯 Retention Campaign Simulator")

    sim_options = executive_agent.get_simulator_options()
    df_campaigns_hist = sim_options["history_df"]

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        sim_segment = st.selectbox(
            "Select Customer Segment",
            options=sim_options["segments"],
            index=0,
        )
    with col_in2:
        sim_offer = st.selectbox(
            "Select Offer",
            options=sim_options["offers"],
            index=0,
        )

    sim_out = executive_agent.run_campaign_simulation(
        customer_segment=sim_segment,
        offer=sim_offer,
    )

    # 4 Required System Outputs (Full-text cards with zero truncation)
    o1, o2, o3, o4 = st.columns([1.35, 0.85, 0.85, 1.25])
    with o1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Matching Historical Campaign</div>
                <div class="kpi-value-text">{sim_out['campaign_name']}</div>
                <div class="kpi-sub">Campaign ID: {sim_out['campaign_id']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with o2:
        st.markdown(
            f"""
            <div class="kpi-card low-risk">
                <div class="kpi-label">Retention Improvement</div>
                <div class="kpi-value">+{sim_out['retention_improvement']:.1f}%</div>
                <div class="kpi-sub positive">{sim_out['delta_retention_vs_segment']:+.1f}% vs. Segment Avg</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with o3:
        st.markdown(
            f"""
            <div class="kpi-card teal-accent">
                <div class="kpi-label">Response Rate</div>
                <div class="kpi-value">{sim_out['response_rate']:.1f}%</div>
                <div class="kpi-sub positive">{sim_out['delta_response_vs_segment']:+.1f}% vs. Segment Avg</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with o4:
        st.markdown(
            f"""
            <div class="kpi-card revenue-risk">
                <div class="kpi-label">Recommended Channel</div>
                <div class="kpi-value-text">{sim_out['recommended_channel']}</div>
                <div class="kpi-sub">Optimal Delivery Channel</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 💡 Historical Campaign Insight")
    st.info(sim_out["historical_insight"])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 Campaign Analytics — Campaign Performance Chart")

    col_sim_chart1, col_sim_chart2 = st.columns([1.35, 1.0])

    with col_sim_chart1:
        fig_perf = go.Figure()
        fig_perf.add_trace(
            go.Bar(
                x=df_campaigns_hist["CampaignName"],
                y=df_campaigns_hist["ResponseRate"],
                name="Response Rate (%)",
                marker_color="#1E3A8A",
                text=df_campaigns_hist["ResponseRate"].map(lambda v: f"{v:.1f}%"),
                textposition="auto",
            )
        )
        fig_perf.add_trace(
            go.Bar(
                x=df_campaigns_hist["CampaignName"],
                y=df_campaigns_hist["RetentionImprovement"],
                name="Retention Improvement (%)",
                marker_color="#0D9488",
                text=df_campaigns_hist["RetentionImprovement"].map(lambda v: f"+{v:.1f}%"),
                textposition="auto",
            )
        )
        fig_perf.update_layout(
            barmode="group",
            title="Campaign Performance Chart: Response Rate vs. Retention Improvement",
            xaxis_title="Historical Campaign",
            yaxis_title="Percentage (%)",
            xaxis_tickangle=-30,
            height=460,
            margin=dict(l=20, r=20, t=55, b=110),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    with col_sim_chart2:
        seg_filtered_df = df_campaigns_hist[df_campaigns_hist["Segment"] == sim_segment]
        fig_seg_perf = px.bar(
            seg_filtered_df,
            x="Offer",
            y=["ResponseRate", "RetentionImprovement"],
            barmode="group",
            title=f"Segment Performance Comparison: {sim_segment}",
            color_discrete_map={
                "ResponseRate": "#1E3A8A",
                "RetentionImprovement": "#0D9488",
            },
        )
        fig_seg_perf.update_layout(
            height=460,
            xaxis_title="Retention Offer",
            yaxis_title="Percentage (%)",
            margin=dict(l=20, r=20, t=55, b=80),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_seg_perf, use_container_width=True)

    st.markdown("#### 📋 Historical Campaign Performance Summary")
    st.dataframe(df_campaigns_hist, use_container_width=True, hide_index=True)
