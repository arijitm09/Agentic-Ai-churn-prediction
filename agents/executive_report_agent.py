"""
Executive Report Generator Agent
Generates the 7-section Customer Retention Executive Report for UI display
and exports a formatted Microsoft Word (.docx) document using `python-docx`.

Required Sections:
1. Executive Summary
2. Customer Risk Analysis
3. SHAP Churn Drivers
4. Retention Strategy
5. Marketing Campaign
6. Historical Insights
7. Final Executive Recommendation
"""

import io
import os
from datetime import datetime
from typing import Dict, Any, List

import docx
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from docx.shared import Inches, Pt, RGBColor

from train_pipeline import BASE_DIR


class ExecutiveReportAgent:
    """Specialized Agent for generating the 7-Section Executive Report (UI + DOCX)."""

    PRIMARY_HEX = "1B365D"      # Deep Enterprise Navy
    ACCENT_HEX = "0D9488"       # Enterprise Teal
    ALT_ROW_HEX = "F1F5F9"      # Soft Slate Tint
    CALLOUT_FILL_HEX = "F8FAFC" # Subtle Callout Surface

    TEXT_DARK = RGBColor(0x1E, 0x29, 0x3B)
    TEXT_NAVY = RGBColor(0x1B, 0x36, 0x5D)
    TEXT_MUTED = RGBColor(0x47, 0x55, 0x69)
    TEXT_WHITE = RGBColor(0xFF, 0xFF, 0xFF)

    def __init__(self) -> None:
        self.reports_dir = os.path.join(BASE_DIR, "reports")
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_report_content(
        self,
        customer_profile: Dict[str, Any],
        explainability_output: Dict[str, Any],
        retention_output: Dict[str, Any],
        marketing_output: Dict[str, Any],
        knowledge_output: Dict[str, Any],
        simulation_output: Dict[str, Any],
        executive_recommendation: str,
    ) -> Dict[str, Any]:
        """Compile all 7 required sections for UI rendering and DOCX generation."""
        cid = customer_profile.get("CustomerID", "N/A")
        cname = customer_profile.get("CustomerName", "Valued Customer")
        prob_pct = explainability_output.get("churn_probability_pct", 0.0)
        risk_level = explainability_output.get("risk_level", "High")
        segment = explainability_output.get("segment", "At Risk")
        monthly = float(customer_profile.get("MonthlyCharges", 0.0))
        total_charges = float(customer_profile.get("TotalCharges", 0.0))
        annual_arr = monthly * 12.0

        # Section 1: Executive Summary
        sec1_summary = (
            f"Customer {cname} ({cid}) is currently classified in the '{segment}' segment with a "
            f"predicted churn probability of {prob_pct:.1f}% ({risk_level} Risk), placing ${annual_arr:,.2f} "
            f"in annualized recurring revenue at risk. The Executive Agent orchestrated Explainability (SHAP), "
            f"Retention, Marketing, Knowledge (RAG), and Campaign Simulation agents to formulate an immediate "
            f"intervention centered on '{retention_output['recommended_offer']}' delivered via "
            f"{retention_output['recommended_channel']} (Priority: {retention_output['priority']})."
        )

        # Section 2: Customer Risk Analysis
        sec2_risk = {
            "narrative": (
                f"Account {cid} has a tenure of {customer_profile.get('Tenure', 0)} months on a "
                f"{customer_profile.get('ContractType', 'N/A')} contract with ${monthly:,.2f} monthly charges "
                f"(${total_charges:,.2f} cumulative total charges). The account utilizes "
                f"{customer_profile.get('InternetService', 'N/A')} internet service, has recorded "
                f"{customer_profile.get('SupportCalls', 0)} recent support calls, Tech Support set to "
                f"'{customer_profile.get('TechSupport', 'No')}', and pays via {customer_profile.get('PaymentMethod', 'N/A')}."
            ),
            "metrics_table": [
                ("Customer ID & Name", f"{cid} — {cname}"),
                ("K-Means Customer Segment", segment),
                ("Predicted Churn Probability", f"{prob_pct:.1f}% ({risk_level} Risk Tier)"),
                ("Contract Type & Tenure", f"{customer_profile.get('ContractType')} ({customer_profile.get('Tenure')} months)"),
                ("Monthly / Annualized Revenue", f"${monthly:,.2f}/mo (${annual_arr:,.2f}/yr)"),
                ("Internet Service & Tech Support", f"{customer_profile.get('InternetService')} | Tech Support: {customer_profile.get('TechSupport')}"),
                ("Payment Method & Support Calls", f"{customer_profile.get('PaymentMethod')} | {customer_profile.get('SupportCalls')} support calls"),
            ],
        }

        # Section 3: SHAP Churn Drivers
        shap_rows = explainability_output.get("shap_table", [])
        top_drivers = explainability_output.get("top_churn_drivers", [])
        sec3_shap = {
            "narrative": (
                "SHAP (SHapley Additive exPlanations) feature attribution isolates the exact magnitude and "
                "direction of each account attribute contributing to the customer's churn risk score. "
                "Positive SHAP values push the customer toward churn, whereas negative values anchor retention."
            ),
            "top_drivers": top_drivers,
            "shap_table": shap_rows,
        }

        # Section 4: Retention Strategy
        sec4_retention = {
            "recommended_offer": retention_output["recommended_offer"],
            "recommended_channel": retention_output["recommended_channel"],
            "priority": retention_output["priority"],
            "business_justification": retention_output["business_justification"],
        }

        # Section 5: Marketing Campaign
        sec5_marketing = {
            "email_subject": marketing_output["email_subject"],
            "email_body": marketing_output["email_body"],
            "sms_copy": marketing_output["sms_copy"],
            "call_script": marketing_output["call_script"],
        }

        # Section 6: Historical Insights
        sec6_insights = {
            "rag_insights": knowledge_output.get("bullet_insights", []),
            "simulation_match": simulation_output.get("matching_campaign", ""),
            "response_rate": simulation_output.get("response_rate", 0.0),
            "retention_improvement": simulation_output.get("retention_improvement", 0.0),
            "historical_narrative": simulation_output.get("historical_insight", ""),
        }

        # Section 7: Final Executive Recommendation
        sec7_recommendation = executive_recommendation

        markdown_report = f"""### 1. Executive Summary
{sec1_summary}

### 2. Customer Risk Analysis
{sec2_risk['narrative']}

### 3. SHAP Churn Drivers
{sec3_shap['narrative']}
""" + "\n".join([f"- **{d['feature']}** (`{d['value']}`): SHAP `{d['shap_value']:+.4f}` — *{d['direction']}*" for d in top_drivers]) + f"""

### 4. Retention Strategy
- **Recommended Offer:** {sec4_retention['recommended_offer']}
- **Recommended Retention Channel:** {sec4_retention['recommended_channel']}
- **Execution Priority:** {sec4_retention['priority']}
- **Business Justification:** {sec4_retention['business_justification']}

### 5. Marketing Campaign
- **Email Subject:** `{sec5_marketing['email_subject']}`
- **SMS / In-App Copy:** *{sec5_marketing['sms_copy']}*

**Personalized Email Body:**
```text
{sec5_marketing['email_body']}
```

### 6. Historical Insights
- **Matching Historical Campaign:** {sec6_insights['simulation_match']}
- **Simulated Response Rate:** {sec6_insights['response_rate']:.1f}%
- **Simulated Retention Improvement:** +{sec6_insights['retention_improvement']:.1f}%
- **Benchmark Assessment:** {sec6_insights['historical_narrative']}

**Supporting Knowledge Base Policies (TF-IDF RAG):**
""" + "\n".join([f"- **`{b['source']}` (Similarity `{b['similarity']:.2f}`):** {b['insight']}" for b in sec6_insights["rag_insights"]]) + f"""

### 7. Final Executive Recommendation
{sec7_recommendation}
"""

        return {
            "customer_id": cid,
            "customer_name": cname,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "section_1_executive_summary": sec1_summary,
            "section_2_customer_risk_analysis": sec2_risk,
            "section_3_shap_churn_drivers": sec3_shap,
            "section_4_retention_strategy": sec4_retention,
            "section_5_marketing_campaign": sec5_marketing,
            "section_6_historical_insights": sec6_insights,
            "section_7_final_executive_recommendation": sec7_recommendation,
            "markdown_report": markdown_report,
        }

    def _add_heading_1(self, doc: docx.Document, text: str) -> None:
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = self.TEXT_NAVY
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True

    def _add_body_paragraph(self, doc: docx.Document, text: str, bold_prefix: str = "") -> None:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.name = "Calibri"
            r_pre.font.size = Pt(11)
            r_pre.font.bold = True
            r_pre.font.color.rgb = self.TEXT_DARK
        r_body = p.add_run(text)
        r_body.font.name = "Calibri"
        r_body.font.size = Pt(11)
        r_body.font.color.rgb = self.TEXT_DARK

    def _add_styled_table(
        self,
        doc: docx.Document,
        headers: List[str],
        rows_data: List[List[str]],
        right_align_cols: List[int] = None,
    ) -> None:
        right_align_cols = right_align_cols or []
        table = doc.add_table(rows=1, cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER

        hdr_row = table.rows[0]
        hdr_tr = hdr_row._tr.get_or_add_trPr()
        hdr_tr.append(parse_xml(r'<w:tblHeader {}/>'.format(nsdecls("w"))))

        for idx, text in enumerate(headers):
            cell = hdr_row.cells[idx]
            cell.text = text
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            shd = parse_xml(r'<w:shd {} w:fill="{}"/>'.format(nsdecls("w"), self.PRIMARY_HEX))
            cell._tc.get_or_add_tcPr().append(shd)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if idx in right_align_cols else WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(4)
                p.paragraph_format.keep_with_next = True
                for run in p.runs:
                    run.font.name = "Calibri"
                    run.font.bold = True
                    run.font.size = Pt(10.5)
                    run.font.color.rgb = self.TEXT_WHITE

        for r_idx, r_vals in enumerate(rows_data):
            row = table.add_row()
            trPr = row._tr.get_or_add_trPr()
            trPr.append(parse_xml(r'<w:cantSplit {}/>'.format(nsdecls("w"))))
            for c_idx, val in enumerate(r_vals):
                cell = row.cells[c_idx]
                cell.text = str(val)
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                if r_idx % 2 == 1:
                    shd = parse_xml(r'<w:shd {} w:fill="{}"/>'.format(nsdecls("w"), self.ALT_ROW_HEX))
                    cell._tc.get_or_add_tcPr().append(shd)
                for p in cell.paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if c_idx in right_align_cols else WD_ALIGN_PARAGRAPH.LEFT
                    p.paragraph_format.space_before = Pt(3)
                    p.paragraph_format.space_after = Pt(3)
                    for run in p.runs:
                        run.font.name = "Calibri"
                        run.font.size = Pt(10)
                        run.font.color.rgb = self.TEXT_DARK
                        if c_idx == 0:
                            run.font.bold = True

        if len(table.rows) <= 5:
            for row in table.rows[:-1]:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        p.paragraph_format.keep_with_next = True

        spacer = doc.add_paragraph()
        spacer.paragraph_format.space_before = Pt(0)
        spacer.paragraph_format.space_after = Pt(4)

    def _add_callout_box(self, doc: docx.Document, title: str, body_text: str) -> None:
        table = doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        row = table.rows[0]
        row._tr.get_or_add_trPr().append(parse_xml(r'<w:cantSplit {}/>'.format(nsdecls("w"))))
        cell = row.cells[0]
        shd = parse_xml(r'<w:shd {} w:fill="{}"/>'.format(nsdecls("w"), self.CALLOUT_FILL_HEX))
        cell._tc.get_or_add_tcPr().append(shd)

        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        r_title = p.add_run(f"{title}\n")
        r_title.font.name = "Calibri"
        r_title.font.size = Pt(11)
        r_title.font.bold = True
        r_title.font.color.rgb = self.TEXT_NAVY

        r_body = p.add_run(body_text)
        r_body.font.name = "Calibri"
        r_body.font.size = Pt(10.5)
        r_body.font.color.rgb = self.TEXT_DARK

    def build_docx_bytes(self, report_data: Dict[str, Any]) -> bytes:
        """Build a Microsoft Word (.docx) document using python-docx and return bytes + save to reports/."""
        doc = docx.Document()

        for section in doc.sections:
            section.top_margin = Inches(1.0)
            section.bottom_margin = Inches(1.0)
            section.left_margin = Inches(1.0)
            section.right_margin = Inches(1.0)

        style_normal = doc.styles["Normal"]
        style_normal.font.name = "Calibri"
        style_normal.font.size = Pt(11)
        style_normal.font.color.rgb = self.TEXT_DARK

        title_p = doc.add_paragraph()
        title_run = title_p.add_run("Customer Retention Executive Report")
        title_run.font.name = "Calibri"
        title_run.font.size = Pt(22)
        title_run.font.bold = True
        title_run.font.color.rgb = self.TEXT_NAVY
        title_p.paragraph_format.space_after = Pt(2)

        sub_p = doc.add_paragraph()
        sub_run = sub_p.add_run(
            f"Account: {report_data['customer_id']} ({report_data['customer_name']}) | "
            f"Generated by Agentic AI Customer Retention Platform | {report_data['generated_at']}"
        )
        sub_run.font.name = "Calibri"
        sub_run.font.size = Pt(11)
        sub_run.font.color.rgb = self.TEXT_MUTED
        sub_p.paragraph_format.space_after = Pt(12)

        # 1. Executive Summary
        self._add_heading_1(doc, "1. Executive Summary")
        self._add_callout_box(doc, "Executive Synthesis", report_data["section_1_executive_summary"])

        # 2. Customer Risk Analysis
        self._add_heading_1(doc, "2. Customer Risk Analysis")
        sec2 = report_data["section_2_customer_risk_analysis"]
        self._add_body_paragraph(doc, sec2["narrative"])
        self._add_styled_table(
            doc,
            headers=["Account Dimension", "Current Profile & Risk Telemetry"],
            rows_data=[[k, v] for k, v in sec2["metrics_table"]],
        )

        # 3. SHAP Churn Drivers
        self._add_heading_1(doc, "3. SHAP Churn Drivers")
        sec3 = report_data["section_3_shap_churn_drivers"]
        self._add_body_paragraph(doc, sec3["narrative"])
        shap_rows = [
            [
                r["Feature"],
                r["CustomerValue"],
                f"{r['SHAPValue']:+.4f}",
                r["ImpactDirection"],
            ]
            for r in sec3["shap_table"]
        ]
        self._add_styled_table(
            doc,
            headers=["Feature", "Customer Value", "SHAP Attribution", "Directional Impact"],
            rows_data=shap_rows,
            right_align_cols=[2],
        )

        # 4. Retention Strategy
        self._add_heading_1(doc, "4. Retention Strategy")
        sec4 = report_data["section_4_retention_strategy"]
        self._add_styled_table(
            doc,
            headers=["Strategy Parameter", "Retention Agent Recommendation"],
            rows_data=[
                ["Recommended Offer", sec4["recommended_offer"]],
                ["Recommended Retention Channel", sec4["recommended_channel"]],
                ["Execution Priority", sec4["priority"]],
                ["Business Justification", sec4["business_justification"]],
            ],
        )

        # 5. Marketing Campaign
        self._add_heading_1(doc, "5. Marketing Campaign")
        sec5 = report_data["section_5_marketing_campaign"]
        self._add_body_paragraph(doc, sec5["email_subject"], bold_prefix="Email Subject Line: ")
        self._add_body_paragraph(doc, sec5["sms_copy"], bold_prefix="SMS / In-App Notification: ")
        self._add_callout_box(doc, "Personalized Outbound Email Copy", sec5["email_body"])

        # 6. Historical Insights
        self._add_heading_1(doc, "6. Historical Insights")
        sec6 = report_data["section_6_historical_insights"]
        self._add_body_paragraph(doc, sec6["historical_narrative"])
        rag_rows = [
            [b["source"], f"{b['similarity']:.2f}", b["insight"]]
            for b in sec6["rag_insights"]
        ]
        if rag_rows:
            self._add_styled_table(
                doc,
                headers=["Knowledge Source", "Cosine Sim.", "Retrieved Policy & Benchmark Insight"],
                rows_data=rag_rows,
                right_align_cols=[1],
            )

        # 7. Final Executive Recommendation
        self._add_heading_1(doc, "7. Final Executive Recommendation")
        self._add_body_paragraph(doc, report_data["section_7_final_executive_recommendation"])

        out_filename = f"Customer_Retention_Executive_Report_{report_data['customer_id']}.docx"
        out_path = os.path.join(self.reports_dir, out_filename)
        doc.save(out_path)

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer.getvalue()
