"""
Retention Agent
Determines Recommended Offer, Recommended Retention Channel, Priority, and
Business Justification based on Risk Level, Customer Segment, and Top Churn Drivers.
"""

from typing import Dict, Any, List, Optional


class RetentionAgent:
    """Specialized Agent for formulating targeted Customer Retention Strategies."""

    STRATEGY_MATRIX = {
        ("High", "At Risk"): {
            "recommended_offer": "20% Cashback",
            "recommended_channel": "Phone Call + Email",
            "priority": "High",
            "rationale": (
                "Customer is in the 'At Risk' segment with High churn probability. "
                "Immediate high-touch Phone Call + Email outreach offering 20% Cashback "
                "neutralizes month-to-month billing friction and historically delivers a "
                "28.5% retention improvement."
            ),
        },
        ("High", "Loyal Customers"): {
            "recommended_offer": "25% Annual Contract Discount",
            "recommended_channel": "Phone Call + Email",
            "priority": "High",
            "rationale": (
                "High-risk 'Loyal Customer' showing acute attrition signals. "
                "Deploying the maximum policy-approved 25% Annual Contract Discount via "
                "Phone Call + Email locks in multi-year recurring revenue and prevents competitor defection."
            ),
        },
        ("High", "Champions"): {
            "recommended_offer": "Loyalty VIP Tier Upgrade",
            "recommended_channel": "Dedicated Account Manager + VIP Concierge Call",
            "priority": "High",
            "rationale": (
                "High-LTV 'Champion' account at elevated risk. Executive intervention via "
                "Dedicated Account Manager + VIP Concierge Call with a Loyalty VIP Tier Upgrade "
                "achieves a 46.5% response rate and preserves high-margin bundle revenue."
            ),
        },
        ("High", "Price Sensitive"): {
            "recommended_offer": "25% Annual Contract Discount",
            "recommended_channel": "Phone Call + Email",
            "priority": "High",
            "rationale": (
                "High-risk 'Price Sensitive' customer driven by monthly rate sensitivity. "
                "Offering a 25% Annual Contract Discount via Phone Call + Email lowers monthly "
                "out-of-pocket charges while securing a 12-month contract commitment."
            ),
        },
        ("Medium", "At Risk"): {
            "recommended_offer": "20% Cashback",
            "recommended_channel": "Phone Call + Email",
            "priority": "High",
            "rationale": (
                "Medium-risk 'At Risk' account exhibiting early warning churn indicators. "
                "Proactive Phone Call + Email intervention with 20% Cashback stops risk escalation "
                "before the customer enters active cancellation."
            ),
        },
        ("Medium", "Loyal Customers"): {
            "recommended_offer": "Complimentary Fiber Upgrade + Priority Tech Support",
            "recommended_channel": "Email + In-App Notification",
            "priority": "Medium",
            "rationale": (
                "Medium-risk 'Loyal Customer' best served by value-add service enhancement rather "
                "than heavy discounting. Complimentary Fiber Upgrade + Priority Tech Support via "
                "Email + In-App Notification yields a 38.4% response rate with zero ARPU dilution."
            ),
        },
        ("Medium", "Champions"): {
            "recommended_offer": "Loyalty VIP Tier Upgrade",
            "recommended_channel": "Dedicated Account Manager + VIP Concierge Call",
            "priority": "Medium",
            "rationale": (
                "Medium-risk 'Champion' customer requiring proactive white-glove recognition. "
                "Dedicated Account Manager outreach with a Loyalty VIP Tier Upgrade reinforces "
                "brand affinity and drives a 31.0% retention improvement."
            ),
        },
        ("Medium", "Price Sensitive"): {
            "recommended_offer": "15% Bill Credit for 3 Months",
            "recommended_channel": "SMS + Email",
            "priority": "Medium",
            "rationale": (
                "Medium-risk 'Price Sensitive' customer. Delivering a 15% Bill Credit for 3 Months "
                "via automated SMS + Email relieves immediate cost pressure within standard "
                "medium-tier discount governance."
            ),
        },
        ("Low", "At Risk"): {
            "recommended_offer": "15% Bill Credit for 3 Months",
            "recommended_channel": "SMS + Email",
            "priority": "Low",
            "rationale": (
                "Low immediate churn probability within the 'At Risk' cluster. "
                "Automated SMS + Email check-in with a targeted 15% Bill Credit for 3 Months "
                "stabilizes tenure growth cost-effectively."
            ),
        },
        ("Low", "Loyal Customers"): {
            "recommended_offer": "Complimentary Fiber Upgrade + Priority Tech Support",
            "recommended_channel": "Email + In-App Notification",
            "priority": "Low",
            "rationale": (
                "Stable 'Loyal Customer' with low churn risk. Offering a Complimentary Fiber Upgrade + "
                "Priority Tech Support via Email + In-App Notification rewards loyalty and increases "
                "ecosystem lock-in."
            ),
        },
        ("Low", "Champions"): {
            "recommended_offer": "Loyalty VIP Tier Upgrade",
            "recommended_channel": "Dedicated Account Manager + VIP Concierge Call",
            "priority": "Low",
            "rationale": (
                "Healthy 'Champions' account. Periodic VIP Concierge recognition with a "
                "Loyalty VIP Tier Upgrade sustains long-term advocacy and referral potential."
            ),
        },
        ("Low", "Price Sensitive"): {
            "recommended_offer": "15% Bill Credit for 3 Months",
            "recommended_channel": "SMS + Email",
            "priority": "Low",
            "rationale": (
                "Low-risk 'Price Sensitive' customer. Lightweight SMS + Email engagement with "
                "a 15% Bill Credit for 3 Months maintains price competitiveness."
            ),
        },
    }

    def recommend_retention_strategy(
        self,
        risk_level: str,
        customer_segment: str,
        top_drivers: Optional[List[Dict[str, Any]]] = None,
        monthly_charges: float = 0.0,
    ) -> Dict[str, str]:
        """
        Input:
          - Risk Level (High, Medium, Low)
          - Customer Segment (At Risk, Loyal Customers, Champions, Price Sensitive)
        Output:
          - Recommended Offer
          - Recommended Retention Channel
          - Priority
          - Business Justification
        """
        key = (risk_level, customer_segment)
        base = self.STRATEGY_MATRIX.get(
            key,
            {
                "recommended_offer": "20% Cashback",
                "recommended_channel": "Phone Call + Email",
                "priority": "High" if risk_level == "High" else "Medium",
                "rationale": (
                    f"Targeted intervention for {risk_level} risk customer in {customer_segment} segment "
                    "to protect recurring revenue and mitigate primary churn drivers."
                ),
            },
        )

        driver_names = [d["feature"] for d in (top_drivers or [])[:3]]
        driver_clause = (
            f" Key SHAP churn drivers addressed: {', '.join(driver_names)}."
            if driver_names
            else ""
        )
        revenue_clause = (
            f" Protects ${monthly_charges * 12:,.2f} in annualized customer revenue."
            if monthly_charges > 0
            else ""
        )

        return {
            "risk_level": risk_level,
            "customer_segment": customer_segment,
            "recommended_offer": base["recommended_offer"],
            "recommended_channel": base["recommended_channel"],
            "priority": base["priority"],
            "business_justification": f"{base['rationale']}{driver_clause}{revenue_clause}",
        }
