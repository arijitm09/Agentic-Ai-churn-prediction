"""
Marketing Agent
Generates personalized retention campaign content (Email, SMS/In-App, and Call Script)
based on Customer Segment and Recommended Offer.
"""

from typing import Dict


class MarketingAgent:
    """Specialized Agent for Personalized Retention Campaign Generation."""

    def generate_campaign(
        self,
        customer_segment: str,
        recommended_offer: str,
        customer_name: str = "Valued Customer",
        recommended_channel: str = "Phone Call + Email",
        tenure_months: int = 12,
        monthly_charges: float = 85.0,
    ) -> Dict[str, str]:
        """
        Generate personalized retention campaign content based on:
        - Customer Segment
        - Recommended Offer
        """
        first_name = customer_name.split()[0] if customer_name else "Valued Customer"

        if customer_segment == "At Risk":
            subject = f"Exclusive Account Benefit for {first_name}: Claim Your {recommended_offer} Today"
            opening = (
                f"Dear {customer_name},\n\n"
                f"Thank you for being part of our community over the past {tenure_months} months. "
                f"We recently reviewed your account plan and want to ensure you are getting the absolute "
                f"best value and reliability from your service."
            )
            offer_pitch = (
                f"As a priority account benefit, we have unlocked a special **{recommended_offer}** "
                f"on your plan (saving you up to ${monthly_charges * 0.20 * 12:,.2f} annually) along with "
                f"a complimentary account health check from our Senior Retention Specialist team."
            )
            cta = "Click 'Activate My Savings' below or expect a brief courtesy call from your dedicated specialist today."
        elif customer_segment == "Loyal Customers":
            subject = f"Celebrating {tenure_months} Months Together, {first_name} — Your {recommended_offer} Is Ready!"
            opening = (
                f"Dear {customer_name},\n\n"
                f"Your {tenure_months}-month partnership means the world to us. Because you are one of our "
                f"most valued Loyal Customers, we are proactively upgrading your experience."
            )
            offer_pitch = (
                f"You have been selected for our **{recommended_offer}** reward package at zero additional cost—"
                f"designed to give you faster performance, priority technical routing, and seamless connectivity."
            )
            cta = "Activate your loyalty upgrade in one click inside your account dashboard or via the link below."
        elif customer_segment == "Champions":
            subject = f"VIP Executive Invitation for {customer_name}: {recommended_offer}"
            opening = (
                f"Dear {customer_name},\n\n"
                f"With {tenure_months} months of premier partnership, you are in the top tier of our Champions community. "
                f"On behalf of our executive leadership team, thank you for your continued trust."
            )
            offer_pitch = (
                f"It is my pleasure to extend a private invitation to our **{recommended_offer}**—complete with "
                f"white-glove concierge support, priority hardware refreshes, and exclusive executive benefits."
            )
            cta = "Your Dedicated Account Manager will reach out via VIP Concierge Call to tailor your activation."
        else:  # Price Sensitive
            subject = f"Lower Your Monthly Bill Immediately, {first_name}: {recommended_offer}"
            opening = (
                f"Dear {customer_name},\n\n"
                f"We know how important predictable, affordable monthly billing is. To help you save more every month "
                f"without sacrificing the service you rely on, we have prepared a custom bill relief offer for your account."
            )
            offer_pitch = (
                f"Activate your **{recommended_offer}** right now to reduce your upcoming billing statements with "
                f"zero hidden fees or service interruptions."
            )
            cta = "Reply YES to our SMS or click the button below to apply your bill credit immediately."

        email_body = (
            f"{opening}\n\n"
            f"{offer_pitch}\n\n"
            f"What This Includes for Your Account:\n"
            f"• Personalized Offer: {recommended_offer}\n"
            f"• Tailored Segment Tier: {customer_segment} Priority Program\n"
            f"• Dedicated Support Channel: {recommended_channel}\n"
            f"• 100% Price-Lock & Service Satisfaction Guarantee\n\n"
            f"{cta}\n\n"
            f"Warm regards,\n"
            f"Enterprise Customer Success & Retention Team"
        )

        sms_copy = (
            f"Hi {first_name}! As a valued {customer_segment} member, you're pre-approved for "
            f"{recommended_offer}. Activate in 1 click: https://retain.enterprise.ai/claim or reply HELP to connect."
        )

        call_script = (
            f"1. Greet {customer_name} warmly and acknowledge their {tenure_months}-month tenure.\n"
            f"2. Address service/billing value proactively without pressure.\n"
            f"3. Present the pre-authorized '{recommended_offer}' tailored for the {customer_segment} segment.\n"
            f"4. Confirm immediate activation on their next billing cycle and verify technical satisfaction."
        )

        full_text = (
            f"### Campaign Channel: {recommended_channel}\n"
            f"**Target Segment:** {customer_segment} | **Featured Offer:** {recommended_offer}\n\n"
            f"**Email Subject Line:**\n`{subject}`\n\n"
            f"**Personalized Email Copy:**\n{email_body}\n\n"
            f"**SMS / In-App Notification Copy:**\n> {sms_copy}\n\n"
            f"**Outbound Specialist Call Script:**\n{call_script}"
        )

        return {
            "customer_segment": customer_segment,
            "recommended_offer": recommended_offer,
            "recommended_channel": recommended_channel,
            "email_subject": subject,
            "email_body": email_body,
            "sms_copy": sms_copy,
            "call_script": call_script,
            "campaign_text": full_text,
        }
