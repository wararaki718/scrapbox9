REFUND_INSTRUCTIONS = """You extract refund workflow information for a music store support agent.
Identify only details that are explicitly stated or directly implied by the customer.
Prefer invoice_id when the user names a full invoice.
Prefer invoice_line_ids when the user names specific purchased items or selected purchase rows.
Collect customer_first_name, customer_last_name, and customer_phone whenever present.
Collect track_name, album_title, artist_name, and purchase_date_iso_8601 when they help identify a purchase.
Write purchase_date_iso_8601 in ISO 8601 form when you can determine it.
If the request still needs identity details before a purchase lookup can continue, write a short followup question.
Do not invent identifiers, names, phone numbers, products, or dates.
Return structured data only."""

MISSING_IDENTITY_FOLLOWUP = (
    "To process the refund, please share the customer's first name, last name, and phone number "
    "so I can look up the purchase."
)
NO_PURCHASES_FOLLOWUP = (
    "I couldn't find any matching purchases. Please verify the customer's first name, last name, "
    "phone number, and any track, album, artist, or purchase date details."
)
