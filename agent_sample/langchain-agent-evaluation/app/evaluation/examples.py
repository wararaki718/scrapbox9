from .models import EvaluationExample


EVALUATION_EXAMPLES = [
    EvaluationExample(
        name="james-brown-lookup",
        question="What James Brown songs do you have?",
        expected_response=["James Brown", "Sex Machine", "Cold Sweat"],
        expected_trajectory=["intent_classifier", "question_answering_agent", "tools", "lookup_track"],
        expected_route="question_answering_agent",
    ),
    EvaluationExample(
        name="incomplete-aaron-mitchell-refund",
        question="Please refund Aaron Mitchell's purchase of Black Dog.",
        expected_response=["phone number", "refund", "purchase"],
        expected_trajectory=["intent_classifier", "refund_agent", "respond"],
        expected_route="refund_agent",
    ),
    EvaluationExample(
        name="aaron-led-zeppelin-purchase-lookup",
        question=(
            "Please help refund Aaron Mitchell's Led Zeppelin purchase. "
            "His phone number is +1 (204) 452-6452."
        ),
        expected_response=["How Many More Times", "What Is And What Should Never Be", "2009-08-06"],
        expected_trajectory=["intent_classifier", "refund_agent", "lookup"],
        expected_route="refund_agent",
    ),
    EvaluationExample(
        name="wish-you-were-here-pink-floyd-lookup",
        question="Do you have the album Wish You Were Here by Pink Floyd?",
        expected_response=["no matches", "Wish You Were Here", "Pink Floyd"],
        expected_trajectory=["intent_classifier", "question_answering_agent", "tools", "lookup_album"],
        expected_route="question_answering_agent",
    ),
    EvaluationExample(
        name="invoice-237-refund",
        question="Refund invoice 237.",
        expected_response="Previewed a refund total of $0.99. No database changes were made because env is not prod.",
        expected_trajectory=["intent_classifier", "refund_agent", "refund"],
        expected_route="refund_agent",
    ),
]
