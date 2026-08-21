from ..models import RefundState
from ..prompts import MISSING_IDENTITY_FOLLOWUP
from ..utils import normalized_text


def respond(state: RefundState) -> RefundState:
    return {"followup": normalized_text(state.get("followup")) or MISSING_IDENTITY_FOLLOWUP}
