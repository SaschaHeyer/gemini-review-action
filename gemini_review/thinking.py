"""
Description: Reasoning budget for the review call.

The review is a single generateContent call with a response schema and, by default, no
reasoning budget. On a real pull request that produced 213 output tokens for the whole
review, and the model asserted the opposite of what the diff showed: it praised commit-SHA
pinning on a file where two of three actions were on floating tags. The context was there
(97.7% cached repo context plus the diff). It did not look hard.

`thinking_budget` gives the model room to reason before it answers. Off by default so no
existing user's cost or behaviour changes without them asking for it.

Cost note: thinking tokens bill at the OUTPUT rate, and the API reports them separately as
`thoughts_token_count`. They are priced in `pricing.estimate_cost`, so the telemetry does
not silently understate a run with reasoning enabled.
"""

import os
import sys
from typing import Any

# -1 asks the model to choose its own budget. 0 disables reasoning explicitly. Anything
# above 0 is a token ceiling.
DYNAMIC = -1
DISABLED = 0


def resolve_thinking_budget(config: dict | None = None) -> int | None:
    """The configured budget, or None to leave the API default untouched."""
    config = config or {}
    raw = os.environ.get("GEMINI_THINKING_BUDGET", config.get("thinking_budget"))
    if raw is None or str(raw).strip() == "":
        return None
    try:
        budget = int(str(raw).strip())
    except (TypeError, ValueError):
        print(
            f"Notice: ignoring non-numeric thinking_budget '{raw}'. Use a token count, "
            f"{DYNAMIC} for dynamic, or {DISABLED} to disable.",
            file=sys.stderr,
        )
        return None
    if budget < DYNAMIC:
        print(f"Notice: ignoring thinking_budget '{budget}', which is below {DYNAMIC}.", file=sys.stderr)
        return None
    return budget


def build_thinking_config(types_module: Any, config: dict | None = None) -> Any | None:
    """A ThinkingConfig for the review call, or None to leave it unset."""
    budget = resolve_thinking_budget(config)
    if budget is None:
        return None
    label = "dynamic" if budget == DYNAMIC else ("disabled" if budget == DISABLED else f"{budget} tokens")
    print(f"Reasoning budget: {label}.", file=sys.stderr)
    return types_module.ThinkingConfig(thinking_budget=budget)
