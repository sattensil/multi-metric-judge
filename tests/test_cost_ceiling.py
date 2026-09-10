from __future__ import annotations

from packing.criteria import HELPSTEER2_CRITERIA
from packing.prompt import JudgePrompt, token_estimate, token_upper_bound
from packing.schema import schema_token_estimate, schema_token_upper_bound


def test_prompt_authorization_bound_dominates_the_heuristic():
    prompt = JudgePrompt(
        system="System framing",
        user="Unicode cannot make the bound smaller: 你好 👩🏽‍💻",
        group=HELPSTEER2_CRITERIA[:1],
    )
    assert token_upper_bound(prompt) >= token_estimate(prompt)


def test_schema_authorization_bound_dominates_the_heuristic():
    group = HELPSTEER2_CRITERIA[:3]
    assert schema_token_upper_bound(group) >= schema_token_estimate(group)
