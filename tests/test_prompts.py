from bias_dynamics.prompts import build_identity_conditioned_prompts


def test_prompt_counts():
    prompts = build_identity_conditioned_prompts(50)
    assert set(prompts.keys()) == {"gender", "race", "religion", "profession"}
    for pairs in prompts.values():
        assert len(pairs) >= 50
