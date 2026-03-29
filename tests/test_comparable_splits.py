from bias_dynamics.comparable_splits import extract_triplet


def test_extract_triplet_direct_keys():
    row = {"prompt": "Q", "chosen": "A", "rejected": "B"}
    p, c, r = extract_triplet(row)
    assert (p, c, r) == ("Q", "A", "B")


def test_extract_triplet_message_fallback():
    row = {
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Tell me a joke"},
        ],
        "chosen": "Sure! joke",
        "rejected": "No",
    }
    p, c, r = extract_triplet(row)
    assert p == "Tell me a joke"
    assert c == "Sure! joke"
    assert r == "No"


def test_extract_triplet_stereoset_schema():
    row = {
        "context": "The engineer said that",
        "target": "engineer",
        "sentences": [
            {"gold_label": "stereotype", "sentence": "he is naturally better at logic."},
            {"gold_label": "anti-stereotype", "sentence": "she can be excellent at logic."},
        ],
    }
    p, c, r = extract_triplet(row)
    assert p.startswith("The engineer said that")
    assert c == "she can be excellent at logic."
    assert r == "he is naturally better at logic."
