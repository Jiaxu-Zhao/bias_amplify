from bias_dynamics.real_eval import distinct_n


def test_distinct_n():
    texts = ["a b c", "a b d"]
    d1 = distinct_n(texts, 1)
    d2 = distinct_n(texts, 2)
    assert 0 < d1 <= 1
    assert 0 < d2 <= 1
    assert d2 >= d1 / 2
