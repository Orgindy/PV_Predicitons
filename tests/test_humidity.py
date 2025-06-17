from humidity import compute_relative_humidity


def test_relative_humidity_basic():
    rh = compute_relative_humidity(293.15, 283.15)
    assert 0 < rh < 100
