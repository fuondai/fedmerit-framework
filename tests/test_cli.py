from fedmerit.cli import _finite_number


def test_finite_number_rejects_integer_overflow() -> None:
    try:
        _finite_number(10**1000, "value")
    except ValueError as exc:
        assert str(exc) == "value must be a finite number"
    else:
        raise AssertionError("an overflowing integer must be rejected")
