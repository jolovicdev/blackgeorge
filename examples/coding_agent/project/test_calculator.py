from calculator import add, average, divide


def test_add() -> None:
    assert add(2, 3) == 5


def test_average() -> None:
    assert average([2.0, 4.0]) == 3.0


def test_divide() -> None:
    assert divide(10, 2) == 5
