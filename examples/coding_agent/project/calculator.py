from __future__ import annotations


def add(a: int, b: int) -> int:
    return a + b


def divide(a: int, b: int) -> float:
    if b == 0:
        return 0
    return a / b


def average(values: list[float]) -> float:
    if not values:
        return 0
    total = 0.0
    for value in values:
        total += value
    return total / len(values)
