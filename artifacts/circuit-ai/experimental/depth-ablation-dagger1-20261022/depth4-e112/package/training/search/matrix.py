"""Small zero-sum matrix-game solver for simultaneous search nodes."""

from __future__ import annotations

from itertools import combinations
from math import isfinite
from typing import Sequence


def _linear_solve(matrix: list[list[float]], rhs: list[float], eps: float = 1e-10) -> list[float] | None:
    n = len(rhs)
    work = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) <= eps:
            return None
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(n):
            if row == column:
                continue
            factor = work[row][column]
            if factor:
                work[row] = [a - factor * b for a, b in zip(work[row], work[column])]
    return [work[i][-1] for i in range(n)]


def _maximizer(payoffs: list[list[float]], eps: float) -> tuple[list[float], float] | None:
    rows, columns = len(payoffs), len(payoffs[0])
    constraints: list[tuple[list[float], float]] = []
    for i in range(rows):
        constraints.append(([1.0 if j == i else 0.0 for j in range(rows)] + [0.0], 0.0))
    for j in range(columns):
        constraints.append(([payoffs[i][j] for i in range(rows)] + [-1.0], 0.0))
    equality = [1.0] * rows + [0.0]
    best: tuple[list[float], float] | None = None
    for active in combinations(range(len(constraints)), rows):
        equations = [equality] + [constraints[i][0] for i in active]
        values = [1.0] + [constraints[i][1] for i in active]
        solution = _linear_solve(equations, values)
        if solution is None:
            continue
        probability, value = solution[:-1], solution[-1]
        if any(p < -eps for p in probability) or abs(sum(probability) - 1.0) > 1e-7:
            continue
        if any(sum(probability[i] * payoffs[i][j] for i in range(rows)) < value - eps for j in range(columns)):
            continue
        if best is None or value > best[1] + eps:
            best = ([max(0.0, p) for p in probability], value)
    return best


def _minimizer(payoffs: list[list[float]], eps: float) -> tuple[list[float], float] | None:
    rows, columns = len(payoffs), len(payoffs[0])
    constraints: list[tuple[list[float], float]] = []
    for j in range(columns):
        constraints.append(([1.0 if k == j else 0.0 for k in range(columns)] + [0.0], 0.0))
    for i in range(rows):
        constraints.append(([-payoffs[i][j] for j in range(columns)] + [1.0], 0.0))
    equality = [1.0] * columns + [0.0]
    best: tuple[list[float], float] | None = None
    for active in combinations(range(len(constraints)), columns):
        equations = [equality] + [constraints[i][0] for i in active]
        values = [1.0] + [constraints[i][1] for i in active]
        solution = _linear_solve(equations, values)
        if solution is None:
            continue
        probability, value = solution[:-1], solution[-1]
        if any(p < -eps for p in probability) or abs(sum(probability) - 1.0) > 1e-7:
            continue
        if any(sum(payoffs[i][j] * probability[j] for j in range(columns)) > value + eps for i in range(rows)):
            continue
        if best is None or value < best[1] - eps:
            best = ([max(0.0, p) for p in probability], value)
    return best


def solve_zero_sum(payoffs: Sequence[Sequence[float]], eps: float = 1e-8) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    """Return value and optimal row/column mixed strategies for a small matrix.

    The row player maximizes and the column player minimizes. This vertex
    enumeration is intended for the arena's at-most-4x4 legal action matrices,
    not as a general-purpose LP package.
    """
    matrix = [[float(value) for value in row] for row in payoffs]
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("payoff matrix must be non-empty and rectangular")
    if any(not isfinite(value) for row in matrix for value in row):
        raise ValueError("payoffs must be finite")
    row_solution = _maximizer(matrix, eps)
    column_solution = _minimizer(matrix, eps)
    if row_solution is None or column_solution is None:
        raise ArithmeticError("could not solve zero-sum payoff matrix")
    value0, value1 = row_solution[1], column_solution[1]
    if abs(value0 - value1) > 1e-6:
        raise ArithmeticError(f"primal/dual equilibrium values disagree: {value0} vs {value1}")
    row = row_solution[0]
    column = column_solution[0]
    row_total, column_total = sum(row), sum(column)
    row = [p / row_total for p in row]
    column = [p / column_total for p in column]
    return (value0 + value1) / 2.0, tuple(row), tuple(column)
