"""Unit tests for the ElitistSelection implementation."""

import pytest

from llm_evolution.implementations.elitist_selection import ElitistSelection
from llm_evolution.interfaces.selection import Selection


def test_elitist_selection_implements_selection_protocol():
    """ElitistSelection should satisfy the Selection protocol."""
    selector = ElitistSelection(max_size=2)
    assert isinstance(selector, Selection)


def test_elitist_selection_picks_top_unique_individuals():
    """Selection should keep top-scored unique individuals in score order."""
    selector = ElitistSelection(max_size=3)

    population = ["a", "b"]
    offspring = ["a", "c", "d"]
    fitness_scores = [1.0, 2.0, 3.0, 2.5, 1.5]

    result = selector(population, offspring, fitness_scores)

    assert result == ["a", "c", "b"]


def test_elitist_selection_limits_output_to_max_size():
    """Selection should never return more than max_size individuals."""
    selector = ElitistSelection(max_size=2)

    result = selector(
        population=["p1", "p2", "p3"],
        offspring=["o1"],
        fitness_scores=[1.0, 3.0, 2.0, 4.0],
    )

    assert result == ["o1", "p2"]
    assert len(result) == 2


def test_elitist_selection_falls_back_to_first_population_item():
    """If no individuals are selected, selector should return first population item."""
    selector = ElitistSelection(max_size=1)

    class UnhashableIndividual:
        __hash__ = None

    population = [UnhashableIndividual()]

    result = selector(population=population, offspring=[], fitness_scores=[1.0])

    assert result == [population[0]]


def test_elitist_selection_returns_empty_if_no_candidates():
    """If both population and offspring are empty, selector returns empty list."""
    selector = ElitistSelection(max_size=1)
    assert selector(population=[], offspring=[], fitness_scores=[]) == []


def test_elitist_selection_raises_on_length_mismatch():
    """Selection should fail fast when score count does not match candidates."""
    selector = ElitistSelection(max_size=1)

    with pytest.raises(ValueError, match="Mismatch: 2 individuals vs 1 scores"):
        selector(population=["a"], offspring=["b"], fitness_scores=[1.0])


def test_elitist_selection_raises_on_invalid_max_size():
    """max_size must be strictly positive."""
    with pytest.raises(ValueError, match="max_size must be greater than 0"):
        ElitistSelection(max_size=0)
