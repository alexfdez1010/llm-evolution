from typing import List
from llm_evolution.interfaces.crossover import crossover_fn, Crossover
from llm_evolution.interfaces.mutation import mutation_fn, Mutation
from llm_evolution.interfaces.evaluation import evaluation_fn, Evaluation
from llm_evolution.interfaces.selection import selection_fn, Selection
from llm_evolution.interfaces.initial_population import (
    initial_population_fn,
    InitialPopulation,
)
from llm_evolution.interfaces.finish_condition import (
    finish_condition_fn,
    FinishCondition,
)


def test_crossover_decorator():
    @crossover_fn
    def my_crossover(parents: List[int]) -> List[int]:
        return [sum(parents)]

    assert isinstance(my_crossover, Crossover)
    assert my_crossover([1, 2]) == [3]


def test_mutation_decorator():
    @mutation_fn
    def my_mutation(instance: int) -> int:
        return instance + 1

    assert isinstance(my_mutation, Mutation)
    assert my_mutation(1) == 2


def test_evaluation_decorator():
    @evaluation_fn
    def my_evaluation(instance: int) -> float:
        return float(instance * 2)

    assert isinstance(my_evaluation, Evaluation)
    assert my_evaluation(5) == 10.0


def test_selection_decorator():
    @selection_fn
    def my_selection(pop: List[int], off: List[int], scores: List[float]) -> List[int]:
        return [pop[0]]

    assert isinstance(my_selection, Selection)
    assert my_selection([1], [2], [1.0, 2.0]) == [1]


def test_initial_population_decorator():
    @initial_population_fn
    def my_init(size: int) -> List[int]:
        return [0] * size

    assert isinstance(my_init, InitialPopulation)
    assert my_init(3) == [0, 0, 0]


def test_finish_condition_decorator():
    @finish_condition_fn
    def my_finish(pop: List[int], gen: int, scores: List[float]) -> bool:
        return gen > 5

    assert isinstance(my_finish, FinishCondition)
    assert my_finish([], 6, []) is True
    assert my_finish([], 4, []) is False
