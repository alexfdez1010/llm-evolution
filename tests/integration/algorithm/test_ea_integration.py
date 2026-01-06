import random
from typing import List
from llm_evolution.algorithm.evolutionary_algorithm import EvolutionaryAlgorithm
from llm_evolution.interfaces.crossover import crossover_fn
from llm_evolution.interfaces.mutation import mutation_fn
from llm_evolution.interfaces.evaluation import evaluation_fn
from llm_evolution.interfaces.selection import selection_fn
from llm_evolution.interfaces.initial_population import initial_population_fn
from llm_evolution.interfaces.finish_condition import finish_condition_fn


def test_evolutionary_algorithm_integration():
    """
    Integration test for the full evolutionary pipeline.
    Goal: Find the integer that maximizes the value (simple optimization).
    """

    @initial_population_fn
    def init_pop(size: int) -> List[int]:
        return [random.randint(0, 10) for _ in range(size)]

    @evaluation_fn
    def evaluate(instance: int) -> float:
        return float(instance)

    @selection_fn
    def select(pop: List[int], off: List[int], scores: List[float]) -> List[int]:
        # Simple elitist selection: pick the best ones from combined population
        combined = pop + off
        # Sort by fitness (scores are for combined population)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        selected_indices = [i for i, _ in indexed_scores[: len(pop)]]
        return [combined[i] for i in selected_indices]

    @crossover_fn
    def crossover(parents: List[int]) -> List[int]:
        # Average of parents
        return [sum(parents) // len(parents)]

    @mutation_fn
    def mutate(instance: int) -> int:
        # Small random change
        return instance + random.choice([-1, 1])

    @finish_condition_fn
    def finish(pop: List[int], gen: int, scores: List[float]) -> bool:
        # Finish if we reach a target value or max generations
        return max(scores) >= 100 or gen >= 50

    # Instantiate algorithm
    ea = EvolutionaryAlgorithm(
        initial_population=init_pop,
        evaluation=evaluate,
        selection=select,
        finish_condition=finish,
        crossover=crossover,
        mutation=mutate,
        population_size=20,
    )

    # Run algorithm
    result = ea.run(log=True)

    # Assertions
    assert result.best_instance is not None
    assert result.best_fitness >= 0
    assert len(result.population) == 20
    assert result.generation >= 0


def test_ea_no_crossover_no_mutation():
    """Test algorithm with only selection and no genetic operators."""

    @initial_population_fn
    def init_pop(size: int) -> List[int]:
        return [5] * size

    @evaluation_fn
    def evaluate(instance: int) -> float:
        return float(instance)

    @selection_fn
    def select(pop: List[int], off: List[int], scores: List[float]) -> List[int]:
        return pop

    @finish_condition_fn
    def finish(pop: List[int], gen: int, scores: List[float]) -> bool:
        return gen >= 5

    ea = EvolutionaryAlgorithm(
        initial_population=init_pop,
        evaluation=evaluate,
        selection=select,
        finish_condition=finish,
        population_size=10,
    )
    result = ea.run()
    assert result.generation == 5
    assert all(ind == 5 for ind in result.population)


def test_ea_immediate_finish():
    """Test algorithm that finishes at generation 0."""

    @initial_population_fn
    def init_pop(size: int) -> List[int]:
        return [1] * size

    @evaluation_fn
    def evaluate(instance: int) -> float:
        return 1.0

    @selection_fn
    def select(pop: List[int], off: List[int], scores: List[float]) -> List[int]:
        return pop

    @finish_condition_fn
    def finish(pop: List[int], gen: int, scores: List[float]) -> bool:
        return True

    ea = EvolutionaryAlgorithm(
        initial_population=init_pop,
        evaluation=evaluate,
        selection=select,
        finish_condition=finish,
        population_size=5,
    )
    result = ea.run()
    assert result.generation == 0
    assert len(result.population) == 5


def test_ea_population_size_one():
    """Test algorithm with a single individual (minimal population)."""

    @initial_population_fn
    def init_pop(size: int) -> List[int]:
        return [10]

    @evaluation_fn
    def evaluate(instance: int) -> float:
        return float(instance)

    @selection_fn
    def select(pop: List[int], off: List[int], scores: List[float]) -> List[int]:
        combined = pop + off
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return [combined[indexed_scores[0][0]]]

    @mutation_fn
    def mutate(instance: int) -> int:
        return instance + 1

    @finish_condition_fn
    def finish(pop: List[int], gen: int, scores: List[float]) -> bool:
        return gen >= 10

    ea = EvolutionaryAlgorithm(
        initial_population=init_pop,
        evaluation=evaluate,
        selection=select,
        finish_condition=finish,
        mutation=mutate,
        population_size=1,
    )
    result = ea.run()
    assert len(result.population) == 1
    assert result.best_fitness >= 10


def test_ea_complex_type_integration():
    """Test algorithm with a non-primitive type (dictionary)."""
    from dataclasses import dataclass

    @dataclass
    class Individual:
        genes: List[float]

    @initial_population_fn
    def init_pop(size: int) -> List[Individual]:
        return [Individual([random.random() for _ in range(2)]) for _ in range(size)]

    @evaluation_fn
    def evaluate(instance: Individual) -> float:
        return sum(instance.genes)

    @selection_fn
    def select(
        pop: List[Individual], off: List[Individual], scores: List[float]
    ) -> List[Individual]:
        combined = pop + off
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return [combined[i] for i, _ in indexed_scores[: len(pop)]]

    @mutation_fn
    def mutate(instance: Individual) -> Individual:
        new_genes = [g + random.uniform(-0.1, 0.1) for g in instance.genes]
        return Individual(new_genes)

    @finish_condition_fn
    def finish(pop: List[Individual], gen: int, scores: List[float]) -> bool:
        return gen >= 20

    ea = EvolutionaryAlgorithm(
        initial_population=init_pop,
        evaluation=evaluate,
        selection=select,
        finish_condition=finish,
        mutation=mutate,
        population_size=10,
    )
    result = ea.run()
    assert isinstance(result.best_instance, Individual)
    assert len(result.population) == 10
