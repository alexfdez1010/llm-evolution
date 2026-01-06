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
        crossover=crossover,
        mutation=mutate,
        population_size=20,
        mutation_rate=0.2,
    )

    # Run algorithm
    result = ea.run(finish_condition=finish)

    # Assertions
    assert result.best_instance is not None
    assert result.best_fitness >= 0
    assert len(result.population) == 20
    assert result.generation >= 0

    # Since it's a simple maximization of integers, the fitness should ideally increase or stay same
    initial_best_fitness = max([evaluate(ind) for ind in init_pop(20)])
    assert result.best_fitness >= initial_best_fitness
