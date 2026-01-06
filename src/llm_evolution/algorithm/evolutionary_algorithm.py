from typing import List, TypeVar, Generic, Optional
from dataclasses import dataclass
from ..interfaces.crossover import Crossover
from ..interfaces.mutation import Mutation
from ..interfaces.selection import Selection
from ..interfaces.evaluation import Evaluation
from ..interfaces.initial_population import InitialPopulation
from ..interfaces.finish_condition import FinishCondition

T = TypeVar("T")


@dataclass
class EvolutionResult(Generic[T]):
    """Result of the evolutionary algorithm."""

    best_instance: T
    best_fitness: float
    population: List[T]
    generation: int


class EvolutionaryAlgorithm(Generic[T]):
    """
    Standard Evolutionary Algorithm implementation.

    This class orchestrates the evolutionary process by coordinating
    initialization, evaluation, crossover, mutation, and selection.
    """

    def __init__(
        self,
        initial_population: InitialPopulation[T],
        evaluation: Evaluation[T],
        selection: Selection[T],
        crossover: Optional[Crossover[T]] = None,
        mutation: Optional[Mutation[T]] = None,
        population_size: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
    ):
        self.initial_population = initial_population
        self.evaluation = evaluation
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def run(self, finish_condition: FinishCondition[T]) -> EvolutionResult[T]:
        """
        Execute the evolutionary algorithm until the finish condition is met.
        """
        import random

        population = self.initial_population(self.population_size)
        generation = 0

        while True:
            # 1. Evaluate population
            fitness_scores = [self.evaluation(ind) for ind in population]

            # 2. Check finish condition
            if finish_condition(population, generation, fitness_scores):
                break

            # 3. Generate offspring
            offspring: List[T] = []

            # Crossover
            if self.crossover and len(population) >= 2:
                num_crossovers = int(self.population_size * self.crossover_rate / 2)
                for _ in range(num_crossovers):
                    parents = random.sample(population, 2)
                    offspring.extend(self.crossover(parents))

            # Mutation
            if self.mutation:
                num_mutations = int(self.population_size * self.mutation_rate)
                to_mutate = random.sample(
                    population + offspring,
                    min(num_mutations, len(population) + len(offspring)),
                )
                for ind in to_mutate:
                    offspring.append(self.mutation(ind))

            # 4. Selection
            # Evaluate offspring for selection
            offspring_fitness = [self.evaluation(ind) for ind in offspring]

            combined_fitness = fitness_scores + offspring_fitness

            population = self.selection(population, offspring, combined_fitness)
            generation += 1

            # Ensure population size is maintained if selection didn't handle it
            if len(population) > self.population_size:
                indexed_fitness = list(
                    enumerate([self.evaluation(ind) for ind in population])
                )
                indexed_fitness.sort(key=lambda x: x[1], reverse=True)
                population = [
                    population[i] for i, _ in indexed_fitness[: self.population_size]
                ]

        final_fitness = [self.evaluation(ind) for ind in population]
        best_idx = final_fitness.index(max(final_fitness))

        return EvolutionResult(
            best_instance=population[best_idx],
            best_fitness=final_fitness[best_idx],
            population=population,
            generation=generation,
        )
