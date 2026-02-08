import logging
import random
from dataclasses import dataclass
from typing import Generic, TypeVar

from ..interfaces.crossover import Crossover
from ..interfaces.evaluation import Evaluation
from ..interfaces.finish_condition import FinishCondition
from ..interfaces.initial_population import InitialPopulation
from ..interfaces.mutation import Mutation
from ..interfaces.selection import Selection

T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass
class EvolutionResult(Generic[T]):
    """
    Result of the evolutionary algorithm.

    Attributes:
        best_instance: The best individual found during evolution.
        best_fitness: The fitness score of the best individual.
        population: The final population after evolution.
        generation: The number of generations executed.
    """

    best_instance: T
    best_fitness: float
    population: list[T]
    generation: int


class EvolutionaryAlgorithm(Generic[T]):
    """
    Standard Evolutionary Algorithm implementation.

    This class orchestrates the evolutionary process by coordinating
    initialization, evaluation, crossover, mutation, and selection.

    Attributes:
        initial_population: Strategy to generate the initial population.
        evaluation: Strategy to evaluate individual fitness.
        selection: Strategy to select survivors for the next generation.
        finish_condition: Strategy to determine when evolution should stop.
        crossover: Optional strategy for crossover operations.
        mutation: Optional strategy for mutation operations.
        population_size: The number of individuals in the population.
    """

    def __init__(
        self,
        initial_population: InitialPopulation[T],
        evaluation: Evaluation[T],
        selection: Selection[T],
        finish_condition: FinishCondition[T],
        crossover: Crossover[T] | None = None,
        mutation: Mutation[T] | None = None,
        population_size: int = 100,
    ):
        """
        Initialize the evolutionary algorithm.

        Args:
            initial_population: Strategy to generate the initial population.
            evaluation: Strategy to evaluate individual fitness.
            selection: Strategy to select survivors for the next generation.
            finish_condition: Strategy to determine when evolution should stop.
            crossover: Optional strategy for crossover operations.
            mutation: Optional strategy for mutation operations.
            population_size: The number of individuals in the population.
        """
        self.initial_population = initial_population
        self.evaluation = evaluation
        self.selection = selection
        self.finish_condition = finish_condition
        self.crossover = crossover
        self.mutation = mutation
        self.population_size = population_size

    def run(self, log: bool = False) -> EvolutionResult[T]:
        """
        Execute the evolutionary algorithm.

        Args:
            log: Whether to enable logging of the evolutionary process.
                 Note: This flag controls whether the algorithm emits log messages.
                 You must configure the logging system (e.g., logging.basicConfig)
                 to actually see the output.

        Returns:
            EvolutionResult: The result containing the best individual and final population.
        """
        if log:
            logger.info(
                "Starting evolutionary algorithm with population size %d",
                self.population_size,
            )

        population = self.initial_population(self.population_size)
        generation = 0

        while True:
            fitness_scores = [self.evaluation(ind) for ind in population]
            best_fitness = max(fitness_scores)

            if log:
                logger.info(
                    "Generation %d: Best fitness = %.4f", generation, best_fitness
                )

            if self.finish_condition(population, generation, fitness_scores):
                if log:
                    logger.info("Finish condition met at generation %d", generation)
                break

            offspring: list[T] = []

            if self.crossover and len(population) >= 2:
                parents_list = [
                    random.sample(population, 2)
                    for _ in range(self.population_size // 2)
                ]
                for parents in parents_list:
                    offspring.extend(self.crossover(parents))

            if self.mutation:
                to_mutate = random.sample(
                    population + offspring,
                    min(len(population), len(population) + len(offspring)),
                )
                for ind in to_mutate:
                    mutated = self.mutation(ind)
                    if mutated is not None:
                        offspring.append(mutated)

            offspring_fitness = [self.evaluation(ind) for ind in offspring]
            combined_fitness = fitness_scores + offspring_fitness

            population = self.selection(population, offspring, combined_fitness)
            generation += 1

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

        if log:
            logger.info(
                "Evolution finished. Best fitness: %.4f", final_fitness[best_idx]
            )

        return EvolutionResult(
            best_instance=population[best_idx],
            best_fitness=final_fitness[best_idx],
            population=population,
            generation=generation,
        )
