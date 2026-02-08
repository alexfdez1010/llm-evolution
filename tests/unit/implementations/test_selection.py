import unittest
import sys
import os

# Add src to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src")))

from llm_evolution.implementations.selection import TournamentSelection


class TestTournamentSelection(unittest.TestCase):
    def test_tournament_selection_basic(self):
        """Test that selection returns the correct number of individuals."""
        population = ["p1", "p2"]
        offspring = ["o1", "o2"]
        fitness_scores = [1.0, 2.0, 3.0, 4.0]  # p1, p2, o1, o2

        # Target size 2, tournament size 2
        selection = TournamentSelection(tournament_size=2, population_size=2)
        survivors = selection(population, offspring, fitness_scores)

        self.assertEqual(len(survivors), 2)
        # Since we are selecting from a pool of 4, the survivors should be from the pool.
        for s in survivors:
            self.assertIn(s, population + offspring)

    def test_tournament_selection_logic(self):
        """Test that better individuals are preferred (probabilistically)."""
        # Create a scenario where one individual is vastly superior
        population = ["loser"]
        offspring = ["winner"]
        fitness_scores = [0.0, 1000.0]

        # Tournament size 2 implies both will be picked in a tournament of 2,
        # and 'winner' should always win.
        selection = TournamentSelection(tournament_size=2, population_size=5)
        survivors = selection(population, offspring, fitness_scores)

        self.assertEqual(len(survivors), 5)
        for s in survivors:
            self.assertEqual(s, "winner")

    def test_tournament_selection_empty_pool(self):
        """Test behavior with empty population."""
        selection = TournamentSelection(tournament_size=3, population_size=10)
        survivors = selection([], [], [])
        self.assertEqual(survivors, [])

    def test_mismatch_error(self):
        """Test that ValueError is raised on size mismatch."""
        selection = TournamentSelection()
        with self.assertRaises(ValueError):
            selection(["p1"], [], [1.0, 2.0])

if __name__ == '__main__':
    unittest.main()
