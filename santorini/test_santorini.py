import unittest

import numpy as np

from santorini.SantoriniGame import SantoriniGame


class TestSantoriniRules(unittest.TestCase):
    def setUp(self):
        np.random.seed(7)
        self.game = SantoriniGame(5)

    def empty_board(self):
        board = np.zeros((2, 5, 5), dtype=int)
        board[0, 0, 0] = 2
        board[0, 4, 4] = -1
        board[0, 4, 3] = -2
        return board

    def is_outer_edge(self, location):
        x, y = location
        return x == 0 or y == 0 or x == 4 or y == 4

    def test_random_starts_reject_players_with_both_workers_on_outer_edge(self):
        for _ in range(200):
            board = self.game.getInitBoard()
            player_locations = self.game.getCharacterLocations(board, 1)
            opponent_locations = self.game.getCharacterLocations(board, -1)

            self.assertEqual(len({*player_locations, *opponent_locations}), 4)
            self.assertFalse(all(self.is_outer_edge(loc) for loc in player_locations))
            self.assertFalse(all(self.is_outer_edge(loc) for loc in opponent_locations))

    def test_cannot_move_onto_dome_even_from_level_three(self):
        board = self.empty_board()
        board[0, 2, 2] = 1
        board[1, 2, 2] = 3
        board[1, 2, 3] = 4

        valids = self.game.getValidMoves(board, 1)

        east_move_actions = valids[4 * 8:5 * 8]
        self.assertEqual(int(east_move_actions.sum()), 0)

    def test_winning_move_ignores_build_suffix(self):
        board = self.empty_board()
        board[0, 2, 2] = 1
        board[1, 2, 2] = 2
        board[1, 2, 3] = 3

        valids = self.game.getValidMoves(board, 1)
        east_move_actions = valids[4 * 8:5 * 8]
        self.assertEqual(int(east_move_actions.sum()), 8)

        next_board, next_player = self.game.getNextState(board, 1, 4 * 8)
        self.assertEqual(next_board[0, 2, 3], 1)
        self.assertEqual(next_board[0, 2, 2], 0)
        self.assertEqual(int(next_board[1].sum()), int(board[1].sum()))
        self.assertEqual(self.game.getGameEnded(next_board, next_player), -1)

    def test_can_build_on_vacated_square(self):
        board = self.empty_board()
        board[0, 2, 2] = 1

        # Worker 1 moves east, then builds west back on the square it left.
        action = 4 * 8 + 3
        self.assertEqual(self.game.getValidMoves(board, 1)[action], 1)

        next_board, _ = self.game.getNextState(board, 1, action)
        self.assertEqual(next_board[0, 2, 3], 1)
        self.assertEqual(next_board[0, 2, 2], 0)
        self.assertEqual(next_board[1, 2, 2], 1)

    def test_building_on_level_three_creates_dome_and_blocks_future_movement(self):
        board = self.empty_board()
        board[0, 2, 2] = 1
        board[1, 1, 1] = 3

        # Move north, build west onto a level-three tower.
        action = 1 * 8 + 3
        self.assertEqual(self.game.getValidMoves(board, 1)[action], 1)

        next_board, _ = self.game.getNextState(board, 1, action)
        self.assertEqual(next_board[1, 1, 1], 4)

        # From (1, 2), moving west onto the dome at (1, 1) is illegal.
        self.assertEqual(int(self.game.getValidMoves(next_board, 1)[3 * 8:4 * 8].sum()), 0)

    def test_player_with_no_legal_moves_loses(self):
        board = np.zeros((2, 5, 5), dtype=int)
        board[0, 1, 1] = 1
        board[0, 3, 3] = 2
        board[0, 0, 4] = -1
        board[0, 4, 0] = -2

        for x in range(5):
            for y in range(5):
                if (x, y) not in [(1, 1), (3, 3), (0, 4), (4, 0)]:
                    board[1, x, y] = 4

        self.assertEqual(self.game.getGameEnded(board, 1), -1)

    def test_symmetries_preserve_legal_action_mask(self):
        board = self.game.getInitBoard()

        for _ in range(6):
            valids = self.game.getValidMoves(board, 1)
            action = int(np.flatnonzero(valids)[0])
            board, next_player = self.game.getNextState(board, 1, action)
            board = self.game.getCanonicalForm(board, next_player)

        valids = self.game.getValidMoves(board, 1)
        for sym_board, sym_pi in self.game.getSymmetries(board, valids):
            self.assertTrue(np.array_equal(np.array(sym_pi), self.game.getValidMoves(sym_board, 1)))


if __name__ == "__main__":
    unittest.main()
