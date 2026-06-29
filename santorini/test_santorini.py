import unittest

import numpy as np

from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniLogic import Board
from santorini.SantoriniPlayers import HumanSantoriniPlayer, coordinate_label, parse_coordinate


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
        random_game = SantoriniGame(5, true_random_placement=True)
        seen_starts = set()
        for _ in range(200):
            board = random_game.getInitBoard()
            seen_starts.add(board[0].tobytes())
            player_locations = random_game.getCharacterLocations(board, 1)
            opponent_locations = random_game.getCharacterLocations(board, -1)

            self.assertEqual(len({*player_locations, *opponent_locations}), 4)
            self.assertFalse(all(self.is_outer_edge(loc) for loc in player_locations))
            self.assertFalse(all(self.is_outer_edge(loc) for loc in opponent_locations))
        self.assertGreater(len(seen_starts), 1)

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

    def test_fast_valid_moves_match_legacy_board_logic_on_reachable_positions(self):
        board = self.game.getInitBoard()

        for _ in range(20):
            legacy_board = Board(5)
            legacy_board.pieces = np.copy(board)
            expected = np.array(legacy_board.get_legal_moves_binary(1))

            actual = self.game.getValidMoves(board, 1)

            np.testing.assert_array_equal(actual, expected)
            if self.game.getGameEnded(board, 1) != 0:
                break

            legal_actions = np.flatnonzero(actual)
            action = int(legal_actions[np.random.randint(len(legal_actions))])
            board, next_player = self.game.getNextState(board, 1, action)
            board = self.game.getCanonicalForm(board, next_player)

    def test_combined_game_ended_returns_valids_for_non_terminal_state(self):
        board = self.game.getInitBoard()

        ended, valids = self.game.getGameEndedAndValidMoves(board, 1)

        self.assertEqual(ended, 0)
        np.testing.assert_array_equal(valids, self.game.getValidMoves(board, 1))

    def test_human_coordinate_parser_uses_letter_columns_and_one_based_rows(self):
        self.assertEqual(parse_coordinate('B3', 5), (2, 1))
        self.assertEqual(parse_coordinate('b3', 5), (2, 1))
        self.assertEqual(coordinate_label((2, 1)), 'B3')
        with self.assertRaises(ValueError):
            parse_coordinate('F1', 5)

    def test_human_text_action_maps_to_engine_action(self):
        board = self.game.getInitBoard()
        player = HumanSantoriniPlayer(self.game)

        action = player._parse_action(board, 'O B1 A1')

        self.assertEqual(action, 3)
        self.assertEqual(self.game.getValidMoves(board, 1)[action], 1)


if __name__ == "__main__":
    unittest.main()
