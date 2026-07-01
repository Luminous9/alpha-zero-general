import unittest

import numpy as np

from evaluate_santorini import make_board, parse_grid, validate_board


class TestEvaluateSantoriniCliHelpers(unittest.TestCase):
    def test_parse_grid_accepts_semicolon_rows_and_space_values(self):
        grid = parse_grid(
            "0 0 0 0 0; 0 0 1 0 0; 0 -1 0 -2 0; 0 0 2 0 0; 0 0 0 0 0",
            "pieces",
            5,
        )

        self.assertEqual(grid.shape, (5, 5))
        self.assertEqual(grid[1, 2], 1)
        self.assertEqual(grid[2, 3], -2)

    def test_parse_grid_accepts_slash_rows_and_comma_values(self):
        grid = parse_grid(
            "0,0,0,0,0 / 0,0,1,0,0 / 0,-1,0,-2,0 / 0,0,2,0,0 / 0,0,0,0,0",
            "pieces",
            5,
        )

        self.assertEqual(grid.shape, (5, 5))
        self.assertEqual(grid[3, 2], 2)

    def test_validate_board_requires_each_worker_once(self):
        pieces = np.zeros((5, 5), dtype=int)
        pieces[0, 0] = 1
        pieces[1, 1] = -1
        pieces[2, 2] = -2
        heights = np.zeros((5, 5), dtype=int)
        board = make_board(pieces, heights, 5)

        with self.assertRaisesRegex(ValueError, "worker 2 exactly once"):
            validate_board(board)

    def test_validate_board_rejects_worker_on_dome(self):
        pieces = np.zeros((5, 5), dtype=int)
        pieces[0, 0] = 1
        pieces[1, 1] = 2
        pieces[2, 2] = -1
        pieces[3, 3] = -2
        heights = np.zeros((5, 5), dtype=int)
        heights[0, 0] = 4
        board = make_board(pieces, heights, 5)

        with self.assertRaisesRegex(ValueError, "domes"):
            validate_board(board)


if __name__ == "__main__":
    unittest.main()
