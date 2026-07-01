import unittest

import numpy as np

from migrate_santorini_v2 import translate_policy_v1_to_v2
from santorini.SantoriniGame import SantoriniGame


class TestSantoriniV2Migration(unittest.TestCase):
    def setUp(self):
        self.game = SantoriniGame(5)

    def test_translate_policy_maps_worker_slots_to_physical_origins(self):
        board = np.zeros((2, 5, 5), dtype=int)
        board[0, 1, 2] = 1
        board[0, 3, 4] = 2
        board[0, 0, 0] = -1
        board[0, 4, 0] = -2

        policy = np.zeros(128, dtype=np.float32)
        policy[4] = 0.25
        policy[64 + 12] = 0.75

        translated = translate_policy_v1_to_v2(self.game, board, policy)

        self.assertEqual(translated.shape, (self.game.getActionSize(),))
        self.assertEqual(
            translated[self.game.getActionFromOrigin((1, 2), 0, 4)],
            0.25,
        )
        self.assertEqual(
            translated[self.game.getActionFromOrigin((3, 4), 1, 4)],
            0.75,
        )
        self.assertAlmostEqual(float(translated.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
