import unittest

import numpy as np

from MCTS import MCTS
from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.NNet import NNetWrapper, args as nnet_args
from utils import dotdict


class TestSantoriniNNet(unittest.TestCase):
    def setUp(self):
        np.random.seed(11)
        self.game = SantoriniGame(5)
        self.nnet = NNetWrapper(self.game)

    def test_encode_board_uses_worker_identity_and_height_planes(self):
        board = np.zeros((2, 5, 5), dtype=int)
        board[0, 1, 1] = 1
        board[0, 2, 2] = 2
        board[0, 3, 3] = -1
        board[0, 4, 4] = -2
        board[1, 0, 0] = 1
        board[1, 0, 1] = 2
        board[1, 0, 2] = 3
        board[1, 0, 3] = 4

        encoded = NNetWrapper.encode_board(board)

        self.assertEqual(encoded.shape, (8, 5, 5))
        self.assertEqual(encoded[0, 1, 1], 1)
        self.assertEqual(encoded[1, 2, 2], 1)
        self.assertEqual(encoded[2, 3, 3], 1)
        self.assertEqual(encoded[3, 4, 4], 1)
        self.assertEqual(encoded[4, 0, 0], 1)
        self.assertEqual(encoded[5, 0, 1], 1)
        self.assertEqual(encoded[6, 0, 2], 1)
        self.assertEqual(encoded[7, 0, 3], 1)

    def test_predict_returns_policy_and_scalar_value(self):
        board = self.game.getCanonicalForm(self.game.getInitBoard(), 1)

        pi, v = self.nnet.predict(board)

        self.assertEqual(pi.shape, (128,))
        self.assertAlmostEqual(float(pi.sum()), 1.0, places=5)
        self.assertIsInstance(v, float)
        self.assertGreaterEqual(v, -1.0)
        self.assertLessEqual(v, 1.0)

    def test_single_training_step_runs(self):
        old_epochs = nnet_args.epochs
        old_batch_size = nnet_args.batch_size
        nnet_args.epochs = 1
        nnet_args.batch_size = 2
        try:
            examples = []
            for _ in range(2):
                board = self.game.getCanonicalForm(self.game.getInitBoard(), 1)
                valids = self.game.getValidMoves(board, 1).astype(np.float32)
                pi = valids / valids.sum()
                examples.append((board, pi, 1))

            self.nnet.train(examples)
        finally:
            nnet_args.epochs = old_epochs
            nnet_args.batch_size = old_batch_size

    def test_mcts_can_use_santorini_network(self):
        mcts_args = dotdict({'numMCTSSims': 2, 'cpuct': 1.0})
        mcts = MCTS(self.game, self.nnet, mcts_args)
        board = self.game.getCanonicalForm(self.game.getInitBoard(), 1)

        probs = mcts.getActionProb(board, temp=1)

        self.assertEqual(len(probs), 128)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
