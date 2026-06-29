import unittest

import numpy as np

from BatchedArena import BatchedMCTSArena
from utils import dotdict


class TinyGame:
    def getInitBoard(self):
        return np.array([0], dtype=np.int64)

    def getActionSize(self):
        return 2

    def getNextState(self, board, player, action):
        return np.array([board[0] + 1], dtype=np.int64), -player

    def getValidMoves(self, board, player):
        return np.array([1, 1], dtype=np.int64)

    def getGameEnded(self, board, player):
        if board[0] >= 2:
            return -1
        return 0

    def getCanonicalForm(self, board, player):
        return board

    def stringRepresentation(self, board):
        return board.tobytes()


class BatchCountingNNet:
    def __init__(self):
        self.batch_sizes = []

    def predict(self, board):
        return np.array([0.5, 0.5], dtype=np.float32), 0.0

    def predict_batch(self, boards):
        self.batch_sizes.append(len(boards))
        policies = np.tile(np.array([0.5, 0.5], dtype=np.float32), (len(boards), 1))
        values = np.zeros(len(boards), dtype=np.float32)
        return policies, values


class TestBatchedMCTSArena(unittest.TestCase):
    def test_batched_arena_scores_swapped_starts(self):
        game = TinyGame()
        player1_nnet = BatchCountingNNet()
        player2_nnet = BatchCountingNNet()
        args = dotdict({'numMCTSSims': 2, 'cpuct': 1.0})
        arena = BatchedMCTSArena(
            game,
            player1_nnet,
            player2_nnet,
            args,
            batch_size=2,
            quiet=True,
        )

        one_won, two_won, draws = arena.playGames(4)

        self.assertEqual((one_won, two_won, draws), (2, 2, 0))
        self.assertGreaterEqual(max(player1_nnet.batch_sizes), 2)
        self.assertGreaterEqual(max(player2_nnet.batch_sizes), 2)


if __name__ == "__main__":
    unittest.main()
