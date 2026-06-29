import logging
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append('../..')
from NeuralNet import NeuralNet
from utils import AverageMeter, dotdict

import torch
import torch.optim as optim

from .SantoriniNNet import SantoriniNNet

log = logging.getLogger(__name__)


args = dotdict({
    'lr': 0.001,
    'dropout': 0.2,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'input_channels': 8,
    'num_channels': 64,
    'num_residual_blocks': 5,
    'value_hidden_size': 128,
    'quiet': False,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = SantoriniNNet(game, args)
        _, self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    @staticmethod
    def encode_board(board):
        pieces = board[0]
        heights = board[1]
        encoded = np.zeros((args.input_channels, pieces.shape[0], pieces.shape[1]), dtype=np.float32)

        encoded[0] = pieces == 1
        encoded[1] = pieces == 2
        encoded[2] = pieces == -1
        encoded[3] = pieces == -2
        encoded[4] = heights == 1
        encoded[5] = heights == 2
        encoded[6] = heights == 3
        encoded[7] = heights >= 4

        return encoded

    @classmethod
    def encode_boards(cls, boards):
        return np.array([cls.encode_board(board) for board in boards], dtype=np.float32)

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)
        encoded_boards, target_pis, target_vs = self._encode_training_examples(examples)

        for epoch in range(args.epochs):
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = max(1, int(np.ceil(len(examples) / args.batch_size)))

            if args.quiet:
                log.info('Training epoch %s/%s (%s batches)', epoch + 1, args.epochs, batch_count)
            else:
                print('EPOCH ::: ' + str(epoch + 1))

            t = tqdm(range(batch_count), desc='Training Net', disable=args.quiet)
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards = torch.from_numpy(encoded_boards[sample_ids])
                batch_target_pis = torch.from_numpy(target_pis[sample_ids])
                batch_target_vs = torch.from_numpy(target_vs[sample_ids])

                if args.cuda:
                    boards = boards.contiguous().cuda()
                    batch_target_pis = batch_target_pis.contiguous().cuda()
                    batch_target_vs = batch_target_vs.contiguous().cuda()

                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(batch_target_pis, out_pi)
                l_v = self.loss_v(batch_target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                if not args.quiet:
                    t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            if args.quiet:
                log.info(
                    'Finished epoch %s/%s: pi_loss=%.4f v_loss=%.4f',
                    epoch + 1,
                    args.epochs,
                    pi_losses.avg,
                    v_losses.avg,
                )

    def _encode_training_examples(self, examples):
        boards, pis, vs = list(zip(*examples))
        return (
            self.encode_boards(boards),
            np.array(pis, dtype=np.float32),
            np.array(vs, dtype=np.float32),
        )

    def predict(self, board):
        """
        board: canonical Santorini rules board with shape (2, n, n)
        """
        pis, vs = self.predict_batch([board])
        return pis[0], float(vs[0])

    def predict_batch(self, boards):
        """
        boards: iterable of canonical Santorini rules boards with shape (2, n, n)
        """
        encoded = torch.FloatTensor(self.encode_boards(boards))
        if args.cuda:
            encoded = encoded.contiguous().cuda()

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(encoded)

        return torch.exp(pi).data.cpu().numpy(), v.view(-1).data.cpu().numpy()

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            if not args.quiet:
                print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.makedirs(folder)
        else:
            if not args.quiet:
                print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
