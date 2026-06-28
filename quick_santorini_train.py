import logging

import coloredlogs
import numpy as np
import torch

from Coach import Coach
from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.NNet import NNetWrapper, args as nnet_args
from utils import dotdict

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')


quick_args = dotdict({
    'numIters': 2,
    'numEps': 2,
    'tempThreshold': 5,
    'updateThreshold': 0.5,
    'maxlenOfQueue': 1000,
    'numMCTSSims': 4,
    'arenaCompare': 2,
    'cpuct': 1.0,

    'checkpoint': './temp/santorini_quick/',
    'load_model': False,
    'load_folder_file': ('./temp/santorini_quick', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 2,
})


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    nnet_args.epochs = 1
    nnet_args.batch_size = 8

    log.info('Starting tiny Santorini training shakedown')
    game = SantoriniGame(5, true_random_placement=True)
    nnet = NNetWrapper(game)
    coach = Coach(game, nnet, quick_args)
    coach.learn()
    log.info('Santorini training shakedown completed')


if __name__ == "__main__":
    main()
