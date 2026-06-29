import argparse
import json
import os

import numpy as np

import Arena
from MCTS import MCTS
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniPlayers import GreedySantoriniPlayer, RandomPlayer
from santorini.pytorch.NNet import NNetWrapper
from utils import dotdict


def build_baseline(game, name):
    if name == 'random':
        return RandomPlayer(game).play
    if name == 'greedy':
        return GreedySantoriniPlayer(game).play
    raise ValueError("Unknown baseline: {}".format(name))


def main():
    parser = argparse.ArgumentParser(description='Pit a Santorini neural MCTS player against a baseline.')
    parser.add_argument('--baseline', choices=['random', 'greedy'], default='random')
    parser.add_argument('--games', type=int, default=4)
    parser.add_argument('--sims', type=int, default=25)
    parser.add_argument('--checkpoint-folder', default='./temp/santorini_quick/')
    parser.add_argument('--checkpoint-file', default='best.pth.tar')
    parser.add_argument('--fresh', action='store_true', help='Use an untrained network even if a checkpoint exists.')
    parser.add_argument('--json-out', help='Optional path to write evaluation results as JSON.')
    args = parser.parse_args()

    game = SantoriniGame(5, true_random_placement=True)
    nnet = NNetWrapper(game)

    checkpoint_path = os.path.join(args.checkpoint_folder, args.checkpoint_file)
    if not args.fresh and os.path.exists(checkpoint_path):
        nnet.load_checkpoint(args.checkpoint_folder, args.checkpoint_file)
        print("Loaded checkpoint: {}".format(checkpoint_path))
    else:
        print("Using fresh untrained network.")

    mcts_args = dotdict({'numMCTSSims': args.sims, 'cpuct': 1.0})
    mcts = MCTS(game, nnet, mcts_args)
    nnet_player = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
    baseline_player = build_baseline(game, args.baseline)

    arena = Arena.Arena(nnet_player, baseline_player, game, display=SantoriniGame.display)
    nnet_wins, baseline_wins, draws = arena.playGames(args.games, verbose=False)

    print("Neural MCTS wins: {}".format(nnet_wins))
    print("{} wins: {}".format(args.baseline.title(), baseline_wins))
    print("Draws: {}".format(draws))

    if args.json_out:
        result = {
            'baseline': args.baseline,
            'games': args.games,
            'sims': args.sims,
            'checkpoint_folder': args.checkpoint_folder,
            'checkpoint_file': args.checkpoint_file,
            'fresh': args.fresh,
            'neural_mcts_wins': int(nnet_wins),
            'baseline_wins': int(baseline_wins),
            'draws': int(draws),
        }
        json_dir = os.path.dirname(args.json_out)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        with open(args.json_out, 'w') as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print("Wrote JSON results: {}".format(args.json_out))


if __name__ == "__main__":
    main()
