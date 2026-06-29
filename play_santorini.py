import argparse
import os

from pit_santorini import NeuralMCTSPlayer
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniPlayers import (
    HumanSantoriniPlayer,
    SANTORINI_DIRECTIONS,
    coordinate_label,
)


DEFAULT_CHECKPOINT_FOLDER = './temp/santorini_colab_training2'


def parse_args():
    parser = argparse.ArgumentParser(description='Play Santorini against a trained neural MCTS player.')
    parser.add_argument('--checkpoint-folder', default=DEFAULT_CHECKPOINT_FOLDER)
    parser.add_argument('--checkpoint-file', default='best.pth.tar')
    parser.add_argument('--sims', type=int, default=1024, help='MCTS simulations per AI move.')
    parser.add_argument('--human-first', action='store_true', help='Let the human play first.')
    parser.add_argument('--ai-first', action='store_true', help='Let the AI play first.')
    parser.add_argument(
        '--fixed-start',
        action='store_true',
        help='Use the deterministic default start instead of randomized worker placement.',
    )
    return parser.parse_args()


def describe_action(game, board, player, action, perspective_player):
    worker_idx = action // 64
    local_action = action % 64
    move_direction = local_action // 8
    build_direction = local_action % 8

    origin = game.getCharacterLocations(board, player)[worker_idx]
    move_delta = SANTORINI_DIRECTIONS[move_direction]
    build_delta = SANTORINI_DIRECTIONS[build_direction]
    move = (origin[0] + move_delta[0], origin[1] + move_delta[1])
    build = (move[0] + build_delta[0], move[1] + build_delta[1])

    if player == perspective_player:
        worker = ('O', 'U')[worker_idx]
    else:
        worker = ('X', 'Y')[worker_idx]
    return "{} {} {}".format(worker, coordinate_label(move), coordinate_label(build))


def play_game(game, human, ai, human_player):
    board = game.getInitBoard()
    cur_player = 1
    turn = 0

    ai.startGame()

    while game.getGameEnded(board, cur_player) == 0:
        turn += 1
        actor = 'Human' if cur_player == human_player else 'AI'
        print("\nTurn {}: {} to move".format(turn, actor))
        print("Board from your perspective: pieces first, then tower heights.")
        SantoriniGame.display(game.getCanonicalForm(board, human_player))

        canonical_board = game.getCanonicalForm(board, cur_player)
        if cur_player == human_player:
            action = human(canonical_board)
        else:
            print("AI thinking...")
            action = ai(canonical_board)
            print("AI plays {}.".format(describe_action(game, board, cur_player, action, human_player)))

        valids = game.getValidMoves(canonical_board, 1)
        if valids[action] == 0:
            raise AssertionError("Player returned illegal action {}".format(action))
        board, cur_player = game.getNextState(board, cur_player, action)

    result = cur_player * game.getGameEnded(board, cur_player)
    print("\nFinal board from your perspective:")
    SantoriniGame.display(game.getCanonicalForm(board, human_player))
    return result


def main():
    args = parse_args()
    if args.human_first and args.ai_first:
        raise ValueError('Choose at most one of --human-first or --ai-first.')

    checkpoint_path = os.path.join(args.checkpoint_folder, args.checkpoint_file)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("No model in path {}".format(checkpoint_path))

    game = SantoriniGame(5, true_random_placement=not args.fixed_start)
    human = HumanSantoriniPlayer(game).play
    ai = NeuralMCTSPlayer(game, args.checkpoint_folder, args.checkpoint_file, args.sims)

    human_starts = args.human_first or not args.ai_first
    human_player = 1 if human_starts else -1

    print("Loaded checkpoint: {}".format(checkpoint_path))
    print("AI MCTS sims per move: {}".format(args.sims))
    print("Human is Player {}.".format(human_player))
    print("Your workers are O and U. AI workers are X and Y.")
    print("Coordinates use lettered columns and numbered rows; the top-left corner is A1.")

    try:
        result = play_game(game, human, ai, human_player)
    except KeyboardInterrupt:
        print("\nGame aborted.")
        return

    if result == human_player:
        winner = 'Human'
    elif result == -human_player:
        winner = 'AI'
    else:
        winner = 'Nobody'
    print("Winner: {}".format(winner))


if __name__ == '__main__':
    main()
