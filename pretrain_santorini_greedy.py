import argparse
import os

import numpy as np
import torch

from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.NNet import NNetWrapper, args as nnet_args


def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain Santorini policy/value heads from one-ply greedy labels.')
    parser.add_argument('--examples', type=int, default=5000)
    parser.add_argument('--max-random-plies', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--tie-policy', choices=['first', 'uniform'], default='first')
    parser.add_argument('--tactical-ratio', type=float, default=0.0)
    parser.add_argument('--tactical-max-attempts', type=int, default=200)
    parser.add_argument('--checkpoint-folder', default='./temp/santorini_greedy_pretrain/')
    parser.add_argument('--checkpoint-file', default='best.pth.tar')
    parser.add_argument('--load-folder')
    parser.add_argument('--load-file', default='best.pth.tar')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def decoded_action(game, board, player, action_id):
    del board, player
    worker, move_direction, build_direction = game.decodeAction(action_id)
    move = (
        worker[0] + DIRECTIONS[move_direction][0],
        worker[1] + DIRECTIONS[move_direction][1],
    )
    build = (
        move[0] + DIRECTIONS[build_direction][0],
        move[1] + DIRECTIONS[build_direction][1],
    )
    return worker, move, build


def greedy_targets(game, board, tie_policy='uniform'):
    valids = game.getValidMoves(board, 1)
    legal_actions = np.flatnonzero(valids)
    if len(legal_actions) == 0:
        return None

    scores = []
    for action in legal_actions:
        next_board, _ = game.getNextState(board, 1, int(action))
        scores.append(game.getScore(next_board, 1))

    scores = np.array(scores, dtype=np.float32)
    best_score = float(np.max(scores))
    best_actions = legal_actions[scores == best_score]

    policy = np.zeros(game.getActionSize(), dtype=np.float32)
    if tie_policy == 'first':
        policy[int(best_actions[0])] = 1.0
    else:
        policy[best_actions] = 1.0 / len(best_actions)

    if abs(best_score) >= 100:
        value = np.sign(best_score)
    else:
        value = np.clip(best_score / 3.0, -1.0, 1.0)

    return policy, float(value), best_score, len(best_actions)


def opponent_immediate_win_squares(game, board):
    valids = game.getValidMoves(board, -1)
    legal_actions = np.flatnonzero(valids)
    win_squares = set()

    for action in legal_actions:
        _, move, _ = decoded_action(game, board, -1, int(action))
        if board[1][move] != 3:
            continue

        next_board, _ = game.getNextState(board, -1, int(action))
        if game.getGameEnded(next_board, 1) == -1:
            win_squares.add(move)

    return win_squares


def blocking_targets(game, board):
    threat_squares = opponent_immediate_win_squares(game, board)
    if not threat_squares:
        return None

    valids = game.getValidMoves(board, 1)
    legal_actions = np.flatnonzero(valids)
    blocking_actions = []

    for action in legal_actions:
        _, _, build = decoded_action(game, board, 1, int(action))
        if build not in threat_squares:
            continue

        next_board, _ = game.getNextState(board, 1, int(action))
        if next_board[1][build] == 4:
            blocking_actions.append(int(action))

    if not blocking_actions:
        return None

    policy = np.zeros(game.getActionSize(), dtype=np.float32)
    policy[blocking_actions] = 1.0 / len(blocking_actions)
    return policy, 0.5, len(threat_squares), len(blocking_actions)


def random_reachable_canonical_board(game, max_random_plies):
    board = game.getInitBoard()
    player = 1
    plies = np.random.randint(0, max_random_plies + 1)

    for _ in range(plies):
        if game.getGameEnded(board, player) != 0:
            break
        valids = game.getValidMoves(board, player)
        legal_actions = np.flatnonzero(valids)
        if len(legal_actions) == 0:
            break
        action = int(np.random.choice(legal_actions))
        board, player = game.getNextState(board, player, action)

    return game.getCanonicalForm(board, player)


def immediate_win_template():
    board = np.zeros((2, 5, 5), dtype=int)
    board[0, 2, 2] = 1
    board[0, 0, 0] = 2
    board[0, 4, 4] = -1
    board[0, 4, 3] = -2
    board[1, 2, 2] = 2
    board[1, 2, 3] = 3
    return board


def blocking_template():
    board = np.zeros((2, 5, 5), dtype=int)
    board[0, 2, 2] = 1
    board[0, 0, 0] = 2
    board[0, 1, 2] = -1
    board[0, 4, 3] = -2
    board[1, 1, 2] = 2
    board[1, 1, 3] = 3
    return board


def random_board_symmetry(game, board):
    rotations = np.random.randint(0, 4)
    flip = bool(np.random.randint(0, 2))

    new_board = np.array([
        np.rot90(board[0], rotations),
        np.rot90(board[1], rotations),
    ])
    if flip:
        new_board = np.array([
            np.fliplr(new_board[0]),
            np.fliplr(new_board[1]),
        ])
    return new_board


def tactical_example(game, max_random_plies, tie_policy, max_attempts):
    del max_random_plies, max_attempts

    if np.random.randint(0, 2) == 0:
        board = random_board_symmetry(game, immediate_win_template())
        targets = greedy_targets(game, board, tie_policy=tie_policy)
        if targets is not None and targets[2] >= 100:
            policy, value, best_score, tie_count = targets
            return (board, policy, value), 'win', best_score, tie_count

    board = random_board_symmetry(game, blocking_template())
    targets = blocking_targets(game, board)
    if targets is not None:
        policy, value, threat_count, block_count = targets
        return (board, policy, value), 'block', threat_count, block_count

    board = random_board_symmetry(game, immediate_win_template())
    targets = greedy_targets(game, board, tie_policy=tie_policy)
    if targets is not None and targets[2] >= 100:
        policy, value, best_score, tie_count = targets
        return (board, policy, value), 'win', best_score, tie_count

    return None


def build_examples(game, count, max_random_plies, tie_policy, tactical_ratio, tactical_max_attempts):
    examples = []
    best_scores = []
    tie_counts = []
    tactical_counts = {'win': 0, 'block': 0, 'fallback': 0}

    while len(examples) < count:
        if np.random.random() < tactical_ratio:
            tactical = tactical_example(game, max_random_plies, tie_policy, tactical_max_attempts)
            if tactical is not None:
                example, tactical_type, best_score, tie_count = tactical
                examples.append(example)
                best_scores.append(best_score)
                tie_counts.append(tie_count)
                tactical_counts[tactical_type] += 1
                continue
            tactical_counts['fallback'] += 1

        board = random_reachable_canonical_board(game, max_random_plies)
        if game.getGameEnded(board, 1) != 0:
            continue

        targets = greedy_targets(game, board, tie_policy=tie_policy)
        if targets is None:
            continue

        policy, value, best_score, tie_count = targets
        examples.append((board, policy, value))
        best_scores.append(best_score)
        tie_counts.append(tie_count)

    return examples, np.array(best_scores), np.array(tie_counts), tactical_counts


def policy_accuracy(nnet, game, examples, limit=512):
    if len(examples) == 0:
        return 0.0

    correct = 0
    checked = 0
    for board, target_policy, _ in examples[:limit]:
        policy, _ = nnet.predict(board)
        valids = game.getValidMoves(board, 1)
        policy = policy * valids
        if np.sum(policy) <= 0:
            policy = valids
        action = int(np.argmax(policy))
        if target_policy[action] > 0:
            correct += 1
        checked += 1

    return correct / checked


def main():
    parsed_args = parse_args()
    np.random.seed(parsed_args.seed)
    torch.manual_seed(parsed_args.seed)

    nnet_args.epochs = parsed_args.epochs
    nnet_args.batch_size = parsed_args.batch_size

    game = SantoriniGame(5, true_random_placement=True)
    nnet = NNetWrapper(game)

    if parsed_args.load_folder:
        nnet.load_checkpoint(parsed_args.load_folder, parsed_args.load_file)
        print('Loaded checkpoint: {}/{}'.format(parsed_args.load_folder, parsed_args.load_file))

    examples, best_scores, tie_counts, tactical_counts = build_examples(
        game,
        parsed_args.examples,
        parsed_args.max_random_plies,
        parsed_args.tie_policy,
        parsed_args.tactical_ratio,
        parsed_args.tactical_max_attempts,
    )

    print('Built {} greedy-labeled examples.'.format(len(examples)))
    print('Best score mean: {:.3f}; min: {:.1f}; max: {:.1f}'.format(
        float(np.mean(best_scores)),
        float(np.min(best_scores)),
        float(np.max(best_scores)),
    ))
    print('Mean greedy tie count: {:.2f}'.format(float(np.mean(tie_counts))))
    print('Tie policy: {}'.format(parsed_args.tie_policy))
    print('Tactical ratio: {:.2f}; examples: win={}, block={}, fallback={}'.format(
        parsed_args.tactical_ratio,
        tactical_counts['win'],
        tactical_counts['block'],
        tactical_counts['fallback'],
    ))
    print('Policy top-1 support accuracy before: {:.3f}'.format(policy_accuracy(nnet, game, examples)))

    nnet.train(examples)

    print('Policy top-1 support accuracy after: {:.3f}'.format(policy_accuracy(nnet, game, examples)))
    nnet.save_checkpoint(parsed_args.checkpoint_folder, parsed_args.checkpoint_file)
    print('Saved checkpoint: {}'.format(os.path.join(parsed_args.checkpoint_folder, parsed_args.checkpoint_file)))


if __name__ == '__main__':
    main()
