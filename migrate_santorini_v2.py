import argparse
import os
import pickle
from collections import deque

import numpy as np
import torch

from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.NNet import NNetWrapper, args as nnet_args


DEFAULT_SOURCE_FOLDER = "temp/santorini_kaggle_training6"
DEFAULT_OUTPUT_FOLDER = "temp/santorini_kaggle_training6_v2"


def resolve_folder(path):
    if os.path.exists(path):
        return path
    if path.startswith("/temp/"):
        relative_path = path.lstrip("/")
        if os.path.exists(relative_path):
            return relative_path
    return path


def translate_policy_v1_to_v2(game, board, policy):
    policy = np.asarray(policy, dtype=np.float32)
    if policy.shape == (game.getActionSize(),):
        return policy.copy()
    if policy.shape != (128,):
        raise ValueError("Expected V1 policy shape (128,), got {}.".format(policy.shape))

    translated = np.zeros(game.getActionSize(), dtype=np.float32)
    for worker_idx, origin in enumerate(game.getCharacterLocations(board, 1)):
        old_offset = worker_idx * 64
        new_offset = game.getActionFromOrigin(origin, 0, 0)
        translated[new_offset:new_offset + 64] = policy[old_offset:old_offset + 64]
    return translated


def translate_examples_history(game, examples_history):
    translated_history = []
    translated_count = 0

    for history in examples_history:
        translated = deque([], maxlen=history.maxlen)
        for board, policy, value in history:
            translated.append((
                np.array(board, copy=True),
                translate_policy_v1_to_v2(game, board, policy),
                value,
            ))
            translated_count += 1
        translated_history.append(translated)

    return translated_history, translated_count


def translate_examples_file(input_path, output_path, game):
    with open(input_path, "rb") as examples_file:
        examples_history = pickle.Unpickler(examples_file).load()

    translated_history, translated_count = translate_examples_history(game, examples_history)

    output_folder = os.path.dirname(output_path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    with open(output_path, "wb+") as examples_file:
        pickle.Pickler(examples_file).dump(translated_history)

    return {
        "history_count": len(translated_history),
        "example_count": translated_count,
        "history_lengths": [len(history) for history in translated_history],
    }


def build_v2_state_dict_from_v1(v1_state_dict, v2_state_dict):
    migrated = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in v2_state_dict.items()
    }

    old_stem = v1_state_dict["stem.0.weight"]
    new_stem = migrated["stem.0.weight"].clone()
    new_stem[:, 0] = old_stem[:, 0:2].mean(dim=1)
    new_stem[:, 1] = old_stem[:, 2:4].mean(dim=1)
    new_stem[:, 2] = old_stem[:, 4]
    new_stem[:, 3] = old_stem[:, 5]
    new_stem[:, 4] = old_stem[:, 6]
    new_stem[:, 5] = old_stem[:, 7]
    migrated["stem.0.weight"] = new_stem

    if "stem.1.weight" in v1_state_dict:
        migrated["stem.1.weight"] = v1_state_dict["stem.1.weight"].clone()
    if "stem.1.bias" in v1_state_dict:
        migrated["stem.1.bias"] = v1_state_dict["stem.1.bias"].clone()
    if "stem.1.running_mean" in migrated:
        migrated["stem.1.running_mean"].zero_()
    if "stem.1.running_var" in migrated:
        migrated["stem.1.running_var"].fill_(1)
    if "stem.1.num_batches_tracked" in migrated:
        migrated["stem.1.num_batches_tracked"].zero_()

    skipped_prefixes = ("stem.0.", "stem.1.", "policy_")
    transferred = []
    skipped = []

    for key, old_value in v1_state_dict.items():
        if key.startswith(skipped_prefixes):
            skipped.append(key)
            continue
        if key not in migrated:
            skipped.append(key)
            continue
        if migrated[key].shape != old_value.shape:
            skipped.append(key)
            continue
        migrated[key] = old_value.clone()
        transferred.append(key)

    return migrated, transferred, skipped


def migrate_checkpoint(input_path, output_path, game):
    map_location = "cpu"
    checkpoint = torch.load(input_path, map_location=map_location)
    v1_state_dict = checkpoint["state_dict"]

    old_cuda = nnet_args.cuda
    nnet_args.cuda = False
    try:
        v2_nnet = NNetWrapper(game)
        v2_state_dict = v2_nnet.nnet.state_dict()
    finally:
        nnet_args.cuda = old_cuda

    migrated, transferred, skipped = build_v2_state_dict_from_v1(v1_state_dict, v2_state_dict)

    output_folder = os.path.dirname(output_path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    torch.save({
        "state_dict": migrated,
        "migration": {
            "source_checkpoint": input_path,
            "transferred_keys": transferred,
            "skipped_keys": skipped,
            "notes": (
                "V1 Santorini checkpoint migrated to V2 anonymous workers and "
                "1600 spatial policy actions. Policy head is newly initialized."
            ),
        },
    }, output_path)

    return {
        "transferred_count": len(transferred),
        "skipped_count": len(skipped),
        "skipped_policy_keys": [key for key in skipped if key.startswith("policy_")],
    }


def flatten_examples_history(examples_history):
    examples = []
    for history in examples_history:
        examples.extend(history)
    return examples


def bootstrap_checkpoint(checkpoint_path, examples_path, output_path, game, epochs, batch_size, use_cuda):
    old_epochs = nnet_args.epochs
    old_batch_size = nnet_args.batch_size
    old_cuda = nnet_args.cuda
    nnet_args.epochs = epochs
    nnet_args.batch_size = batch_size
    nnet_args.cuda = bool(use_cuda and torch.cuda.is_available())
    try:
        nnet = NNetWrapper(game)
        nnet.load_checkpoint(os.path.dirname(checkpoint_path), os.path.basename(checkpoint_path))
        with open(examples_path, "rb") as examples_file:
            examples_history = pickle.Unpickler(examples_file).load()
        examples = flatten_examples_history(examples_history)
        nnet.train(examples)
        nnet.save_checkpoint(os.path.dirname(output_path), os.path.basename(output_path))
    finally:
        nnet_args.epochs = old_epochs
        nnet_args.batch_size = old_batch_size
        nnet_args.cuda = old_cuda

    return {
        "example_count": len(examples),
        "epochs": epochs,
        "batch_size": batch_size,
        "cuda": nnet_args.cuda,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Migrate Santorini V1 checkpoints/examples to the V2 anonymous-worker architecture."
    )
    parser.add_argument("--source-folder", default=DEFAULT_SOURCE_FOLDER)
    parser.add_argument("--output-folder", default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--checkpoint-file", default="best.pth.tar")
    parser.add_argument("--examples-file", default="latest.examples")
    parser.add_argument("--output-checkpoint-file", default="best.pth.tar")
    parser.add_argument("--output-examples-file", default="latest.examples")
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--skip-examples", action="store_true")
    parser.add_argument("--bootstrap-epochs", type=int, default=0)
    parser.add_argument("--bootstrap-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-cpu", action="store_true")
    parser.add_argument("--bootstrapped-checkpoint-file", default="bootstrapped.pth.tar")
    return parser.parse_args()


def main():
    args = parse_args()
    source_folder = resolve_folder(args.source_folder)
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    game = SantoriniGame(5)

    input_checkpoint = os.path.join(source_folder, args.checkpoint_file)
    output_checkpoint = os.path.join(output_folder, args.output_checkpoint_file)
    input_examples = os.path.join(source_folder, args.examples_file)
    output_examples = os.path.join(output_folder, args.output_examples_file)

    if not args.skip_checkpoint:
        if not os.path.isfile(input_checkpoint):
            raise FileNotFoundError("No checkpoint found at {}".format(input_checkpoint))
        result = migrate_checkpoint(input_checkpoint, output_checkpoint, game)
        print("Wrote V2 checkpoint: {}".format(output_checkpoint))
        print("Transferred {} tensors; skipped {} tensors.".format(
            result["transferred_count"],
            result["skipped_count"],
        ))
        if result["skipped_policy_keys"]:
            print("Reinitialized policy tensors: {}".format(", ".join(result["skipped_policy_keys"])))

    if not args.skip_examples:
        if not os.path.isfile(input_examples):
            raise FileNotFoundError("No examples found at {}".format(input_examples))
        result = translate_examples_file(input_examples, output_examples, game)
        print("Wrote V2 examples: {}".format(output_examples))
        print("Translated {} examples across {} history windows.".format(
            result["example_count"],
            result["history_count"],
        ))
        print("History lengths: {}".format(result["history_lengths"]))

    if args.bootstrap_epochs > 0:
        bootstrapped_checkpoint = os.path.join(output_folder, args.bootstrapped_checkpoint_file)
        result = bootstrap_checkpoint(
            output_checkpoint,
            output_examples,
            bootstrapped_checkpoint,
            game,
            args.bootstrap_epochs,
            args.bootstrap_batch_size,
            not args.bootstrap_cpu,
        )
        print("Wrote bootstrapped V2 checkpoint: {}".format(bootstrapped_checkpoint))
        print("Bootstrapped on {example_count} examples for {epochs} epochs.".format(**result))
        print("CUDA used for bootstrap: {}".format(result["cuda"]))


if __name__ == "__main__":
    main()
