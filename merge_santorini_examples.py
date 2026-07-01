import argparse
import hashlib
import os
import pickle
from collections import deque


def window_fingerprint(window):
    digest = hashlib.sha1()
    digest.update(str(len(window)).encode("ascii"))
    for board, policy, value in window:
        digest.update(board.tobytes())
        digest.update(policy.tobytes())
        digest.update(str(value).encode("ascii"))
    return digest.hexdigest()


def load_history(path):
    with open(path, "rb") as examples_file:
        return pickle.Unpickler(examples_file).load()


def save_history(path, history):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "wb+") as examples_file:
        pickle.Pickler(examples_file).dump(history)


def merge_histories(paths, max_windows, dedupe=True):
    merged = []
    seen = set()
    skipped_duplicates = []

    for path in paths:
        history = load_history(path)
        for index, window in enumerate(history):
            fingerprint = window_fingerprint(window)
            if dedupe and fingerprint in seen:
                skipped_duplicates.append((path, index, fingerprint[:10]))
                continue

            seen.add(fingerprint)
            merged.append(deque(window, maxlen=window.maxlen))

    if max_windows is not None and len(merged) > max_windows:
        merged = merged[-max_windows:]

    return merged, skipped_duplicates


def parse_args():
    parser = argparse.ArgumentParser(description="Merge Santorini replay example histories.")
    parser.add_argument("inputs", nargs="+", help="Input .examples files in oldest-to-newest order.")
    parser.add_argument("--output", required=True, help="Output merged .examples file.")
    parser.add_argument("--max-windows", type=int, default=20)
    parser.add_argument("--allow-duplicates", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    merged, skipped_duplicates = merge_histories(
        args.inputs,
        max_windows=args.max_windows,
        dedupe=not args.allow_duplicates,
    )
    save_history(args.output, merged)

    print("Wrote merged examples: {}".format(args.output))
    print("History windows: {}".format(len(merged)))
    print("History lengths: {}".format([len(window) for window in merged]))
    print("Skipped duplicate windows: {}".format(len(skipped_duplicates)))
    for path, index, fingerprint in skipped_duplicates:
        print("  duplicate {} window {} ({})".format(path, index, fingerprint))


if __name__ == "__main__":
    main()
