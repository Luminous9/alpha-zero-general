# Santorini Colab Launch Recipe

This is the recommended notebook shape for longer Santorini training runs. Use a GPU runtime, keep checkpoints on Google Drive, and resume with replay examples loaded.

## 1. Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2. Get the repo

```bash
%cd /content
!git clone https://github.com/Luminous9/alpha-zero-general.git
%cd /content/alpha-zero-general
```

If you are working from a different fork, replace the clone URL with your fork.

## 3. Install the Santorini training dependencies

Colab already includes PyTorch, so start with the lightweight dependencies.

```bash
!pip install coloredlogs tqdm
```

Only use the full `requirements.txt` install if you need the older TensorFlow examples too; it pins old package versions that are not needed for Santorini.

## 4. Choose a Drive checkpoint folder

```python
CHECKPOINT = "/content/drive/MyDrive/santorini_az/checkpoints"
```

```bash
%env CHECKPOINT=/content/drive/MyDrive/santorini_az/checkpoints
```

## 5. First long run

If you do not already have a checkpoint in Drive, start without `--load-model` or copy a local bootstrap checkpoint into `CHECKPOINT` first.

```bash
!python main_santorini.py \
  --preset local \
  --checkpoint "$CHECKPOINT" \
  --num-iters 100 \
  --num-eps 80 \
  --num-mcts-sims 64 \
  --self-play-batch-size 32 \
  --arena-compare 80 \
  --update-threshold 0.50 \
  --epochs 3 \
  --history-iters 5 \
  --seed 8 \
  --quiet
```

## 6. Resume a stopped run

This loads `best.pth.tar` by default. For that normal resume path, Santorini asks for `latest.examples` first, then falls back to model-adjacent examples, `best.pth.tar.examples`, and the newest checkpoint examples. An explicit `--examples-file` always wins.

```bash
!python main_santorini.py \
  --preset local \
  --checkpoint "$CHECKPOINT" \
  --load-folder "$CHECKPOINT" \
  --load-model \
  --load-examples \
  --num-iters 100 \
  --num-eps 80 \
  --num-mcts-sims 64 \
  --self-play-batch-size 32 \
  --arena-compare 80 \
  --update-threshold 0.50 \
  --epochs 3 \
  --history-iters 5 \
  --seed 9 \
  --quiet
```

Use `--self-play-batch-size` to run multiple active self-play games while batching MCTS leaf inference through the GPU. On Colab, start around `32`; if GPU memory is comfortable and CPU move generation is keeping up, try `64`.

Use `--examples-file filename.examples` when you want to force a specific replay file; relative paths are checked inside the load folder and from the launch directory. Use `--skip-first-self-play` only if the loaded examples were generated from the exact loaded model and you intentionally want to train immediately before collecting fresh games.

## 7. Evaluate against greedy

```bash
!python pit_santorini.py \
  --baseline greedy \
  --checkpoint-folder "$CHECKPOINT" \
  --games 100 \
  --sims 64 \
  --json-out "$CHECKPOINT/eval_greedy_100_s64.json"
```

For promotion confidence, also run a second seed and at least one deeper search evaluation:

```bash
!python pit_santorini.py --baseline greedy --checkpoint-folder "$CHECKPOINT" --games 100 --sims 128
```
