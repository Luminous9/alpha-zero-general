Santorini is an incredible candidate for an AI project. Because it is a perfect-information, zero-sum game with no hidden elements and a small 5x5 board, it is perfectly suited for the **AlphaZero architecture** (a combination of Deep Neural Networks and Monte Carlo Tree Search).

Because the board is so small and games rarely exceed 30–40 plies (half-moves), you can train a strong Santorini AI on a standard laptop with a decent GPU, or by using free tiers of Google Colab or Kaggle. We should still treat "superhuman" as an empirical target rather than an assumption: rules correctness, self-play volume, and evaluation quality will decide how far the model actually gets.

Here is the architectural blueprint for building your Santorini AI.

---

## The Core Strategy: AlphaZero

Instead of writing complex, hard-coded heuristics for what makes a "good" Santorini position, the AI will learn entirely by playing against itself.

1. A **Neural Network** evaluates the board and suggests moves.
2. **Monte Carlo Tree Search (MCTS)** looks ahead a few steps to verify those suggestions.
3. The results of the MCTS make the Neural Network smarter, creating a feedback loop.

### Phase 1: The Fast Rules Engine (Crucial)

Before touching neural networks, you need a headless (no graphics) game engine that strictly enforces Santorini's rules.

- **Performance is king:** Your AI will need to play hundreds of thousands of self-play games. If your engine is slow, training will take months.
- **Language:** Start with a carefully tested Python/NumPy rules engine. If profiling later shows legal move generation is the bottleneck, move only that hot path to Numba/C++/Rust.
- **Scope:** Start with the **base game only** (no God powers). God powers completely change the state representation and action space, so leave them for version 2.0.
- **Starting positions:** Train from randomized legal post-placement positions for v1. Reject sampled starts where either player has both workers on outer-edge squares. This keeps early games varied without making the model first learn the worker-placement phase.

---

## Phase 2: Neural Network Architecture

Your neural network acts as the "intuition" of the AI. It takes the current board state as input and outputs two things: which moves look promising (Policy), and who is currently winning (Value).

### 1. State Representation (The Input Tensor)

Neural networks for board games "see" the board as a stack of 2D grids (a 3D tensor of shape $5 \times 5 \times C$, where $C$ is the number of channels). Do not feed the network raw integers (like a `3` for a 3-block tower); use binary channels (1s and 0s) so the network can easily recognize patterns.

| Channel | Data ($5 \times 5$ grids) | Description                                                              |
| ------- | ------------------------- | ------------------------------------------------------------------------ |
| **0**   | Current Worker 1          | `1` where the canonical current player's worker `1` stands, `0` else.   |
| **1**   | Current Worker 2          | `1` where the canonical current player's worker `2` stands, `0` else.   |
| **2**   | Opponent Worker 1         | `1` where the canonical opponent's worker `1` stands, `0` else.         |
| **3**   | Opponent Worker 2         | `1` where the canonical opponent's worker `2` stands, `0` else.         |
| **4**   | Level 1 Buildings         | `1` on spaces with exactly 1 block.                                      |
| **5**   | Level 2 Buildings         | `1` on spaces with exactly 2 blocks.                                     |
| **6**   | Level 3 Buildings         | `1` on spaces with exactly 3 blocks.                                     |
| **7**   | Domes                     | `1` on spaces with a dome.                                               |

_(Note: Ground level is implied when channels 4-7 are all 0)._

Worker identity is intentional. The compact action vector is split into "worker 1 actions" and "worker 2 actions", so the network must know which current-player worker each policy half refers to.

### 2. The Action Space (The Output Tensor)

In Santorini, a turn consists of **Move + Build**. You must map every possible Move+Build combination to a single flat array.

- **Workers:** You have up to 2 workers to choose from.
- **Move:** 8 possible directions (N, S, E, W, NE, NW, SE, SW).
- **Build:** 8 possible directions from the new location.
- **Total size:** $2 \times 8 \times 8 = 128$ possible discrete actions.

Your network's policy head will output a vector of size 128. Because many of these 128 moves will be illegal (e.g., moving off the board, moving into a dome), your engine will apply an **action mask**—forcing the probabilities of illegal moves to exactly $0$ before feeding them to the MCTS.

### 3. The Network Structure

Because the board is 5x5, a massive network will overfit and run too slowly. A lightweight **Convolutional Residual Network (ResNet)** is perfect here.

- **Body:** 5 to 10 Residual Blocks. Each block uses $3 \times 3$ convolutional layers with 64 to 128 filters.
- **Policy Head:** Outputs 128 logits. The wrapper applies log-softmax for training and MCTS masks illegal actions.
- **Value Head:** Flattens the output and passes it to a single neuron outputting a scalar between $-1$ (Current player loses) and $1$ (Current player wins) using a Tanh activation.

---

## Phase 3: The Training Loop

To train this on a laptop or free cloud service, you will iterate through this loop continuously.

1. **Self-Play (Data Generation):** Bottleneck step.
   The current best neural network plays against itself (using MCTS for move selection) for hundreds of games. It records the board state, the MCTS move probabilities, and the final game winner (+1 or -1) into a replay buffer.

2. **Network Training:** Requires GPU.
   Sample a random batch of board states from the replay buffer. Train the neural network to minimize the error between its Value prediction and the actual game winner, and to match its Policy prediction to the MCTS search probabilities.

3. **Arena Evaluation:** Quality Control.
   The newly trained network plays a tournament (e.g., 40 games) against the previous version of the network. If the new network wins a significant majority (e.g., >55%), it replaces the old network as the new champion.

4. **Repeat:**
   The new champion goes back to Step 1 to generate higher-quality self-play games.

---

## How to execute this on a budget

If you don't want to code the entire MCTS and PyTorch pipeline from scratch, I highly recommend leveraging the open-source **`alpha-zero-general`** repository on GitHub. It provides a generalized AlphaZero framework in Python.

**Your workflow would be:**

1. Clone the `alpha-zero-general` repo.
2. Harden `SantoriniGame.py` and `SantoriniLogic.py` with unit tests before trusting self-play data.
3. Add a Santorini-specific PyTorch wrapper that converts the compact rules board into the binary feature planes above.
4. Run the training loop on a free **Google Colab T4 GPU** or local GPU. Since Colab times out, save model checkpoints and replay-buffer examples frequently so training can resume.

Before any serious run, do a tiny end-to-end shakedown:

```bash
.venv/bin/python quick_santorini_train.py
```

This intentionally tiny run verifies self-play termination, MCTS integration, training batches, checkpoint writes, replay-buffer example saves, and arena evaluation. After it completes, use the baseline pit script to sanity-check a checkpoint against random or greedy play:

```bash
.venv/bin/python pit_santorini.py --baseline random --games 4 --sims 8
.venv/bin/python pit_santorini.py --baseline greedy --games 4 --sims 8
```

Then run a small-but-real local pilot before moving to Colab:

```bash
.venv/bin/python main_santorini.py --preset local --num-iters 5 --num-eps 5 --num-mcts-sims 8 --arena-compare 4 --epochs 1 --checkpoint ./temp/santorini_pilot/
```

If that run completes cleanly and still beats random, move to the full local preset:

```bash
.venv/bin/python main_santorini.py --preset local
```

Because Santorini is heavily tactical and ends relatively quickly, you should start seeing the AI discover basic blocking and "climbing" strategies within the first 10-20 iterations (usually 1-2 days of training on a Colab GPU).

---

## Current Implementation Notes

The base rules engine now has tests for randomized starts, dome movement, winning moves, build-on-vacated-square behavior, no-legal-move losses, symmetry masks, and tactical fixtures. Randomized starts should be used for training, with the filter that rejects positions where either player has both workers on outer-edge squares.

The first local AlphaZero run produced a model that beat random but lost badly to the one-ply greedy player. That showed the next bottleneck is tactical policy quality rather than just training loop plumbing.

To address that, we added a greedy curriculum:

```bash
.venv/bin/python pretrain_santorini_greedy.py --examples 5000 --epochs 5 --checkpoint-folder ./temp/santorini_greedy_pretrain/
```

This labels random reachable positions with the one-ply greedy action and saves a normal Santorini checkpoint. The initial run improved top-1 greedy-label support accuracy from `0.014` to `0.646`.

Evaluation after this greedy pretrain:

```bash
.venv/bin/python pit_santorini.py --baseline random --games 50 --sims 16 --checkpoint-folder ./temp/santorini_greedy_pretrain/ --json-out ./temp/santorini_greedy_pretrain/eval_random_50_s16.json
.venv/bin/python pit_santorini.py --baseline greedy --games 50 --sims 16 --checkpoint-folder ./temp/santorini_greedy_pretrain/ --json-out ./temp/santorini_greedy_pretrain/eval_greedy_50_s16.json
```

Results:

- vs random: `42-8`
- vs greedy: `2-48`

This is still far from beating greedy, but it is no longer a shutout.

We then tried resuming AlphaZero self-play from `./temp/santorini_greedy_pretrain/best.pth.tar`:

```bash
.venv/bin/python main_santorini.py --preset local --load-model --load-folder ./temp/santorini_greedy_pretrain/ --checkpoint ./temp/santorini_from_greedy/ --num-iters 20 --num-eps 20 --num-mcts-sims 32 --arena-compare 20 --epochs 3
```

The first four challengers were all rejected by the champion gate:

- iter 1: `5-15`
- iter 2: `6-14`
- iter 3: `8-12`
- iter 4: `9-11`

That suggests the direction may be improving slowly, but the local self-play recipe is too small/noisy for reliable champion replacement. After stopping that unchanged run, the standing checkpoint still evaluated at:

- vs random: `43-7`
- vs greedy: `4-46`

At a larger search budget, the same checkpoint did better but still lost:

- vs greedy, 64 sims: `7-13`

We also tried a denser tactical pretrain with immediate-win and block-now template positions:

```bash
.venv/bin/python pretrain_santorini_greedy.py --examples 5000 --epochs 5 --batch-size 64 --load-folder ./temp/santorini_greedy_pretrain/ --checkpoint-folder ./temp/santorini_tactical_pretrain/ --seed 4 --tactical-ratio 0.5 --tie-policy uniform
```

It fit the training targets well, but it overfit the tactical templates and got worse against greedy:

- vs random: `42-8`
- vs greedy: `0-50`

So the tactical generator should remain an explicit experiment, not the default. The greedy baseline breaks ties by choosing the lowest action id, so `--tie-policy first` is the correct default when the goal is specifically to beat that baseline.

We then scaled the broad greedy imitation curriculum from the earlier greedy checkpoint:

```bash
.venv/bin/python pretrain_santorini_greedy.py --examples 50000 --epochs 5 --batch-size 64 --load-folder ./temp/santorini_greedy_pretrain/ --checkpoint-folder ./temp/santorini_greedy_50k/ --tie-policy first --tactical-ratio 0.0 --seed 5
```

This produced a much better bootstrap checkpoint:

- examples: `50000`
- mean greedy tie count: `22.38`
- policy top-1 support accuracy: `0.410 -> 0.957`
- checkpoint: `./temp/santorini_greedy_50k/best.pth.tar`

Evaluation:

- vs random, 16 sims: `50-0`
- vs greedy, 64 sims: `25-25`
- vs greedy, 128 sims: `16-4`

This is the first checkpoint that can clearly beat the one-ply greedy baseline when given enough search. Next adjustment: resume AlphaZero self-play from `./temp/santorini_greedy_50k/best.pth.tar` with larger episode batches, at least 64 MCTS sims, and a less brittle early champion gate.

We then ran that self-play continuation:

```bash
.venv/bin/python main_santorini.py --preset local --load-model --load-folder ./temp/santorini_greedy_50k/ --checkpoint ./temp/santorini_selfplay_50k/ --num-iters 10 --num-eps 40 --num-mcts-sims 64 --arena-compare 40 --update-threshold 0.50 --epochs 3 --history-iters 5 --seed 6
```

Champion gate results:

- accepted: iters 1 (`25-15`), 2 (`25-15`), 4 (`21-19`), 6 (`24-16`), 7 (`21-19`), 10 (`23-17`)
- rejected: iters 3 (`18-22`), 5 (`12-28`), 8 (`16-24`), 9 (`16-24`)

Evaluation of `./temp/santorini_selfplay_50k/best.pth.tar`:

- vs random, 16 sims: `49-1`
- vs greedy, 64 sims: `29-21`
- vs greedy, 128 sims: `15-5`

This is now a genuine improvement over the broad greedy-imitation checkpoint at 64 sims (`25-25 -> 29-21`) while keeping the strong 128-sim result. The next best step is to continue this same self-play recipe for more iterations, but evaluate with a larger arena or multiple seeds before trusting small Elo-like differences. If local runtime stays comfortable, increase to `--num-eps 80`, `--arena-compare 80`, and keep `--num-mcts-sims 64`; otherwise continue with the current settings and accumulate more accepted champions.

We then launched the larger continuation from `./temp/santorini_selfplay_50k/best.pth.tar`:

```bash
.venv/bin/python main_santorini.py --preset local --load-model --load-folder ./temp/santorini_selfplay_50k/ --checkpoint ./temp/santorini_selfplay_50k_more/ --num-iters 10 --num-eps 80 --num-mcts-sims 64 --arena-compare 80 --update-threshold 0.50 --epochs 3 --history-iters 5 --seed 7
```

Because the console output was extremely verbose and the later replay window made each iteration much slower, we stopped the run after the iteration-6 promotion had saved a new `best.pth.tar`. Iteration 7 had started self-play, but it was interrupted before producing training data or a challenger.

Champion gate results for the completed iterations:

- rejected: iter 1 (`38-42`), iter 3 (`37-43`), iter 5 (`30-50`)
- accepted: iter 2 (`45-35`), iter 4 (`48-32`), iter 6 (`45-35`)

Evaluation of `./temp/santorini_selfplay_50k_more/best.pth.tar`:

- vs random, 16 sims: `50-0`
- vs greedy, 64 sims: `42-8`
- vs greedy, 128 sims: `19-1`

This clears the immediate milestone: the model now beats the one-ply greedy baseline convincingly at both 64 and 128 MCTS sims.

Before moving to Colab-scale runs, we tightened the training runner:

- replay examples are now saved as checkpoint-specific files, `latest.examples`, and `best.pth.tar.examples`
- `--load-examples` can resume from an explicit `--examples-file`, the model-adjacent examples file, `latest.examples`, `best.pth.tar.examples`, or the newest checkpoint examples; Santorini's normal `best.pth.tar` resume asks for `latest.examples` first for better interrupted-run continuity
- `--quiet` suppresses noisy progress bars while keeping epoch-level logs
- `--skip-first-self-play` preserves the old behavior only when we intentionally want to train on already-loaded examples before collecting fresh self-play
- Colab setup lives in `santorini/COLAB.md`

Next training direction: continue from `./temp/santorini_selfplay_50k_more/best.pth.tar` on Colab or a long local run with Drive/durable checkpoints, loading replay examples when available. Start with `--num-eps 80`, `--num-mcts-sims 64`, `--arena-compare 80`, `--update-threshold 0.50`, `--epochs 3`, `--history-iters 5`, and `--quiet`; after each long segment, evaluate against greedy at both 64 and 128 sims with at least 100 games.
