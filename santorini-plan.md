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
