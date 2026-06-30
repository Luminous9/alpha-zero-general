# Training Logs

1. Run #1 - 2026/06/28
   Platform: Google Colab (CPU)
   Iterations: 11
   Time: 8 hours
   Notes:
   Results:
    - vs previous local best, 64 sims: 68-32
    - vs previous local best, 128 sims: 71-29
    - vs greedy, 64 sims: 92-8
    - _promoted as new baseline_

2. Run #2 - 2026/06/29
   Platform: Google Colab (GPU)
   Iterations: 10
   Time: 30 minutes
   Notes: We made several performance improvements in code, and adjusted parameters of training to take advantage of hardware better, and switched to GPU instead of CPU
   Results:
    - vs previous run best, 128 sims: 66-34
    - vs greedy, 128 sims: 95-5
    - _promoted as new baseline_

3. Run #3 - 2026/06/29
   Platform: Google Colab (GPU)
   Iterations: 20
   Time: 57 minutes
   Notes: Increased iterations
   Results:
    - vs previous run best, 128 sims: 72-28
    - vs greedy, 128 sims: 97-3
    - _promoted as new baseline_

4. Run #4 - 2026/06/29
   Platform: Google Colab (CPU)
   Iterations: 8
   Time: 3 hours
   Notes:
   Results:
    - vs previous run best, 128 sims: 58-42
    - vs greedy, 128 sims: 100-0
    - _promoted as new baseline_

5. Run #5 - 2026/06/29
   Platform: Kaggle (GPU)
   Iterations: 20
   Time: 1 hour
   Notes:
   Results:
    - vs previous run best, 128 sims: 64-36
    - vs greedy, 128 sims: 100-0
    - _promoted as new baseline_

6. Run #6 - 2026/06/29
   Platform: Kaggle (GPU)
   Iterations: 20
   Time: 1 hour
   Notes:
   Results:
    - vs previous run best, 128 sims: 73-27
    - vs greedy, 128 sims: 98-2
    - _promoted as new baseline_
