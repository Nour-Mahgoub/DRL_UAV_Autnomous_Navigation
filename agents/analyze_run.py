"""
Quick diagnostic for a completed training run.
Run: python agents/analyze_run.py
"""

import sys
import os
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

if len(sys.argv) > 1:
    rewards_path = os.path.join(RESULTS_DIR, sys.argv[1])
else:
    # default: most recently modified rewards file
    candidates = [f for f in os.listdir(RESULTS_DIR) if f.endswith("_episode_rewards_final.npy")]
    if not candidates:
        print("No results files found in results/. Pass a filename as an argument.")
        sys.exit(1)
    candidates.sort(key=lambda f: os.path.getmtime(os.path.join(RESULTS_DIR, f)))
    rewards_path = os.path.join(RESULTS_DIR, candidates[-1])
    print(f"No filename given — using most recent: {candidates[-1]}\n")

rewards = np.load(rewards_path)
n = len(rewards)

print(f"Total episodes: {n}")
print(f"First 10 avg:  {np.mean(rewards[:10]):.2f}")
print(f"Mid (ep {n//2-10}-{n//2+10}) avg: {np.mean(rewards[n//2-10:n//2+10]):.2f}")
print(f"Last 10 avg:   {np.mean(rewards[-10:]):.2f}")
print()
print(f"Best episode reward:  {rewards.max():.2f} (episode {rewards.argmax()+1})")
print(f"Worst episode reward: {rewards.min():.2f} (episode {rewards.argmin()+1})")
print()

# rolling average, printed in chunks of 20 to see the trend at a glance
chunk = 20
print(f"Reward trend (avg per {chunk}-episode block):")
for i in range(0, n, chunk):
    block = rewards[i:i+chunk]
    print(f"  Episodes {i+1:3d}-{min(i+chunk, n):3d}: {np.mean(block):7.2f}")