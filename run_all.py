"""
run_all.py

Runs the whole project from scratch, in order:
  1. features.py       build the features         -> data/features.csv
  2. test_features.py  check there is no leakage
  3. train.py          train and save the models   -> models/
  4. evaluate.py       test on 2025                -> results/

How to run (from the project folder):  python run_all.py
"""

import os
import sys

# Let Python find the files in src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import evaluate
import features
import test_features
import train

print("=== Step 1: features ===")
features.main()

print("\n=== Step 2: leakage tests ===")
test_features.test_future_does_not_change_past()
test_features.test_same_day_matches_dont_see_each_other()

print("\n=== Step 3: training ===")
train.main()

print("\n=== Step 4: evaluation ===")
evaluate.main()
