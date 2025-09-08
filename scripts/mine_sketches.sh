#!/bin/bash
# Mine program sketches from training data
python tools/mine_sketches.py --train_json data/arc-agi_training_challenges.json --out models/sketches.json
