#!/bin/bash
# Train neural guidance classifier from training data
python tools/train_guidance.py --train_json data/arc-agi_training_challenges.json --out models/guidance.pkl
