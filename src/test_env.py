"""
Quick script for checking the SlimeVolley environment setup.
"""

import gym
import slimevolleygym

env = gym.make("SlimeVolley-v0")
obs = env.reset()

print("Observation:", obs)
print("Action space:", env.action_space)

'''
cd /Users/shusato/Desktop/neat-slime
source .venv/bin/activate
PYTHONPATH=./slimevolleygym python src/test_env.py
'''
