from __future__ import print_function

import globals_variables as GV
import RL_Functions as Functions
import GameEnv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

model,obj = Functions.loadConfig(sys.argv[1])

print("Model loaded")
print("Average Score (Rewards)",obj['average_score'])
print("Avarage Norm Items",obj['average_n_items'])
print("Epsilon min",obj['epsilon_min'])
print("Epsilon repetition",obj['epsilon_repetition'])
print("Epsilon stop",obj['epsilon_stop'])
print("Epochs fit",obj['epochs_fit'])
print("Max repetitions",obj['max_repetitions'])
print("Max moves",obj['max_moves'])
print("Number of trajectories",obj['n_trajectories'])
print("N games",obj['n_games'])
print("Number of iterations calculated in model",obj['c_iterations'])
print("Discount factor",obj['discount_factor'])
print("Best score",obj['best_score'])
print("Total time in training",obj['total_time'])
print()
while True:

    GV.PLAYED_FRAMES = Functions.play(model)

    if GV.RGB:
        GameEnv.output_sequence_RGB(GV.PLAYED_FRAMES,GV.IMAGES_DIR, GV.N_PLAYERS)
    else:
        GameEnv.output_sequence_Y(GV.PLAYED_FRAMES,GV.IMAGES_DIR, GV.N_PLAYERS)

    a = raw_input()
    if a == 'sair':
        exit(0)

