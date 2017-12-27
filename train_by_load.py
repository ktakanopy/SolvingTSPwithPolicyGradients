from __future__ import print_function
import sys
import globals_variables as GV

import RL_Functions as Functions
import GameEnv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# m_name = 'models/model_%s' % GV.NX +'_%s' % GV.NY +'_np%s' % GV.N_PLAYERS +'_%.2f'%  GV.BEST_SCORE + "_%d" % GV.N_GAMES + "_%d" % GV.N_TRAJECTORIES


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
# store the network and copies for the target and best 
model_best = Functions.create_duplicate_model(model)

GV.C_ITERATIONS = obj['c_iterations']
GV.OBSERVE = 0
GV.EPSILON = GV.EPSILON_MIN
GV.INIT_TRAINING_TIME = obj['total_time']
model_best = Functions.train(model,model_best)

while True:
    a = raw_input()

    GV.PLAYED_FRAMES = Functions.play(model_best)

    if GV.RGB:
        GameEnv.output_sequence_RGB(GV.PLAYED_FRAMES,GV.IMAGES_DIR, GV.N_PLAYERS)
    else:
        GameEnv.output_sequence_Y(GV.PLAYED_FRAMES,GV.IMAGES_DIR, GV.N_PLAYERS)


    if a == 'sair':
        exit(0)

