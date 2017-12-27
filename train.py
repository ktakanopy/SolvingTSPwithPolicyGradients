from __future__ import print_function

import globals_variables as GV
import RL_Functions as Functions
import GameEnv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

model = Functions.create_pg_model(GV.IMG_CHANNELS, GV.NY, GV.NX, n_actions=GV.N_ACTIONS)

# store the network and copies for the target and best 
model_best = Functions.create_duplicate_model(model)

model = Functions.train(model,model_best)

while True:
    a = raw_input()

    GV.PLAYED_FRAMES = Functions.play(model)

    if GV.RGB:
        GameEnv.output_sequence_RGB(GV.PLAYED_FRAMES,GV.IMAGES_DIR, GV.N_PLAYERS)
    else:
        GameEnv.output_sequence_Y(GV.PLAYED_FRAMES,GV.IMAGES_DIR, GV.N_PLAYERS)

    if a == 'sair':
        exit(0)

