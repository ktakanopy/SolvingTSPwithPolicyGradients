# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 13:24:03 2016

Originally inspired by
http://outlace.com/Reinforcement-Learning-Part-3/

@author: jesseclark
"""
from __future__ import print_function
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.optimizers import RMSprop,SGD,Adam
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import model_from_json

import os
import copy
import numpy as np
import random
from IPython.display import clear_output
import time
import pickle
import GameEnv
import globals_variables as GV
from threading import Thread, Lock
from Queue import Queue

VERBOSE = GV.VERBOSE_LEVEL # levels: 0, 1 , 2

def printV(message,iterations=0):
    if VERBOSE == 1 and iterations % GV.PRINT_FACTOR == 0:
        print(message)
    if VERBOSE == 2:
        print(message)

def printTerminalReason(terminal_reason,n_games):
    if terminal_reason == 0:
        printV("\tnone",n_games)
    if terminal_reason == 1:
        printV("\thit wall",n_games)
    if terminal_reason == 2:
        printV("\thit boundry",n_games)
    if terminal_reason == 3:
        printV("\trepeated actions",n_games)
    if terminal_reason == 4:
        printV("\tpassed moves",n_games)
    if terminal_reason == 5:
        printV("\tall fixes collecteds",n_games)


def saveConfig(rw_list,model,average_n_items,average_score,total_time):
    m_name = 'models/model_%s' % GV.NX +'_%s' % GV.NY +'_%s' % GV.N_ITEMS + "_%s" % GV.N_PLAYERS + '_%.2f'%  GV.BEST_SCORE 
    save_model(model,m_name)

    obj = { 'rewards_collected' : rw_list,
            'average_n_items': average_n_items,
            'average_score': average_score,
            'epsilon_repetition': GV.EPSILON_REPETITION,
            'epsilon_min': GV.EPSILON_MIN,
            'epsilon_stop': GV.EPSILON_STOP,
            'epochs_fit': GV.EPOCHS_FIT,
            'max_repetitions': GV.MAX_REPETITIONS,
            'max_moves': GV.MAX_MOVES,
            'n_trajectories': GV.N_TRAJECTORIES,
            'n_games': GV.N_GAMES,
            'c_iterations': GV.C_ITERATIONS,
            'discount_factor': GV.DISCOUNT_FACTOR,
            'best_score': GV.BEST_SCORE,
            'total_time': total_time
    }
    print("** Saving model as m_name: " + m_name)
    fileNameConfig = m_name + ".config"
    fileObject = open(fileNameConfig,'wb')
    pickle.dump(obj,fileObject)
    fileObject.close()

def loadConfig(m_name):
    loaded_model = load_model(m_name)
    config_name = m_name + ".config"
    fileObject = open(config_name,'r')
    obj = pickle.load(fileObject)

    return loaded_model, obj

def create_pg_model_wrapper(img_channels,img_rows,img_cols,n_actions=4):

    printV("img_channels: " + str(img_channels))
    printV("img_rows: " + str(img_rows))
    printV("img_cols: " + str(img_cols))
    # try clipping or huber loss


    model = Sequential()
    model.add(Dense(8,  input_shape=(img_channels,img_rows,img_cols)))
    model.add(PReLU())
    # if img_rows < 8:
    # model.add(Conv2D(2, kernel_size=(2, 2), activation='relu', input_shape=(img_channels,img_rows,img_cols),padding='same'))
        # model.add(Conv2D(16, kernel_size=(2, 2),strides=(2,2), activation='relu',padding='same'))
    # else:
        # model.add(Conv2D(8, kernel_size=(6, 6),strides=(3,3), activation='relu', input_shape=(img_channels,img_rows,img_cols),padding='valid'))
        # model.add(Conv2D(16, kernel_size=(3, 3),strides=(2,2), activation='relu',padding='same'))
 
    model.add(Flatten())
    # model.add(Dropout(0.1))

    model.add(Dense(n_actions,activation='softmax'))

    # opt = RMSprop()
    opt = Adam(lr=GV.LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    return model


def create_pg_model(img_channels, img_rows, img_cols,n_actions=4):
    """
        Make a keras CNN model for policy gradient.
    """
    if GV.ASYNC:
        GV.GRAPH  = tf.get_default_graph()
    
    if GV.ASYNC:
        with GV.GRAPH.as_default():
            model = create_pg_model_wrapper(img_channels, img_rows, img_cols, n_actions)
            model._make_predict_function()  # have to initialize before threading
    else:
        model = create_pg_model_wrapper(img_channels, img_rows, img_cols, n_actions)

    return model

def save_model(model, m_name):
    """Save keras model to json and weights to h5. """

    json_string = model.to_json()
    open(m_name+'.json', 'w').write(json_string)
    model.save_weights(m_name+'.h5')


def load_model(m_name):
    """Load keras model from json and h5."""

    GV.BEST_SCORE = float(m_name.split('_')[-1])

    model_l = model_from_json(open(m_name+'.json').read())
    model_l.load_weights(m_name+'.h5')
    opt = Adam(lr=GV.LEARNING_RATE)
    model_l.compile(loss='categorical_crossentropy', optimizer=opt)

    if GV.ASYNC:
        model_l._make_predict_function()  # have to initialize before threading
        GV.GRAPH  = tf.get_default_graph()

    return model_l


def transfer_dense_weights(model1, model2):
    """
     Transfer weights for dense layers between keras models.
     transfer model1 to model2
    """
    for ind in range(len(model2.layers)):
        if 'dense' in model2.layers[ind].get_config()['NAME'].lower():
            try:
                printV('*')
                weights = copy.deepcopy(model1.layers[ind].get_weights())
                model2.layers[ind].set_weights(weights)
            except:
                printV('!')
    return model2


def transfer_conv_weights(model1, model2):
    """
     Transfer weights for conv layers between keras models.
     Transfer model1 to model2.
    """

    for ind in range(len(model2.layers)):
        if 'convolution' in model2.layers[ind].get_config()['NAME'].lower():
            try:
                weights = copy.deepcopy(model1.layers[ind].get_weights())
                model2.layers[ind].set_weights(weights)
            except:
                printV('!')
    return model2


def transfer_all_weights(model1,model2):
    """ Transfer all weights between keras models"""

    for ind in range(len(model2.layers)):
        try:
            weights = copy.deepcopy(model1.layers[ind].get_weights())
            model2.layers[ind].set_weights(weights)
        except:
            printV('!')

    return model2


def create_duplicate_model(model):
    """Create a duplicate keras model."""
    new_model = Sequential.from_config(model.get_config())
    new_model.set_weights(copy.deepcopy(model.get_weights()))
    new_model.compile(loss=model.loss,optimizer=model.optimizer)

    return new_model


def add_to_replay(replay, state, action, reward, new_state, replay_buffer, n_times=1):
    # append to replay
    [replay.append((state.copy(), action, reward, new_state.copy())) for ind in range(n_times)]

    [replay.pop(0) for ind in range(n_times) if len(replay) > replay_buffer]

def sample_minibatch(replay, minibatch_size, priority=False):
    # priority replay minibatch sampling
    return random.sample(replay, minibatch_size)

def prepare_game():
    Game = GameEnv.WhGame(GV.NX,GV.NY,n_fixes=GV.N_ITEMS,n_players=GV.N_PLAYERS,
            wall_loc=GV.WALL_LOC,term_on_collision=GV.TERM_ON_COLLISION, n_frames=GV.N_FRAMES)

    # set this ti False for 2D only, will need to adjust the input for the netwrok accordingly
    Game.RGB = GV.RGB
    Game.init_game()

    # do nothing frame at the start - 4 is the do nothing move number (anything > 3 will do nothing)
    _ = [Game.frame_step(4, ind) for ind in range(GV.N_PLAYERS) for dind in range(GV.N_FRAMES)]

    return Game

def play_in_training(model,t_queue,n_games):

    #init game 
    Game = prepare_game()
    lock = Lock()
    terminal = False
    moves_game = 0
    random_takes = 0.0
    n_games += 1
    no_random_takes = 0.0
    total_reward = 0
    trajectory = [[],[],[]]

    while(terminal == False):
        moves_game +=1
        cur_ind = (moves_game % GV.N_PLAYERS)

        state = Game.game['s_t'][cur_ind].copy().reshape((1,GV.IMG_CHANNELS,Game.ny,Game.nx))

        action_probs = None

        if GV.ASYNC:
            with GV.GRAPH.as_default():
                action_probs = model.predict(state)
        else:
            action_probs = model.predict(state)

        if (random.random() < GV.EPSILON) or n_games < GV.OBSERVE: # exploration vs explotation
            random_takes += 1.0
            action = np.random.randint(0,GV.N_ACTIONS)
        else:
            no_random_takes += 1.0
            action = np.argmax(action_probs)

        x_t, reward, terminal = Game.frame_step(action, cur_ind)
        total_reward += reward
        # getting trajectories
        trajectory[0].append(state.copy().squeeze())
        trajectory[1].append(reward)
        trajectory[2].append(action_probs.squeeze())

        passed_total_moves = moves_game >= GV.MAX_MOVES
        if passed_total_moves:
            Game.terminal_reason = 4 #passed moves
            terminal =  True

        if GV.EPSILON_EXPLORE == False and terminal == True and Game.terminal_reason != 5: # all fixes collected
            lock.acquire()
            GV.REPEATED_TERMINAL_STATES[Game.terminal_reason] += 1
            lock.release()
    printV("\tThe game terminated.",n_games)

    GV.REWARDS_COLLECTED.append(total_reward)
    GV.NORM_ITEMS_COLLECTED.append(1.0*Game.game['n_fixes_collected']/Game.game['n_fixes_start'])
    GV.N_MOVES_MADE.append(moves_game)

    clear_output(wait=True)

    # display some  during
    printV("\tRandom takes %.2f" % (random_takes/(random_takes + no_random_takes)) ,n_games)
    printV("\tNo random takes %.2f" % (no_random_takes/(random_takes + no_random_takes)) ,n_games)
    printV("\tGame #: %s" % (n_games,),n_games)
    printV("\tMoves this round %s" % moves_game,n_games)
    printV("\tItems collected %s" % Game.game['n_fixes_collected'],n_games)
    printV("\tTotal reward: " + str(total_reward),n_games)
    printV("\tEpsilon %s" % GV.EPSILON,n_games)
    printTerminalReason(Game.terminal_reason,n_games)
    printV('',n_games)

    if VERBOSE == 2:
        # keep track of terminal reasons - good for debugging
        GV.TERM_REASONS.append(Game.terminal_reason)

    t_queue.put(trajectory)

def collect_trajectories(n_games,model):

    trajectories = [[],[],[]]

    t = 0
    while(t < GV.N_TRAJECTORIES):
        procs = []

        t_queue = Queue()

        dispatched_threads = 0
        while dispatched_threads < GV.N_THREADS and dispatched_threads < (GV.N_TRAJECTORIES - t):
            procs.append(Thread(target=play_in_training,args=(model,t_queue,n_games)))

            if n_games > GV.OBSERVE:
                GV.EPSILON = max(GV.EPSILON_MIN,GV.EPSILON - 1./GV.EPSILON_STOP) # update epsilon after each game

            if n_games > GV.OBSERVE and max(GV.REPEATED_TERMINAL_STATES) > GV.MAX_REPETITIONS:
                print("Epsilon restore!!! ****")
                GV.REPEATED_TERMINAL_STATES = [ 0 for i in range(0,6) ]
                GV.EPSILON = GV.EPSILON_REPETITION

            if GV.EPSILON != GV.EPSILON_MIN: # activate epsilon explore
                GV.EPSILON_EXPLORE = True
            else:
                GV.EPSILON_EXPLORE = False

            if n_games % GV.CONFIG_FACTOR == 0: # save model after a certain number of games
                saveConfig(GV.REWARDS_COLLECTED,model, np.mean(GV.NORM_ITEMS_COLLECTED[-GV.N_LAST_TO_COMPARE:]), np.mean(GV.REWARDS_COLLECTED[-GV.N_LAST_TO_COMPARE:]), time.time() + GV.INIT_TRAINING_TIME)

                GV.NORM_ITEMS_COLLECTED = GV.NORM_ITEMS_COLLECTED[-GV.N_LAST_TO_COMPARE*2:] # removing excess

            n_games += 1
            GV.C_ITERATIONS += 1
            dispatched_threads+=1

        for proc in procs:
            proc.start()

        for proc in procs:
            proc.join()

        t += dispatched_threads
        # t_queue.close()
        # t_queue.join_thread()
        while not t_queue.empty():
            el = t_queue.get()
            trajectories[0].extend(el[0])
            trajectories[1].extend(el[1])
            trajectories[2].extend(el[2])

    return trajectories,n_games

def collect_trajectories_sync(n_games,model):

    trajectories = []
    trajectory = [[],[],[]]

    #while game still in progress
    while(len(trajectories) < GV.N_TRAJECTORIES):

        #init game 
        Game = prepare_game()

        terminal = False

        moves_game = 0
        random_takes = 0.0
        n_games += 1
        no_random_takes = 0.0
        total_reward = 0
        while(terminal == False):

            # game move counter
            moves_game +=1
            # get one of the current states
            cur_ind = (moves_game % GV.N_PLAYERS)

            # get the current concatenated game state (constructed within GameEnv,
            # could be done here)
            state = Game.game['s_t'][cur_ind].copy().reshape((1,GV.IMG_CHANNELS,Game.ny,Game.nx))

            # We are in state S
                # Let's run our Q function on S to get Q values for all possible actions
            action_probs = model.predict(state)

            # choose random action, 4 is for up/down/left/right -
            # the number of possible moves
            if (random.random() < GV.EPSILON) or n_games < GV.OBSERVE:
                random_takes += 1.0
                action = np.random.randint(0,GV.N_ACTIONS)
            else:
                no_random_takes += 1.0
                action = np.argmax(action_probs)

            # Take action, observe new state S' and get terminal
            # terminal - all items collected, hit wall, hit boundry, repeated actions
            # still an edge case in multiplayer that needs to be addressed
            x_t, reward, terminal = Game.frame_step(action, cur_ind)

            # make the state histroy for player cur_ind
            # new_state = Game.game['S_T'][cur_ind].copy().reshape((1,['IMG_CHANNELS'],Game.ny,Game.nx))

            # getting trajectories
            total_reward += reward
            trajectory[0].append(state.copy().squeeze())
            trajectory[1].append(reward)
            trajectory[2].append(action_probs.squeeze())

            passed_total_moves = moves_game >= GV.MAX_MOVES
            if passed_total_moves:
                printV("\tPassed moves",n_games)
                terminal =  True
            if terminal:
                printV("\tThe game terminated.",n_games)
                # metrics to keep track of game learning progress
                GV.REWARDS_COLLECTED.append(total_reward)
                GV.N_MOVES_MADE.append(moves_game)
                GV.NORM_ITEMS_COLLECTED.append(1.0*Game.game['n_fixes_collected']/Game.game['n_fixes_start'])

                clear_output(wait=True)

                # display some  during
                printV("\tRandom takes %.2f" % (random_takes/(random_takes + no_random_takes)) ,n_games)
                printV("\tNo random takes %.2f" % (no_random_takes/(random_takes + no_random_takes)) ,n_games)
                printV("\tGame #: %s" % (n_games,),n_games)
                printV("\tMoves this round %s" % moves_game,n_games)
                printV("\tItems collected %s" % Game.game['n_fixes_collected'],n_games)
                printV("\tTotal reward: " + str(total_reward),n_games)
                printV("\tEpsilon %s" % GV.EPSILON,n_games)
                printV("\t" + Game.terminal_reason + "\n",n_games)

                trajectories.append(trajectory)

                if VERBOSE == 2:
                    # keep track of terminal reasons - good for debugging
                    GV.TERM_REASONS.append(Game.terminal_reason)


        # decrement epsilon over games
        if GV.EPSILON > GV.EPSILON_MIN:
            GV.EPSILON -= (1./GV.EPSILON_STOP)
        else:
            GV.EPSILON_EXPLORE = False
            GV.EPSILON = GV.EPSILON_MIN

        if n_games % GV.CONFIG_FACTOR == 0: # save model after a certain number of games
                saveConfig(GV.REWARDS_COLLECTED,model, np.mean(GV.NORM_ITEMS_COLLECTED[-GV.N_LAST_TO_COMPARE:]), np.mean(GV.REWARDS_COLLECTED[-GV.N_LAST_TO_COMPARE:]), time.time() + GV.INIT_TRAINING_TIME)

    return trajectories,n_games




def train(model,model_best):
    """
        Train the Q network using RL.
    """
    GV.INIT_TRAINING_TIME = time.time()
    n_games = 0
    # iterate over the games
    for i in range(GV.N_GAMES):

        printV("Collecting trajectories..",n_games)
        if GV.ASYNC:
            trajectories,n_games = collect_trajectories(n_games,model)
        else:
            trajectories,n_games = collect_trajectories_sync(n_games,model)
        # fitting the model by the collected trajectories
        printV("Fitting the model",n_games)
        X_train, y_train, discounted_rewards = process_trajectory(trajectories,GV.DISCOUNT_FACTOR)

        advantage = discounted_rewards  - discounted_rewards.mean() # reward normalizatio
        advantage /= np.std(discounted_rewards)

        if GV.ASYNC:
            with GV.GRAPH.as_default():
                model_temp = model.fit(X_train, y_train,sample_weight=advantage,epochs=GV.EPOCHS_FIT, verbose=0)
        else:
            model_temp = model.fit(X_train, y_train,sample_weight=advantage,epochs=GV.EPOCHS_FIT, verbose=0)
        # model.train_on_batch(X_train, y_train,sample_weight=advantage)

        # GV.LOSS.append(model_temp.history['loss'][0])

        if np.mean(GV.REWARDS_COLLECTED[-GV.N_LAST_TO_COMPARE:]) > GV.BEST_SCORE and  GV.EPSILON_EXPLORE == False and n_games > GV.OBSERVE: # updating the model
            model_best = transfer_all_weights(model,model_best)
            GV.BEST_SCORE = np.mean(GV.REWARDS_COLLECTED[-GV.N_LAST_TO_COMPARE:])
            saveConfig(GV.REWARDS_COLLECTED,model, np.mean(GV.NORM_ITEMS_COLLECTED[-GV.N_LAST_TO_COMPARE:]), np.mean(GV.REWARDS_COLLECTED[-GV.N_LAST_TO_COMPARE:]), time.time() + GV.INIT_TRAINING_TIME)
            print('^^ Updated best ^^ ***')

        printV("Avg. items (normalized) %s " % np.mean(GV.NORM_ITEMS_COLLECTED[-GV.N_LAST_TO_COMPARE:]),n_games)
        printV("Avg. score %s" % np.mean(GV.REWARDS_COLLECTED[-GV.N_LAST_TO_COMPARE:]),n_games)
        printV("Current Best Score %.2f" % (GV.BEST_SCORE),n_games)
        printV("Current Game %d" % n_games,n_games)
        printV("Current time: %f" % (time.time() + GV.INIT_TRAINING_TIME), n_games)
    saveConfig(GV.REWARDS_COLLECTED,model, np.mean(GV.NORM_ITEMS_COLLECTED[-GV.N_LAST_TO_COMPARE:]), np.mean(GV.REWARDS_COLLECTED[-GV.N_LAST_TO_COMPARE:]), time.time() + GV.INIT_TRAINING_TIME)
    print("Total time: %f" % (time.time() + GV.INIT_TRAINING_TIME))

    return model


def process_trajectory(trajectory, gamma=0.9):

    """Process the trajectory to get the X,Y and discounted rewards"""

    X_train = []
    y_train = []

    discounted_rewards = np.zeros(len(trajectory[1]))

    rewards = trajectory[1]
    running_add = 0

    # Loop through our batch and create arrays for X and y
    # so that we can fit our model at every step.
    for t in reversed(xrange(0,len(rewards))):
        reward = rewards[t]
        running_add = running_add * gamma + reward
        discounted_rewards[t] = running_add

    X_train = np.array(trajectory[0])
    y_train = np.array(trajectory[2])

    return X_train, y_train, discounted_rewards


def play(model):
    """
    s
        Play using the Q network.
    """

    # moves played in the game
    PLAYED_FRAMES = []

    Game = GameEnv.WhGame(GV.NX,GV.NY,n_fixes=GV.N_ITEMS,
            n_players=GV.N_PLAYERS,wall_loc=GV.WALL_LOC,
            term_on_collision=GV.TERM_ON_COLLISION, n_frames=GV.N_FRAMES)

    Game.RGB = GV.RGB
    Game.init_game()

    # do nothing frame at the start - 4 is the do nothing move number (anything > 3 will do nothing)
    _ = [Game.frame_step(4, ind) for ind in range(GV.N_PLAYERS) for dind in range(GV.N_FRAMES)]

    status = 1
    moves_game = 0

    random_takes = 0
    no_random_takes = 0.0

    # while game still in progress
    while(status == 1 and moves_game < GV.MAX_MOVES_IN_PLAY):

        # game move counter
        moves_game +=1

        # get one of the current states
        cur_ind = (moves_game % GV.N_PLAYERS)
        # don't continue if no result after max_moves - only terminal
        # condition we set outside the game
        # although we don't adjust the Q update with this one
        if moves_game >= GV.MAX_MOVES_IN_PLAY:
            printV("## MAX MOVES ##")
            Game.terminal_reason = 'maximum moves'
            status = 0

        # get the current concatenated game state (constructed within GameEnv,
        # could be done here)
        state = Game.game['s_t'][cur_ind].copy().reshape((1,GV.IMG_CHANNELS,Game.ny,Game.nx))

        # We are in state S
        # get the action by the model.

        if GV.ASYNC:
            with GV.GRAPH.as_default():
                action_probs = model.predict(state)
        else:
            action_probs = model.predict(state)

       # choose random action, 4 is for up/down/left/right -
        # the number of possible moves

        no_random_takes += 1.0
        action = np.argmax(action_probs)

        # Take action, observe new state S' and get terminal
        # terminal - all items collected, hit wall, hit boundry, repeated actions
        x_t, reward, terminal = Game.frame_step(action, cur_ind)

        # store the time step
        PLAYED_FRAMES.append(x_t)

        # stop playing if we are terminal
        if terminal:
            status = 0

    clear_output(wait=True)
    GV.N_MOVES_PLAYED = moves_game
    # display some  during

    printV("\tRandom takes: %.2f" % (random_takes/(random_takes + no_random_takes)))
    printV("\tNo random takes: %.2f" % (no_random_takes/(random_takes + no_random_takes)))
    printV("\tMoves this round %s" % moves_game)
    printV("\tItems collected %s" % Game.game['n_fixes_collected'])
    printTerminalReason(Game.terminal_reason,0)

    printV("\tFinished")

    return PLAYED_FRAMES

