
# x dimension in pixels
NX = 10#17#32#17#32#17
# y dimension
NY = 10#11#18#11*2
# number of previous frames to use for s
N_FRAMES = 4
# number of items to collect per player
N_ITEMS = 10#*2
# number of players
N_PLAYERS = 1#3
# number of allowed actions (up,down,left,right)
N_ACTIONS= 4

# a list of cords for the obstacle locations [[y1,x1],[y2,x2]...]
WALL_LOC= []

RGB = True
TERM_ON_COLLISION = False
# we use a seperate channel for the game (3 in total) - can also map to a single frame
IMG_CHANNELS = N_FRAMES*3

if not RGB:
    IMG_CHANNELS = N_FRAMES*1

LEARNING_RATE = .001

REPEATED_TERMINAL_STATES = [ 0 for i in range(0,6) ]

## MODEL

# number of trajectories to collect in training
N_TRAJECTORIES = 20

# number of games to play
N_GAMES = 50000
C_ITERATIONS = 0 # number of interations runned

# print variables
PRINT_PARTS = 100000
PRINT_FACTOR = (N_TRAJECTORIES * N_GAMES ) / PRINT_PARTS

# loss list
LOSS = []

# number of frames to observe before training
OBSERVE= 200

# max moves before terminating a game
# MAX_MOVES = 300
MAX_MOVES = (NX * NY ) * (N_ITEMS * 4)
MAX_MOVES_IN_PLAY = MAX_MOVES

# initial epsilon - explore vs exploit
EPSILON = 1

# lowest epsilon - also use this for playing
EPSILON_MIN = .05

# when to stop annealing epsilon - after this many games
EPSILON_STOP = 2000

# should the game terminate with a collision? will only terminate with max_moves when False

TERM_ON_COLISION = False

DISCOUNT_FACTOR = 0.99

IMAGES_DIR = 'images/'

# keep track of some paramters
BEST_SCORE = -9999999999999

# ideally we set these outside
# number of items and normalised number of items collected
N_ITEMS_COLLECTED = []
NORM_ITEMS_COLLECTED = []

# moves per game
N_MOVES_MADE = []

# why did the game terminate
TERM_REASONS = []
DQ_ERRORS = []

# save model and log factor
CONFIG_FACTOR = (N_TRAJECTORIES * N_GAMES ) // 100

INIT_TRAINING_TIME = 0.0

GRAPH = None

N_CPUS = 8
N_THREADS = N_CPUS * 2

MAX_REPEATED_ACTIONS = int(round(MAX_MOVES * 0.4))

MAX_REPEATED_ACTIONS = MAX_REPEATED_ACTIONS if MAX_REPEATED_ACTIONS % 2 == 0 else MAX_REPEATED_ACTIONS + 1

VERBOSE_LEVEL = 1

REWARDS_COLLECTED = []

MAX_REPETITIONS = ( N_TRAJECTORIES * N_GAMES ) / 300
EPSILON_REPETITION = .1

N_LAST_TO_COMPARE = N_TRAJECTORIES

ASYNC = True

SESSION = None

EPOCHS_FIT = N_TRAJECTORIES

EPSILON_EXPLORE = True
