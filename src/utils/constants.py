"""
Poker constants and enumerations
"""

# Card rank values (2-14, where 11=J, 12=Q, 13=K, 14=A)
RANK_VALUES = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
}

# Rank names
RANK_NAMES = {
    2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
    9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'
}

# Suit names and symbols
SUIT_NAMES = {
    'h': 'hearts',
    'd': 'diamonds',
    'c': 'clubs',
    's': 'spades'
}

SUIT_SYMBOLS = {
    'h': '♥',
    'd': '♦',
    'c': '♣',
    's': '♠'
}

# Action constants
ACTION_NAMES = {
    0: 'fold',
    1: 'check',
    2: 'call',
    3: 'bet',
    4: 'raise',
    5: 'all_in'
}

ACTION_VALUES = {
    'fold': 0,
    'check': 1,
    'call': 2,
    'bet': 3,
    'raise': 4,
    'all_in': 5,
    'allin': 5
}

# Street constants
STREET_NAMES = {
    0: 'preflop',
    1: 'flop',
    2: 'turn',
    3: 'river',
    4: 'showdown'
}

STREET_VALUES = {
    'preflop': 0,
    'flop': 1,
    'turn': 2,
    'river': 3,
    'showdown': 4
}

# Position constants
POSITION_NAMES = {
    0: 'BTN',  # Button
    1: 'SB',   # Small Blind
    2: 'BB',   # Big Blind
    3: 'UTG',  # Under the Gun
    4: 'MP',   # Middle Position
    5: 'CO',   # Cutoff
}

POSITION_ORDER = ['BTN', 'SB', 'BB', 'UTG', 'UTG+1', 'MP', 'MP+1', 'CO', 'HJ']

# Hand rankings (higher is better)
HAND_RANKINGS = {
    'high_card': 0,
    'one_pair': 1,
    'two_pair': 2,
    'three_of_a_kind': 3,
    'straight': 4,
    'flush': 5,
    'full_house': 6,
    'four_of_a_kind': 7,
    'straight_flush': 8,
    'royal_flush': 9
}

HAND_RANK_NAMES = {
    0: 'High Card',
    1: 'One Pair',
    2: 'Two Pair',
    3: 'Three of a Kind',
    4: 'Straight',
    5: 'Flush',
    6: 'Full House',
    7: 'Four of a Kind',
    8: 'Straight Flush',
    9: 'Royal Flush'
}

# Default game constants
DEFAULT_STARTING_STACK = 1000.0
DEFAULT_SMALL_BLIND = 5.0
DEFAULT_BIG_BLIND = 10.0
DEFAULT_MIN_BUY_IN = 200.0
DEFAULT_MAX_BUY_IN = 10000.0

# Feature dimensions
FEATURE_DIM_HOLE_CARDS = 52
FEATURE_DIM_BOARD_CARDS = 52
FEATURE_DIM_STACK = 1
FEATURE_DIM_POT = 1
FEATURE_DIM_BET = 1
FEATURE_DIM_STREET = 4
FEATURE_DIM_LEGAL_ACTIONS = 6
FEATURE_DIM_OPPONENT_STACKS = 10
FEATURE_DIM_TIMING = 2
FEATURE_DIM_HISTORY = 10

TOTAL_FEATURE_DIM = (
    FEATURE_DIM_HOLE_CARDS +
    FEATURE_DIM_BOARD_CARDS +
    FEATURE_DIM_STACK +
    FEATURE_DIM_POT +
    FEATURE_DIM_BET +
    FEATURE_DIM_STREET +
    FEATURE_DIM_LEGAL_ACTIONS +
    FEATURE_DIM_OPPONENT_STACKS +
    FEATURE_DIM_TIMING +
    FEATURE_DIM_HISTORY
)

# Action indices
ACTION_FOLD = 0
ACTION_CHECK = 1
ACTION_CALL = 2
ACTION_BET = 3
ACTION_RAISE = 4
ACTION_ALL_IN = 5

# Street indices
STREET_PREFLOP = 0
STREET_FLOP = 1
STREET_TURN = 2
STREET_RIVER = 3
STREET_SHOWDOWN = 4

# Default bet sizes (as fraction of pot)
DEFAULT_BET_SIZES = {
    'preflop': 0.75,
    'flop': 0.66,
    'turn': 0.66,
    'river': 0.50
}

# Raise sizes (as fraction of pot)
DEFAULT_RAISE_SIZES = {
    'preflop': 0.80,
    'flop': 0.75,
    'turn': 0.75,
    'river': 0.60
}

# Bluff frequencies (by street)
DEFAULT_BLUFF_FREQUENCIES = {
    'preflop': 0.10,
    'flop': 0.20,
    'turn': 0.25,
    'river': 0.30
}

# Value thresholds
THRESHOLD_VERY_STRONG = 0.85
THRESHOLD_STRONG = 0.70
THRESHOLD_MODERATE = 0.50
THRESHOLD_WEAK = 0.30

# Number of cards
NUM_CARDS_IN_DECK = 52
NUM_HOLE_CARDS = 2
NUM_FLOP_CARDS = 3
NUM_TURN_CARDS = 1
NUM_RIVER_CARDS = 1
MAX_BOARD_CARDS = 5

# Maximum players
MAX_PLAYERS = 10
MIN_PLAYERS = 2
DEFAULT_PLAYERS = 6

# Path constants
DEFAULT_DATA_DIR = "data"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_RAW_DIR = "data/raw"
DEFAULT_CHECKPOINT_DIR = "experiments/checkpoints"
DEFAULT_LOG_DIR = "experiments/logs"
DEFAULT_RESULTS_DIR = "experiments/results"