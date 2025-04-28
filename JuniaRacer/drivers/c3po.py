# drivers/c3po.py
import numpy as np
from random import uniform, randint

name = "C3PO_QLEARN"
BRAKE = 0
ACCELERATE = 1
LEFT5 = 2
RIGHT5 = 3
NOTHING = 4

ALPHA = 0.1
GAMMA = 0.9
EPSILON_START = 0.5  # Initial epsilon
EPSILON_END = 0.1    # Minimum epsilon
EPSILON = EPSILON_START  # Current epsilon, updated by training

DISTANCE_BINS = [0, 10, 25, 50, 100, float('inf')]  # From refined state space
VELOCITY_BINS = [0, 2, 5, 8, 10]
ACCELERATION_BINS = [-0.2, 0, 0.2]

state_space_size = (len(DISTANCE_BINS)-1)**5 * (len(VELOCITY_BINS)-1) * (len(ACCELERATION_BINS)-1)  # 25,000
action_space_size = 5
q_table = np.zeros((state_space_size, action_space_size))
last_state = None
last_action = None
last_reward = None

def discretize(value, bins):
    for i in range(len(bins)-1):
        if bins[i] <= value < bins[i+1]:
            return i
    return len(bins) - 2

def get_state_index(d1, d2, d3, d4, d5, velocity, acceleration):
    s1 = discretize(d1, DISTANCE_BINS)
    s2 = discretize(d2, DISTANCE_BINS)
    s3 = discretize(d3, DISTANCE_BINS)
    s4 = discretize(d4, DISTANCE_BINS)
    s5 = discretize(d5, DISTANCE_BINS)
    s6 = discretize(velocity, VELOCITY_BINS)
    s7 = discretize(acceleration, ACCELERATION_BINS)
    index = (s1 + s2 * 5 + s3 * 25 + s4 * 125 + s5 * 625 + s6 * 3125 + s7 * 12500)
    return index

def setup():
    global q_table
    print(f"{name} driver setup...")
    try:
        q_table = np.load("q_table.npy")
        print("Table Q chargée depuis q_table.npy")
    except FileNotFoundError:
        print("Aucune table Q trouvée, initialisation à zéro")
    return 0

def drive(d1, d2, d3, d4, d5, velocity, acceleration):
    global last_state, last_action, last_reward, q_table, EPSILON
    state = get_state_index(d1, d2, d3, d4, d5, velocity, acceleration)
    if last_state is not None and last_action is not None and last_reward is not None:
        max_future_q = np.max(q_table[state])
        current_q = q_table[last_state, last_action]
        new_q = current_q + ALPHA * (last_reward + GAMMA * max_future_q - current_q)
        q_table[last_state, last_action] = new_q
    
    # Predictive safety check
    danger_zone = velocity + 10  # Buffer of 10 pixels
    if min(d1, d2, d3, d4, d5) < danger_zone:
        if d1 < danger_zone:
            action = BRAKE  # Brake if front is close
        elif min(d2, d4) < danger_zone:
            action = LEFT5  # Turn left if right side is close
        elif min(d3, d5) < danger_zone:
            action = RIGHT5  # Turn right if left side is close
        else:
            action = BRAKE  # Default to braking
    elif uniform(0, 1) < EPSILON:
        action = randint(0, action_space_size - 1)
    else:
        action = np.argmax(q_table[state])
    
    last_state = state
    last_action = action
    last_reward = None
    return action

def update_reward(reward):
    global last_reward
    last_reward = reward

def save_q_table():
    np.save("q_table.npy2", q_table)
    print("Table Q sauvegardée dans q_table.npy2")