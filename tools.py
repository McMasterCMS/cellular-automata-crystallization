import matplotlib.pyplot as plt
import numpy as np


def display_state(state):
    state = np.array(state)
    plt.figure(figsize=(10,10))
    plt.imshow(state[np.newaxis])

def display_growth(state_history):
    state_history = np.array(state_history)
    plt.figure(figsize=(8,8))
    plt.imshow(state_history)
