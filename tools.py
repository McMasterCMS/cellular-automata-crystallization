import matplotlib.pyplot as plt
import numpy as np
from numpy import array


def display_state(state):
    state = np.array(state)
    plt.figure(figsize=(10,10))
    plt.imshow(state[np.newaxis])


def display_growth(state_history, show_fraction=False):
    state_history = np.array(state_history)
    fig = plt.figure(figsize=(14,7))
    ax1 = fig.add_subplot('121')
    ax1.imshow(state_history)
    if show_fraction:
        ax2 = fig.add_subplot('122')
        size = len(state_history[0])
        state_solid_count = np.sum(array(state_history), axis=1)
        state_solid_fraction = state_solid_count / size
        ax2.plot(state_solid_fraction)

# def display_growth_2D():



