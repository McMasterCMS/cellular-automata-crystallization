
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from matplotlib.animation import FuncAnimation
import copy
from random import randrange


def display_state(state):
    state = np.array(state)
    plt.figure(figsize=(10,10))
    plt.imshow(state[np.newaxis])


def display_growth(state_history, show_fraction=False):

    state_history = np.array(state_history)
    fig = plt.figure(figsize=(14,7))

    if not show_fraction:
        ax_instant = plt.subplot2grid((4,1),(0, 0))
        ax_instant.imshow(state_history[0][np.newaxis])

        ax_overall = plt.subplot2grid((4,1),(1, 0), rowspan=3)
        ax_overall.imshow(state_history)

    # ax1 = plt.subplot2grid((4,2),(0, 0))
    # ax2 = plt.subplot2grid((4,2),(1, 0), rowspan=3)
    # ax3 = plt.subplot2grid((4,2),(0, 1), rowspan=4, colspan=4)

    else:

        ax_instant = plt.subplot2grid((4,2),(0, 0))
        ax_instant.imshow(state_history[0][np.newaxis])

        ax_overall = plt.subplot2grid((4,2),(1, 0), rowspan=3)
        ax_overall.imshow(state_history, aspect='equal')

        size = len(state_history[0])
        state_solid_count = np.sum(array(state_history), axis=1)
        state_solid_fraction = state_solid_count / size

        ax_phase_frac = plt.subplot2grid((4,2),(0, 1), rowspan=4, colspan=4)
        ax_phase_frac.plot(state_solid_fraction)

    fig.tight_layout()


def display_growth_2D(state_history):
    fig = plt.figure(figsize=(14,7))
    ax_instant = plt.subplot2grid((1,2),(0, 0))
    ax_instant.imshow(np.sum(state_history, axis=0), cmap=plt.get_cmap('Blues_r'))
    ax_instant.set_xlabel("Nucleation and Growth of a Solid in Liquid", labelpad=20)
    ax_instant.set_xticks([])
    ax_instant.set_yticks([])

    size, _ = state_history[0].shape
    state_solid_count = np.array([np.sum(state) for state in state_history])
    state_solid_fraction = state_solid_count / size**2
    ax_phase_frac = plt.subplot2grid((1,2),(0, 1))
    ax_phase_frac.plot(state_solid_fraction*100)
    ax_phase_frac.set_xlabel("Simulation Steps")
    ax_phase_frac.set_ylabel("Percentage of Solid")

    fig.tight_layout(w_pad=2)



def animate_growth(state_history):

    state_history = np.array(state_history)
    fig = plt.figure(figsize=(14,7))

    ax_instant = plt.subplot2grid((4,2),(0, 0))
    img_instant = ax_instant.imshow(state_history[0][np.newaxis])


    blank_img = np.zeros_like(state_history)
    frames = []
    n_timesteps, _ = state_history.shape
    for i in range(n_timesteps):
        blank_img[i] = state_history[i]
        frames.append(copy.deepcopy(blank_img))

    ax_overall = plt.subplot2grid((4,2),(1, 0), rowspan=3)
    img_overall = ax_overall.imshow(frames[0])


    size = len(state_history[0])
    state_solid_count = np.sum(array(state_history), axis=1)
    state_solid_fraction = state_solid_count / size

    ax_phase_frac = plt.subplot2grid((4,2),(0, 1), rowspan=4, colspan=4)
    ax_phase_frac.set_xlim([0, n_timesteps])
    line_phase_frac, = ax_phase_frac.plot([],[])

    fig.tight_layout()

    def draw_frame(frame_num):
        img_instant.set_array(state_history[frame_num][np.newaxis])
        img_overall.set_array(frames[frame_num])
        line_phase_frac.set_data(list(range(frame_num)), state_solid_fraction[:frame_num])

        return img_instant, img_overall

    ani = FuncAnimation(fig, draw_frame, interval=10)
    ani.save('test_anim.gif')


def set_nucleation_sites(init_state, number_of_sites):

    size_y, size_x = init_state.shape
    for i in range(number_of_sites):
        rand_x = randrange(0, size_x)
        rand_y = randrange(0, size_y)
        init_state[rand_y, rand_x] = 1

    return init_state
