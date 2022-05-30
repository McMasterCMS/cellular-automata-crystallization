
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


def simulate_growth_2D(size):

    state = np.zeros((size, size))
    state = set_nucleation_sites(state, 100)
    # Nucleation site
    # state[size//2][size//5] = solid
    # state[size//3][size//6] = solid
    # state[size//7][size//2] = solid
    # Copy state to keep track of history
    temp_state = copy.deepcopy(state)
    state_history = []
    # Loop over each cell
    while np.sum(state) < size*size:
        for x in range(size):
            for y in range(size):
                if state[y,x] == 1:
                        temp_state[y-1:y+2, x-1:x+2] = 1

        state = copy.deepcopy(temp_state)
        state_history.append(copy.deepcopy(temp_state))
    return state_history


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


def animate_growth_2D(state_history):

    # state_history.insert(0, np.zeros_like(state_history[0]))
    state_history = np.array(state_history)
    fig = plt.figure(figsize=(1080/100,1080/100))


    # This is for the two subplots
    # ax_overall = plt.subplot2grid((1,2),(0, 1), rowspan=2, colspan=1)
    # img_overall = ax_overall.imshow(state_history[0], animated=True, cmap=plt.get_cmap('Blues_r'), vmin=0, vmax=state_history.shape[0])
    # ax_overall.axis('off')

    ax_overall = plt.subplot2grid((1,1),(0, 0), rowspan=2, colspan=1)
    img_overall = ax_overall.imshow(state_history[0], animated=True, cmap=plt.get_cmap('Blues_r'), vmin=0, vmax=state_history.shape[0])
    ax_overall.axis('off')

    size = state_history[0].size
    state_solid_count = np.array([np.sum(i) for i in state_history])
    state_solid_fraction = state_solid_count / size

    # n_timesteps, _, _ = state_history.shape
    # ax_phase_frac = plt.subplot2grid((1,2),(0, 0), rowspan=2, colspan=1)
    # ax_phase_frac.set_ylim([0,1])
    # ax_phase_frac.set_xlim([0,n_timesteps])
    # ax_phase_frac.tick_params(
    #     axis='both',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     bottom=False,      # ticks along the bottom edge are off
    #     top=False,         # ticks along the top edge are off
    #     labelbottom=False,
    #     left=False,
    #     right=False,
    #     labelleft=False) # labels along the bottom edge are off
    # ax_phase_frac.set_ylabel('Amount Solidified', size=20)
    # ax_phase_frac.set_xlabel('Time', size=20)
    # line_phase_frac, = ax_phase_frac.plot([],[])

    fig.tight_layout()

    def draw_frame(frame_num):
        img_overall.set_array(np.sum(state_history[:frame_num], axis=0))
        # line_phase_frac.set_data(list(range(frame_num)), state_solid_fraction[:frame_num])


    ani = FuncAnimation(fig, draw_frame, interval=1000/30, frames=state_history.shape[0])
    ani.save('test_anim_2D_1.gif')
    plt.show()


def set_nucleation_sites(init_state, number_of_sites):

    size_y, size_x = init_state.shape
    for i in range(number_of_sites):
        rand_x = randrange(0, size_x)
        rand_y = randrange(0, size_y)
        init_state[rand_y, rand_x] = 1

    return init_state


if __name__ == "__main__":
    state_history = simulate_growth_2D(300)
    animate_growth_2D(state_history)
