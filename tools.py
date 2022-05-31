import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from matplotlib.animation import FuncAnimation
import copy
from random import randrange
from IPython.display import Image # For GIFs


liq = 0
sol = 1


def set_nucleation_sites(init_state, number_of_sites):
    """Choose a number of random cells to nuleate from and return the
    nucleated state"""

    size_y, size_x = init_state.shape
    for i in range(number_of_sites):
        rand_x = randrange(0, size_x)
        rand_y = randrange(0, size_y)
        init_state[rand_y, rand_x] = sol

    return init_state


def display_state(state):
    """Show resulting image plot"""
    
    state = np.array(state)
    plt.figure(figsize=(20,20))
    plt.imshow(state[np.newaxis])
    format_imshow(plt.gca())


def format_imshow(ax):
    """Removes tickmarks from image, preserving black outline and
    changes colour sheme to blue"""

    plt.set_cmap('Blues_r')
    plt.xticks([])
    plt.xticks([], minor=True)
    plt.yticks([])
    plt.yticks([], minor=True)


def animate_growth(prev_states):
    """Animate simulation by showing three different graphics:
    the state of the simulation at an instant (i.e. 1D row of cells),
    the state of the simulation with time as the vertical axis (2D image
    of cells), and extent of solification at a given timestep (a line
    of amount solidified vs. time). A gif of the animation is saved as
    growth_anim.gif"""

    divs=11

    prev_states = np.array(prev_states)
    fig = plt.figure(figsize=(14,7))

    ax_instant = plt.subplot2grid((divs,2),(0, 0))
    format_imshow(ax_instant)
    state_img = prev_states[0][np.newaxis]
    img_instant = ax_instant.imshow(state_img, aspect='equal')

    blank_img = np.zeros_like(prev_states)
    frames = []
    n_timesteps, _ = prev_states.shape
    for i in range(n_timesteps):
        blank_img[i] = prev_states[i]
        frames.append(copy.deepcopy(blank_img))

    ax_overall = plt.subplot2grid((divs,2),(1, 0), rowspan=divs-1)
    format_imshow(ax_overall)
    img_overall = ax_overall.imshow(frames[0], aspect='auto')

    size = len(prev_states[0])
    state_solid_count = np.sum(array(prev_states), axis=1)
    state_solid_fraction = state_solid_count / size

    ax_phase_frac = plt.subplot2grid((divs,2),(0, 1), rowspan=divs)
    ax_phase_frac.set_xlim([0, n_timesteps])
    line_phase_frac, = ax_phase_frac.plot([],[])
    ax_phase_frac.set_xlabel('Timestep')
    ax_phase_frac.set_ylabel('Fraction Solified')
    ax_phase_frac.set_ylim([0, 1.05])

    fig.tight_layout()

    def draw_frame(frame_num):
        img_instant.set_array(prev_states[frame_num][np.newaxis])
        img_overall.set_array(frames[frame_num])
        line_phase_frac.set_data(list(range(frame_num)), state_solid_fraction[:frame_num])

        return img_instant, img_overall

    ani = FuncAnimation(fig, draw_frame, interval=10)
    ani.save('growth_anim.gif')
    plt.close()


def animate_growth_2D(prev_states):

    prev_states = np.array(prev_states)
    fig = plt.figure(figsize=(15,7))

    ax_overall = plt.subplot2grid((1,2),(0, 0))
    img_overall = ax_overall.imshow(prev_states[0], animated=True, cmap=plt.get_cmap('Blues_r'), vmin=0, vmax=prev_states.shape[0])
    ax_overall.axis('off')

    ax_phase_frac = plt.subplot2grid((1,2),(0, 1))
    ax_phase_frac.set_aspect('auto')
    size = prev_states[0].size
    state_solid_count = np.array([np.sum(i) for i in prev_states])
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
    ax_phase_frac.set_ylabel('Amount Solidified', size=20)
    ax_phase_frac.set_xlabel('Time', size=20)
    line_phase_frac, = ax_phase_frac.plot([],[])
    ax_phase_frac.set_ylim([0, 1.05])
    ax_phase_frac.set_xlim([0, prev_states.shape[0]])

    fig.tight_layout()

    def draw_frame(frame_num):
        img_overall.set_array(np.sum(prev_states[:frame_num], axis=0))
        line_phase_frac.set_data(list(range(frame_num)), state_solid_fraction[:frame_num])


    ani = FuncAnimation(fig, draw_frame, interval=10, frames=prev_states.shape[0])
    ani.save('test_anim_2D_1.gif')
    plt.close()


if __name__ == "__main__":

    def simulate_growth_2D(size):

        state = np.zeros((size, size))
        state = set_nucleation_sites(state, 100)
        # Copy state to keep track of history
        temp_state = copy.deepcopy(state)
        prev_states = []
        # Loop over each cell
        while np.sum(state) < size*size:
            for x in range(size):
                for y in range(size):
                    if state[y,x] == sol:
                            temp_state[y-1:y+2, x-1:x+2] = sol

            state = copy.deepcopy(temp_state)
            prev_states.append(copy.deepcopy(temp_state))
        return prev_states

    def simulate_growth(size):
        state = [liq] * size
        state = [liq] * size
        state[size//2] = sol
        state[size//10+1] = sol
        temp_state = copy.deepcopy(state)
        prev_states = [state]
        for dt in range(size):
            for cell in range(size):
                if cell == 0:
                    if state[cell+1] == sol:
                        temp_state[cell] = sol
                elif cell == size-1:
                    if state[cell-1] == sol:
                        temp_state[cell] = sol
                elif state[cell-1] == sol or state[cell+1] == sol:
                    temp_state[cell] = sol
            state = copy.deepcopy(temp_state)
            prev_states.append(state)
        return prev_states

    prev_states = simulate_growth_2D(200)
    animate_growth_2D(prev_states)