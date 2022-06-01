import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from matplotlib.animation import FuncAnimation
import copy
from random import randrange


liq = 0
sol = 1
anim_size = (15, 9) # dimensions of animation


def set_nucleation_sites(init_state, number_of_sites):
    """Choose a number of random cells to nuleate from and return the
    nucleated state"""

    size_y, size_x = init_state.shape
    for i in range(number_of_sites):
        # Don't allow sites to be on edges to reduce the number of
        # edge case if-else statements in growth function
        rand_x = randrange(1, size_x)
        rand_y = randrange(1, size_y)
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

def format_line_plot(ax, n_frames):
    ax.set_xlabel('Timestep', size=20)
    ax.set_ylabel('Fraction Solidified', size=20)
    ax.set_ylim([0, 1.05])
    ax.set_xlim([0, n_frames])
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()


def animate_growth(prev_states):
    """Animate simulation by showing three different graphics:
    the state of the simulation at an instant (i.e. 1D row of cells),
    the state of the simulation with time as the vertical axis (2D image
    of cells), and extent of solification at a given timestep (a line
    of amount solidified vs. time). A gif of the animation is saved as
    simulation_anim_1D.gif"""

    prev_states = np.array(prev_states)
    fig = plt.figure(figsize=anim_size)

    divs=11 # number of vertical partitions for graphics
    n_frames, _ = prev_states.shape # Number of animation frames

    # Simulation at an instant animation initialization
    ax_sim_1D = plt.subplot2grid((divs,2),(0, 0))
    format_imshow(ax_sim_1D)
    sim_1D_frame_init = prev_states[0][np.newaxis]
    sim_1D_img = ax_sim_1D.imshow(sim_1D_frame_init,
                                  aspect='equal',
                                  animated=True,
                                  vmin=0,
                                  vmax=n_frames//10+1)

    # Overall simulation as a 2D plot of states (horizontal) and time
    # (vertical)

    # prev_states is a 2D image where every row is the next timestep
    # in the simulation, i.e. a static image showing how the simulation
    # progressed. To animate, a semi-empty image is created for each
    # frame in the animation. Rows are progressivley copied from
    # prev_state to make a collection of frames that 'reveal' the
    # simulation as the animation runs
    sim_2D_frames = []
    sim_2D_frame_empty = np.zeros_like(prev_states)
    for i in range(n_frames):
        sim_2D_frame_empty[i] = prev_states[i]
        sim_2D_frames.append(copy.deepcopy(sim_2D_frame_empty))
        

    ax_sim_2D = plt.subplot2grid((divs,2),(1, 0), rowspan=divs-1)
    format_imshow(ax_sim_2D)
    sim_2D_frame_init = sim_2D_frames[0]
    # sim_2D_img = ax_sim_2D.imshow(sim_2D_frame_init, aspect='auto')
    sim_2D_img = ax_sim_2D.imshow(sim_2D_frame_init, animated=True,
                                  vmin=0, vmax=n_frames//10+1)

    # Create line plot of the amount solidified
    size = len(prev_states[0])
    state_sol_count = np.sum(array(prev_states), axis=1)
    state_sol_fraction = state_sol_count / size

    ax_sol_frac = plt.subplot2grid((divs,2),(0, 1), rowspan=divs)
    sol_frac_line, = ax_sol_frac.plot([],[])
    format_line_plot(ax_sol_frac, n_frames)

    fig.tight_layout()

    def draw_frame(frame_num):
        """Guides how animation is drawn based on the frame number"""
        # sim_1D_img.set_array(prev_states[frame_num][np.newaxis])
        sim_1D_img.set_array(np.sum(prev_states[:frame_num][np.newaxis], axis=1))
        # sim_2D_img.set_array(sim_2D_frames[frame_num])
        sim_2D_img.set_array(np.sum(np.array(sim_2D_frames)[:frame_num], axis=0))
        sol_frac_line.set_data(list(range(frame_num)), 
                               state_sol_fraction[:frame_num])

    # Create animation and save
    ani = FuncAnimation(fig, draw_frame, interval=10)
    ani.save('simulation_anim_1D.gif')
    plt.close()


def animate_growth_2D(prev_states):
    """Animate simulation by showing 2 different graphics: 1. the state
    of the simulation at an instant (2D image) where the lighter the
    shade the blue the 'older' the point of soldification and 2.
    the extent of solification at a given timestep (a line of amount
    solidified vs. time). A gif of the animation is saved as
    simulation_anim_2D.gif"""

    prev_states = np.array(prev_states)
    fig = plt.figure(figsize=anim_size)

    n_frames, _, _ = prev_states.shape

    ax_sim_2D = plt.subplot2grid((1,2),(0, 0))
    format_imshow(ax_sim_2D)
    sim_2D_img = ax_sim_2D.imshow(prev_states[0], animated=True,
                                  vmin=0, vmax=n_frames)

    ax_sol_frac = plt.subplot2grid((1,2),(0, 1))
    ax_sol_frac.set_aspect('auto')
    size = prev_states[0].size
    state_solid_count = np.array([np.sum(i) for i in prev_states])
    state_solid_fraction = state_solid_count / size

    sol_frac_line, = ax_sol_frac.plot([],[])
    format_line_plot(ax_sol_frac, n_frames)

    fig.tight_layout()

    def draw_frame(frame_num):
        sim_2D_img.set_array(np.sum(prev_states[:frame_num], axis=0))
        sol_frac_line.set_data(list(range(frame_num)), state_solid_fraction[:frame_num])


    ani = FuncAnimation(fig, draw_frame, interval=10, frames=prev_states.shape[0])
    ani.save('simulation_anim_2D.gif')
    plt.close()


if __name__ == "__main__":

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

    # prev_states = simulate_growth_2D(200)
    # animate_growth_2D(prev_states)

    prev_states = simulate_growth(100)
    animate_growth(prev_states)