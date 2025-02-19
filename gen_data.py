import os
import ast

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import seed
import control
import time
import gymnasium as gym

seed(42)

class LQRController:
    def __init__(self,
                 mass: float,
                 rod_length: float,
                 gravity: float,
                 dt: float,
                 action_limits=(-2.0, 2.0),
                 Q = np.diag([50, 0.5]),
                 R = np.array([[0.2]])):
        """
        LQR Controller for the Pendulum environment.
        Parameters:
            mass: mass of the pendulum
            rod_length: length of the pendulum
            gravity: gravity
            action_limits: limits of the action space
            dt: time step
            Q: state cost matrix
            R: control effort cost
        """
        self.m = mass
        self.l = rod_length
        self.g = gravity
        self.I = self.m * (self.l ** 2) * 1/3
        self.action_limits = action_limits
        self.dt = dt

        # LQR matrices.
        self.Q = Q
        self.R = R

        # Dynamics.
        self.A = np.array([[0, 1], 
                           [3 * self.g / (self.l * 2), 0.0]])
        self.B = np.array([[0], [1 / self.I]])
        # Discrete dynamics.
        self.A_d = np.eye(2) + dt * self.A
        self.B_d = dt * self.B 
        self.K = control.lqr(self.A, self.B, self.Q, self.R)[0]

    def compute_control(self,
                        state: np.ndarray,
                        state_d: np.ndarray = np.array([0, 0])):
        u = - self.K @ (state - state_d)
        return np.clip(u, a_min=self.action_limits[0], a_max=self.action_limits[1])

class EnergyShapingController:
    def __init__(self,
                 mass: float,
                 rod_length: float,
                 gravity: float,
                 dt: float,
                 action_limits: tuple = (-2.0, 2.0)):
        """
        Energy Shaping Controller for the Pendulum environment.
        See https://underactuated.mit.edu/acrobot.html#section6
        Parameters:
            mass: mass of the pendulum
            rod_length: length of the pendulum
            gravity: gravity
            action_limits: limits of the action space
            dt: time step
        """
        self.m = mass
        self.l = rod_length
        self.g = gravity
        self.I = self.m * (self.l ** 2) * 1/3
        self.dt = dt
        self.action_limits = action_limits

    def compute_total_energy(self,
                            state: np.ndarray):
        kinetic_energy = 1/2 * self.I * (state[1] ** 2)
        potential_energy = self.m * self.g * self.l/2 * np.cos(state[0])
        E = kinetic_energy + potential_energy
        return kinetic_energy, potential_energy, E

    def compute_desired_energy(self):
        E_d = self.m * self.g * self.l/2
        return E_d

    def get_action(self,
                   state: np.ndarray):
        E_d = self.compute_desired_energy()
        _, _, E = self.compute_total_energy(state)
        E_err = E - E_d # Energy error.
        u = - state[1] * E_err
        return np.clip(u, a_min=self.action_limits[0], a_max=self.action_limits[1])


if __name__ == "__main__": 
    N_EPISODES = 5000
    ANGLE_SWITCH_THRESHOLD_DEG = 18 # deg
    EPISODE_DONE_ANGLE_THRESHOLD_DEG = 0.5 # deg
    GRAVITY = 10.0

    env = gym.make("Pendulum-v1") #, render_mode = 'human')
    pendulum_params = {"mass": env.unwrapped.m,
                       "rod_length": env.unwrapped.l,
                       "gravity": GRAVITY,
                       "action_limits": (env.action_space.low, env.action_space.high),
                       'dt': env.unwrapped.dt}

    # Set up controllers.
    energy_controller = EnergyShapingController(**pendulum_params)
    lqr_controller = LQRController(**pendulum_params)

    duration_episodes = []
    data_list = []
    for i in range(N_EPISODES):
        print(f'Episode {i}')
        obs, _ = env.reset(options={'x_init': 1.0, 'y_init': 8.0})
        done = False
        state = obs.squeeze().copy() 
        upright_angle_buffer = []
        ctrl_type = None
        time_start_episode = time.time()
        counter = 0
        cumreward = 0
        while not done:
            angle = np.arctan2(obs[1], obs[0])
            pos_vel = np.array([angle, obs[2]]).squeeze()

            if abs(angle) < np.deg2rad(ANGLE_SWITCH_THRESHOLD_DEG):
                action = lqr_controller.compute_control(pos_vel)
                ctrl_type = 'LQR'
            else:
                action = energy_controller.get_action(pos_vel)
                ctrl_type = 'EnergyShaping'

            obs ,reward ,_ ,_, _ = env.step(action)
            cumreward += reward

            if abs(angle) < np.deg2rad(EPISODE_DONE_ANGLE_THRESHOLD_DEG):
                upright_angle_buffer.append(angle)
            if len(upright_angle_buffer) > 10:
                done = True

            data_list.append([i,
                            action,
                            state.tolist(),
                            reward,
                            cumreward,
                            ctrl_type,
                            int(done)])
            state = obs.squeeze().copy() # use .copy() for arrays because of the shared memory issues
            counter += 1

        time_end_episode = time.time()
        duration_episodes.append(time_end_episode - time_start_episode)
    env.close()

    print('It took total of', sum(duration_episodes), 'seconds to run', N_EPISODES, 'episodes')

    print("Saving data...")
    col_names = ['episode', 'actions', 'states', 'reward', 'cumreward', 'ctrl_type', 'done']
    df = pd.DataFrame(data_list, columns=col_names)
    DATA_FOLDER = 'Data/CSVs'
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    PLOTS_FOLDER = 'Data/Plots'
    if not os.path.exists(PLOTS_FOLDER):
        os.makedirs(PLOTS_FOLDER)

    FILE_NAME = 'data_pendulum_' + str(N_EPISODES) + '.csv'
    df.to_csv(f'{DATA_FOLDER}/{FILE_NAME}', index=False)
    print("Data saved.")

    # Plot state.
    PLOT = True
    if PLOT == False:
        exit()
    print('Plotting...')
    NB_EPISODES_TO_PLOT = 100

    data = pd.read_csv(f'{DATA_FOLDER}/{FILE_NAME}')
    data['states'] = data['states'].apply(lambda x: ast.literal_eval(x))
    # data['actions'] = data['actions'].apply(lambda x: ast.literal_eval(x))

    data['angle_state'] = data['states'].apply(lambda x: np.arctan2(x[1], x[0]))
    N_EPISODES = data['episode'].nunique()

    for idx_episode in range(min(N_EPISODES, NB_EPISODES_TO_PLOT)):
        fig, ax = plt.subplots(5, 1, sharex=True, figsize=(8, 12))
        print(f'Episode {idx_episode}')
        episode_data = data[data['episode'] == idx_episode]

        angles = episode_data['angle_state'].values
        angular_vels = episode_data['states'].apply(lambda x: x[2]).values
        actions = episode_data['actions'].values
        rewards = episode_data['reward'].values
        cumrewards = episode_data['cumreward'].values
        control_type = episode_data['ctrl_type'].values
        time_ = np.arange(angles.shape[0])

        actions_cleaned = [] # hacky way to clean the actions
        for act in actions:
            # extract from '[' ']'
            act = act[1:-1]
            actions_cleaned.append(float(act))

        ax[0].plot(angles, label=f'Episode {idx_episode}')
        ax[1].plot(angular_vels, label=f'Episode {idx_episode}')
        ax[2].plot(actions_cleaned, label=f'Episode {idx_episode}')
        ax[3].plot(cumrewards, label=f'Episode {idx_episode}')
        ax[4].plot(rewards, label=f'Episode {idx_episode}')

        for i in range(5):
            ax[i].grid()
            ax[i].legend(loc='upper right')

        ax[0].set_title(f"Performance for Episode {idx_episode}")
        ax[0].set_ylabel("Angle [rad]")
        ax[1].set_ylabel("Angular Velocity [rad/s]")
        ax[2].set_ylabel("Torque [Nm]")
        ax[3].set_ylabel("Cumulative Reward")
        ax[4].set_ylabel("Reward")
        ax[4].set_xlabel("Time Step")
        plt.savefig(f'{PLOTS_FOLDER}/episode_{idx_episode}.png')

        plt.close()