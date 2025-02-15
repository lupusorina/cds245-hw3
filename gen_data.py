import os
import ast

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import seed
import control
from pendulum import PendulumEnv
import time

seed(42)

class NonlinearController:
    def __init__(self, mass, rod_length, gravity, action_limits, lambda_=10, k=1):
        self.m = mass
        self.l = rod_length
        self.g = gravity
        self.lambda_ = lambda_
        self.k = k
        self.action_limits = action_limits

    def controller(self, angle, angle_d, angular_vel, angular_vel_d, g, l):
        angle_err = angle - angle_d
        angular_vel_ref = angular_vel_d + self.lambda_ * angle_err 
        s = angular_vel - angular_vel_ref
        u = - self.k * s - s * g  * l/2 * np.sin(angle)
        u = np.clip(u, a_min=self.action_limits[0], a_max=self.action_limits[1])
        return u

class LQRController:
    def __init__(self,
                 mass: float,
                 rod_length: float,
                 gravity: float,
                 dt: float,
                 action_limits=(-2.0, 2.0),
                 Q = np.diag([20, 0.5]),
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
        
        self.Q = Q
        self.R = R
        self.I = self.m * (self.l ** 2) * 1/3
        self.action_limits = action_limits

        self.A = np.array([[0, 1], 
                           [3 * self.g / (self.l * 2), 0.0]])
        self.B = np.array([[0], [1 / self.I]])

        self.A_d = np.eye(2) + dt * self.A
        self.B_d = dt * self.B 
        self.K = control.lqr(self.A, self.B, self.Q, self.R)[0]

    def compute_control(self, state: np.ndarray, state_d: np.ndarray = np.array([0, 0])):
        """
        Compute the control action using the LQR feedback law.
        """
        u = - self.K @ (state - state_d)
        return np.clip(u, a_min=self.action_limits[0], a_max=self.action_limits[1])

class EnergyShapingController:
    def __init__(self, mass, rod_length, gravity, dt, action_limits=(-2.0, 2.0)):
        self.m = mass
        self.l = rod_length
        self.g = gravity
        self.I = self.m * (self.l ** 2) * 1/3
        self.dt = dt
        self.action_limits = action_limits

    def compute_total_energy(self, state):
        kinetic_energy = 1/2 * self.I * (state[1] ** 2)
        potential_energy = self.m * self.g * self.l/2 * np.cos(state[0])
        E = kinetic_energy + potential_energy
        return kinetic_energy, potential_energy, E

    def compute_desired_energy(self):
        E_d = self.m * self.g * self.l/2
        return E_d

    def get_action(self, state):
        E_d = self.compute_desired_energy()
        _, _, E = self.compute_total_energy(state)

        E_err = E - E_d # Energy error.
        u = - state[1] * E_err
        action = np.clip(u, a_min=self.action_limits[0], a_max=self.action_limits[1])
        return action


if __name__ == "__main__": 
    N_EPISODES = 5000
    ANGLE_SWITCH_THRESHOLD_DEG = 10 # deg
    EPISODE_DONE_ANGLE_THRESHOLD_DEG = 0.1 # deg
    GRAVITY = 10.0
    DT = 0.01

    env = PendulumEnv(dt=DT, g=GRAVITY) # render_mode = 'human'
    pendulum_params = {"mass": env.m,
                       "rod_length": env.l,
                       "gravity": GRAVITY,
                       "action_limits": (env.action_space.low, env.action_space.high),
                       'dt': DT}

    energy_controller = EnergyShapingController(**pendulum_params)
    # nonlinear_controller = NonlinearController(**pendulum_params)
    lqr_controller = LQRController(**pendulum_params)

    duration_episodes = []
    steps_per_episodes = []
    list_of_all_the_data = []
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

            obs ,reward ,_ ,_, _ = env.step(action.reshape(1, -1))
            cumreward += reward

            if abs(angle) < np.deg2rad(EPISODE_DONE_ANGLE_THRESHOLD_DEG):
                upright_angle_buffer.append(angle)
            if len(upright_angle_buffer) > 40:
                done = True

            list_of_all_the_data.append([i,
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

    print('It took total of', sum(duration_episodes), 'seconds to run', N_EPISODES, 'episodes')

    col_names = ['episode', 'actions', 'states', 'reward', 'cumreward', 'ctrl_type', 'done']
    df = pd.DataFrame(list_of_all_the_data, columns=col_names)

    env.close()

    DATA_FOLDER = 'Data/CSVs'
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    PLOTS_FOLDER = 'Data/Plots'
    if not os.path.exists(PLOTS_FOLDER):
        os.makedirs(PLOTS_FOLDER)

    FILE_NAME = 'data_pendulum_' + str(N_EPISODES) + '.csv'
    df.to_csv(f'{DATA_FOLDER}/{FILE_NAME}', index=False)

    # Plot state.
    PLOT = True
    print('Plotting...')
    if PLOT == False:
        exit()

    NB_EPISODES_TO_PLOT = 100

    data = pd.read_csv(f'{DATA_FOLDER}/{FILE_NAME}')
    data['states'] = data['states'].apply(lambda x: ast.literal_eval(x))
    data['angle_state'] = data['states'].apply(lambda x: np.arctan2(x[1], x[0]))
    N_EPISODES = data['episode'].nunique()

    for idx_episode in range(min(N_EPISODES, NB_EPISODES_TO_PLOT)):
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
        print(f'Episode {idx_episode}')
        episode_data = data[data['episode'] == idx_episode]

        angles = episode_data['angle_state']
        angular_vels = episode_data['states'].apply(lambda x: x[2])
        actions = episode_data['actions']

        ax[0].plot(angles, label=f'Episode {idx_episode}')
        ax[1].plot(angular_vels, label=f'Episode {idx_episode}')
        ax[2].plot(actions, label=f'Episode {idx_episode}')

        for i in range(3):
            ax[i].grid()
            ax[i].legend(loc='upper right')

        ax[0].set_title(f"Performance for Episode {idx_episode}")
        ax[0].set_ylabel("Angle [rad]")
        ax[1].set_ylabel("Angular Velocity [rad/s]")
        ax[2].set_ylabel("Torque [Nm]")
        plt.savefig(f'{PLOTS_FOLDER}/episode_{idx_episode}.png')
        plt.close()