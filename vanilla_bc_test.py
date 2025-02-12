import torch
import numpy as np
from pb1_vanilla_bc import BCModel
from pendulum import PendulumEnv
import matplotlib.pyplot as plt
import os
from gen_data import EnergyShapingController, NonlinearController, LQRController

def run_environment_NN(env, model, obs_list):
    u_list = []
    done = False
    obs_tensor = torch.tensor(np.array(obs_list).flatten(), dtype=torch.float32)
    upright_angle_buffer = []
    angle_list = []
    ang_vel_list = []
    while not done:
        action = model(obs_tensor)
        obs, _, _, _, _ = env.step([action[0].item()])

        angle = np.arctan2(obs[1], obs[0])
        if abs(angle) < np.deg2rad(EPISODE_DONE_ANGLE_THRESHOLD_DEG):
            upright_angle_buffer.append(angle)

        if len(upright_angle_buffer) > UPRIGHT_BUFFER_LEN:
            done = True

        # Save data.
        angle_list.append(angle)
        ang_vel_list.append(obs[2])
        u_list.append(action[0].item())
        obs_list.pop(0)
        obs_list.append(obs)
        obs_tensor = torch.tensor(np.array(obs_list).flatten(), dtype=torch.float32)
    return u_list, angle_list, ang_vel_list


# Constants.
GRAVITY = 10.0
DT = 0.01
EPISODE_DONE_ANGLE_THRESHOLD_DEG = 0.25 # deg
FOLDER_PLOTS = 'Plots'
UPRIGHT_BUFFER_LEN = 40
if not os.path.exists(FOLDER_PLOTS):
    os.makedirs(FOLDER_PLOTS)

# Environment.
env = PendulumEnv(dt=DT, g=GRAVITY) #, render_mode = 'human')
pendulum_params = {"mass": env.m,
                    "rod_length": env.l,
                    "gravity": GRAVITY,
                    "action_limits": (env.action_space.low, env.action_space.high),
                    'dt': DT}

# Initial conditions.
theta0 = np.pi/2
theta_dot0 = 2.0
options = {'theta0': theta0,
           'theta_dot0': theta_dot0}

# Test 1: Imitation Learning. STATE_HORIZON=1, ACTION_HORIZON=1.
STATE_HORIZON = 1
ACTION_HORIZON = 1
SEQ_STATE_SIZE = 3 * STATE_HORIZON
SEQ_ACTION_SIZE = 1 * ACTION_HORIZON
model = BCModel(SEQ_STATE_SIZE, SEQ_ACTION_SIZE) # Default size: 3 (state), 1 (control)
state_dict = torch.load('Models/bc_model_pb1a.pth', weights_only=True)
model.load_state_dict(state_dict)
model.eval()
obs, _ = env.reset(options=options)
obs_list = []
obs_list.append(obs)
u_list_BC, angle_list_BC, ang_vel_list_BC = run_environment_NN(env, model, obs_list)


# Test 2: Imitation Learning. STATE_HORIZON=3, ACTION_HORIZON=3.
STATE_HORIZON = 3
ACTION_HORIZON = 3
SEQ_STATE_SIZE = 3 * STATE_HORIZON
SEQ_ACTION_SIZE = 1 * ACTION_HORIZON
model = BCModel(SEQ_STATE_SIZE, SEQ_ACTION_SIZE)
state_dict = torch.load('Models/bc_model_pb1b.pth', weights_only=True)
model.load_state_dict(state_dict)
model.eval()
obs, _ = env.reset(options=options)
obs_list = []
for i in range(STATE_HORIZON):
    obs_list.append(obs)
u_list_BC_traj, angle_list_BC_traj, ang_vel_list_BC_traj = run_environment_NN(env, model, obs_list)

# Test 3: Classical controller.
obs, _ = env.reset(options=options)
obs = obs.reshape(len(obs), 1)
energy_controller = EnergyShapingController(**pendulum_params)
lqr_controller = LQRController(**pendulum_params)

ANGLE_SWITCH_THRESHOLD_DEG = 10 # deg
action_classical_control = []
upright_angle_buffer = []
angle_list_classical_ctrl = []
angular_velocity_list_classical_ctrl = []
done = False
while not done:
    angle = np.arctan2(obs[1], obs[0])
    pos_vel = np.array([angle, obs[2]]).squeeze()
    if abs(angle) < np.deg2rad(ANGLE_SWITCH_THRESHOLD_DEG):
        action = lqr_controller.compute_control(pos_vel)
        ctrl_type = 'LQR'
    else:
        action = energy_controller.get_action(pos_vel)
        ctrl_type = 'EnergyShaping'

    prev_action = action.copy()
    obs ,_ ,_ ,_, _ = env.step(action.reshape(1, -1))

    if abs(angle) < np.deg2rad(EPISODE_DONE_ANGLE_THRESHOLD_DEG):
        upright_angle_buffer.append(angle)
    if len(upright_angle_buffer) > UPRIGHT_BUFFER_LEN:
        done = True

    action_classical_control.append(action.squeeze())
    angle_list_classical_ctrl.append(angle)
    angular_velocity_list_classical_ctrl.append(obs[2])

# Plots.
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 8))

ax[2].plot(u_list_BC, color='r', label='NN, Vanilla')
ax[2].plot(u_list_BC_traj, color='g', label='NN, Trajectory')
ax[2].plot(action_classical_control, color='b', linestyle='dashed', label='Classical Controller')

ax[0].plot(angle_list_BC, color='r', label='NN, Vanilla')
ax[0].plot(angle_list_BC_traj, color='g', label='NN, Trajectory')
ax[0].plot(angle_list_classical_ctrl, color='b', label='Classical Controller')

ax[1].plot(ang_vel_list_BC, color='r', label='NN, Vanilla')
ax[1].plot(ang_vel_list_BC_traj, color='g', label='NN, Trajectory')
ax[1].plot(angular_velocity_list_classical_ctrl, color='b', label='Classical Controller')

for i in range(3):
    ax[i].grid()
    ax[i].legend(loc='upper right')

ax[0].set_ylabel("Angle [rad]")
ax[1].set_ylabel("Angular Velocity [rad/s]")
ax[2].set_ylabel("Torque [Nm]")
ax[2].set_xlabel("Time Step")
plt.savefig(f'{FOLDER_PLOTS}/comparison.png')
plt.show()