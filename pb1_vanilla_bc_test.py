import torch
import numpy as np
from pb1_vanilla_bc import BCModel
from pendulum import PendulumEnv
import matplotlib.pyplot as plt
from gen_data import EnergyShapingController, NonlinearController, LQRController
GRAVITY = 10.0
DT = 0.01
EPISODE_DONE_ANGLE_THRESHOLD_DEG = 0.2 # deg

env = PendulumEnv(dt=DT, g=GRAVITY, render_mode = 'human')
pendulum_params = {"mass": env.m,
                    "rod_length": env.l,
                    "gravity": GRAVITY,
                    "action_limits": (env.action_space.low, env.action_space.high),
                    'dt': DT}

# Load model
model = BCModel()
state_dict = torch.load('Models/bc_model_pb1.pth', weights_only=True)
model.load_state_dict(state_dict)

# Set to evaluation mode
model.eval()
print('Model loaded')

# Test model with NN controller
theta0 = np.pi/2
theta_dot0 = 0.0
options = {'theta0': theta0,
           'theta_dot0': theta_dot0}

obs, _ = env.reset(options=options)
obs_tensor = torch.tensor(obs, dtype=torch.float32)
done = False
upright_angle_buffer = []
action_list_NN = []
while not done:
    action = model(obs_tensor).item()
    out = env.step([action])
    obs = out[0]
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    angle = np.arctan2(obs[1], obs[0])
    if abs(angle) < np.deg2rad(EPISODE_DONE_ANGLE_THRESHOLD_DEG):
        upright_angle_buffer.append(angle)

    if len(upright_angle_buffer) > 40:
        done = True
    action_list_NN.append(action)
    
print('Classical controller')

env = PendulumEnv(dt=DT, g=GRAVITY, render_mode = 'human')
obs, _ = env.reset(options=options)
done = False
energy_controller = EnergyShapingController(**pendulum_params)
# nonlinear_controller = NonlinearController(**pendulum_params)
lqr_controller = LQRController(**pendulum_params)
ANGLE_SWITCH_THRESHOLD_DEG = 10 # deg
action_classical_control = []
upright_angle_buffer = []
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
    if len(upright_angle_buffer) > 40:
        done = True
    action_classical_control.append(action.squeeze())

print('Done')

plt.plot(action_list_NN, color='r', label='NN')
plt.plot(action_classical_control, color='b', linestyle='dashed', label='Classical')
plt.legend()
plt.show()