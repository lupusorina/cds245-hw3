import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data.
df = pd.read_csv('Data_Racing/las_vegas.txt', delimiter='\t')
print(df.columns)

# Save to csv.
df.to_csv('Data_Racing/las_vegas.csv', index=False)

# Remove invalid data.
df = df[df['lap_time_invalid'] == 0]

# Plot the trajectory.
plt.figure()
plt.plot(df['world_position_X'], df['world_position_Y'])
plt.xlabel('world_position_X')
plt.ylabel('world_position_Y')

# Extract relevant columns.
state_cols = ['velocity_X', 'velocity_Y',
              'gforce_X', 'gforce_Y', 'gforce_Z',
              'tyre_angular_vel_0', 'tyre_angular_vel_1',
              'tyre_angular_vel_2', 'tyre_angular_vel_3']
u_cols = ['steering', 'throttle', 'brake']

FILE_NAME = 'LAS_VEGAS_DATA'
FOLDER = 'Data/CSVs'

data_states = np.array(df[state_cols].values)
data_u = df[u_cols].values
data_episode_nb = [0] * len(data_states)

# Save to csv in the format for the BC.
list_of_all_the_data = []
for states, u, episodes in zip(data_states, data_u, data_episode_nb):
    list_of_all_the_data.append([episodes, states.tolist(), u.tolist()])

col_names = ['episode', 'states', 'actions']
df_curated = pd.DataFrame(list_of_all_the_data, columns=col_names)
df_curated.to_csv(f'{FOLDER}/{FILE_NAME}.csv', index=False)


