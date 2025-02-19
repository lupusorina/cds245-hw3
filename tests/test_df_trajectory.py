import pandas as pd
import numpy as np

DURATION = 10
NB_EPISODES = 3

data = []
for i in range(NB_EPISODES):
    for j in range(DURATION):
        action = np.array([j])
        state = np.array([j + (i+1) * 100, j + (i+1) * 100, j + (i+1) * 100])
        data.append([i, action, state.tolist()])

# Save data to csv.
col_names = ['episode', 'actions', 'states']
df = pd.DataFrame(data, columns=col_names)
df.to_csv('Data/CSVs/test_data.csv', index=False)

    
