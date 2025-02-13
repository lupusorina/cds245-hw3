import pandas as pd
import numpy as np
nb_episodes = 3
duration = 10

data = []
for i in range(3):
    for j in range(duration):
        action = np.array([j])
        state = np.array([j + (i+1) * 100, j + (i+1) * 100, j + (i+1) * 100])
        data.append([i, action, state.tolist()])
        
# save data to csv
col_names = ['episode', 'actions', 'states']
df = pd.DataFrame(data, columns=col_names)
df.to_csv('Data/CSVs/test_data.csv', index=False)

    
