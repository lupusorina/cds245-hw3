import pandas as pd

nb_episodes = 3
duration = 10

data = []
for i in range(3):
    for j in range(duration):
        data.append([i, j, [j + (i+1) * 100,
                            j + (i+1) * 100,
                            j + (i+1) * 100]])
        
# save data to csv
col_names = ['episode', 'actions', 'states']
df = pd.DataFrame(data, columns=col_names)
df.to_csv('Data/CSVs/test_data.csv', index=False)

    
