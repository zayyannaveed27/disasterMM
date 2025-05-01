import numpy as np

# Load the npz file
data = np.load('/data/wildfirets/new/compressed/705_2023-11-27_2023-12-26.npz', allow_pickle=True)

# List the keys stored in the file (e.g., 'data' and 'dates')
print("Keys in the npz file:", data.files)

# Access the dates array
dates = data['dates']
print("Dates present in the file:", dates)

# Extract the 'data' array from the npz file
data_array = data['data']

columns = data['columns']

print(data_array.shape)
# for data in data_array:
#     print(data)
print(columns)

# Suppose you want the temperature map on 2018-01-01
date_idx = 0
column_idx = 0 # depends on what variable you want

sample_grid = data_array[date_idx][column_idx][0][0]

