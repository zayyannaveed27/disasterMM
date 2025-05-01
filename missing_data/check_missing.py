import csv
import pandas as pd
import os

year = 2016
df = pd.read_csv(f"missing_{year}.csv")
missing = []
for idx, row in df.iterrows():
    name = f"{row['date']}_{row['region']}.npy"
    path = f"/data/wildfirets/new/{year}/" + name
    if not os.path.isfile(path):
        # add to missing
        missing.append([row['date'], row['region']])

with open("missing_data.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(["date","region"])
    writer.writerows(missing)