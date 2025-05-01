import os
import numpy as np
import pandas as pd
import argparse
import re

def is_complete_day(day_data):
    """
    Check if all channels in a day's data have no np.nan.
    """
    for channel_data in day_data:
        if np.isnan(channel_data).any():
            return False
    return True

def scan_npz_directory(directory, year):
    """
    Scan all .npz files in the given directory that end with {year}-mm-dd.npz.
    Return a DataFrame of complete (non-NaN) days.
    """
    records = []
    pattern = re.compile(rf".*_{year}-\d{{2}}-\d{{2}}\.npz$")

    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".npz") and pattern.match(filename):
            print(f"Processing {filename}")
            filepath = os.path.join(directory, filename)
            try:
                npz = np.load(filepath, allow_pickle=True)
                data = npz['data']
                dates = npz['dates']
                region = filename.split("_")[0]
                for i in range(len(dates)):
                    date_str = dates[i]
                    complete = is_complete_day(data[i])
                    records.append({
                        "date": date_str,
                        "region": region,
                        "is_complete": complete
                    })
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")
    print(records)
    return pd.DataFrame(records)

def main(npz_dir, year, output_csv):
    df = scan_npz_directory(npz_dir, year)
    complete_df = df[df["is_complete"] == True].sort_values(by=["region", "date"])
    complete_df.to_csv(output_csv, index=False)
    print(f"Saved complete days to {output_csv}")
    print(complete_df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find dates with complete (non-NaN) data for a given year.")
    parser.add_argument("npz_dir", default="/data/wildfirets/new/compressed", type=str, help="Directory containing NPZ files")
    parser.add_argument("year", type=str, help="Year to filter files by (e.g. 2021)")
    parser.add_argument("output_csv", type=str, help="CSV output path for valid dates")
    args = parser.parse_args()
    
    main(args.npz_dir, args.year, args.output_csv)
