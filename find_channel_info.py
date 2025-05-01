import os
import numpy as np
import pandas as pd
import argparse
import re

def get_channel_presence(day_data, channel_names):
    """
    Returns a dictionary mapping each channel name to a boolean:
    True if the channel contains no NaNs, False otherwise.
    """
    return {
        channel_names[i]: not np.isnan(day_data[i]).any()
        for i in range(len(channel_names))
    }

def scan_npz_directory(directory):
    """
    Scan all .npz files for a given year and return a DataFrame
    indicating for each date which columns are complete.
    """
    records = []
    pattern = re.compile(rf".*.npz$")
    # pattern = re.compile(rf"1000_.*_{year}-\d{{2}}-\d{{2}}\.npz$")

    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".npz") and pattern.match(filename):
            print(f"Processing {filename}")
            filepath = os.path.join(directory, filename)
            try:
                npz = np.load(filepath, allow_pickle=True)
                data = npz['data']
                dates = npz['dates']
                region = filename.split("_")[0]
                column_names = list(npz['columns'])

                for i in range(len(dates)):
                    day_data = data[i]
                    date_str = dates[i]
                    row = {"date": date_str, "region": region}
                    row.update(get_channel_presence(day_data, column_names))
                    records.append(row)

            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")

    return pd.DataFrame(records)

def main(npz_dir, output_csv):
    df = scan_npz_directory(npz_dir)
    df.sort_values(by=["date", "region"], inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved detailed channel presence report to {output_csv}")
    print(df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report per-channel presence in NPZ files for a given year.")
    parser.add_argument("npz_dir", type=str, help="Directory containing NPZ files")
    parser.add_argument("year", type=str, help="Year to filter files by (e.g. 2021)")
    parser.add_argument("output_csv", type=str, help="CSV output path for channel presence per date")
    args = parser.parse_args()

    main(args.npz_dir, args.output_csv)
