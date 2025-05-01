import os
import numpy as np
import pandas as pd
import argparse
from datetime import datetime, timedelta

# Define the expected columns (data channels)
expected_columns = [
    'I1', 'M4', 'M3', 'M11', 'I2', 'I3', 'aerosol depth', 'NDVI_last',
    'EVI2_last', 'total precipitation', 'wind speed', 'wind direction',
    'minimum temperature', 'maximum temperature', 'energy release component',
    'specific humidity', 'pdsi', 'LC_Type1', 'forecast total precipitation',
    'forecast wind speed', 'forecast wind direction', 'forecast temperature',
    'forecast specific humidity', 'slope' , 'aspect' , 'elevation'
]

def load_and_align_file(file_path, region, expected_columns):
    """
    Load a single npy file containing a structured 2D array with fields corresponding to channels.
    
    For each expected channel, extract the 2D data if it exists. Otherwise, create an array 
    of the same shape as the data filled with np.nan.
    
    Returns:
        A list of 2D arrays (one for each expected channel) in the given order.
    """
    # print(file_path)
    data = np.load(file_path, allow_pickle=True)
    elevation = np.load(f'/data/wildfirets/new/elevation/{region}.npy', allow_pickle=True)
    aligned_channels = []
    for col in expected_columns[:-3]:
        if col in data.dtype.names:
            aligned_channels.append(data[col])
        else:
            aligned_channels.append(np.full(data.shape, np.nan))
    for col in ['slope' , 'aspect' , 'elevation']:
        aligned_channels.append(elevation[col])
    return aligned_channels

def save_group(region, group_dates, group_data, output_dir):
    """
    Save a group of consecutive days as an NPZ file.
    
    - group_data is a list of lists, where each inner list holds the aligned 2D arrays for a day.
    - The saved NPZ file will have:
         'data': an object array of shape (n_days, n_channels)
         'dates': the list of date strings for these days
         'columns': the expected channel names
         
    The output file is named as: region_startdate_enddate.npz
    """
    n_days = len(group_dates)
    n_channels = len(expected_columns)
    combined_array = np.empty((n_days, n_channels), dtype=object)
    for i, day_data in enumerate(group_data):
        for j in range(n_channels):
            combined_array[i, j] = day_data[j]
    
    start_date = group_dates[0]
    end_date = group_dates[-1]
    output_filename = f"{region}_{start_date}_{end_date}.npz"
    output_filepath = os.path.join(output_dir, output_filename)
    np.savez_compressed(output_filepath,
                        data=combined_array,
                        dates=np.array(group_dates),
                        columns=np.array(expected_columns))
    print(f"Saved file: {output_filepath}")

def combine_data_for_region(region, start_dt, end_dt, input_dir, output_dir):
    """
    Loop through each day in the date range and collect consecutive days (with no missing file)
    into groups. Each group is saved as soon as it reaches 30 days or when a gap is encountered.
    """
    group_data = []   # list of daily data (each day: list of 2D arrays)
    group_dates = []  # list of corresponding date strings
    
    current_date = start_dt
    while current_date <= end_dt:
        date_str = current_date.strftime("%Y-%m-%d")
        filename = f"{date_str}_{region}.npy"
        file_path = os.path.join(input_dir, filename)
        
        if os.path.exists(file_path):
            aligned_data = load_and_align_file(file_path,region, expected_columns)
            group_data.append(aligned_data)
            group_dates.append(date_str)
            # When the group reaches 30 consecutive days, save it and clear the group.
            if len(group_data) == 30:
                save_group(region, group_dates, group_data, output_dir)
                group_data = []
                group_dates = []
        else:
            # If a day is missing and we have a current group, save it immediately.
            if group_data:
                save_group(region, group_dates, group_data, output_dir)
                group_data = []
                group_dates = []
        current_date += timedelta(days=1)
    
    # Save any remaining group (if any)
    if group_data:
        save_group(region, group_dates, group_data, output_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Combine npy files for a given date range and region into NPZ files of consecutive days."
    )
    parser.add_argument("start_date", type=str, help="Start date in format YYYY-MM-DD")
    parser.add_argument("end_date", type=str, help="End date in format YYYY-MM-DD")
    parser.add_argument("input_dir", type=str, help="Input directory containing npy files")
    parser.add_argument("output_dir", type=str, help="Output directory to save NPZ files")
    args = parser.parse_args()

    # Convert the start and end dates to datetime objects.
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # Loop through each region.
    for region in range(705,706):
        print(f"Processing region: {region}")
        combine_data_for_region(region, start_dt, end_dt, args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
