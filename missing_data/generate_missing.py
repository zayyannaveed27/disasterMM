import csv
import sys
from datetime import datetime, timedelta

def generate_date_region_csv(start_date, end_date, output_file):
    # Parse the input dates
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        print("Error: Dates must be in YYYY-MM-DD format.")
        sys.exit(1)

    # Ensure the start date is not after the end date
    if start_dt > end_dt:
        print("Error: Start date must be on or before the end date.")
        sys.exit(1)

    # Open the output CSV file for writing
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        current_dt = start_dt
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y-%m-%d')
            # Write one row for each region (1 to 1065)
            for region in range(1, 1066):
                writer.writerow([date_str, region])
            current_dt += timedelta(days=1)

if __name__ == "__main__":
    # Check that the correct number of command line arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python generate_date_regions.py start_date end_date output_file")
        print("Example: python generate_date_regions.py 2020-03-24 2020-03-31 output.csv")
        sys.exit(1)

    start_date = sys.argv[1]
    end_date = sys.argv[2]
    output_file = sys.argv[3]
    generate_date_region_csv(start_date, end_date, output_file)
