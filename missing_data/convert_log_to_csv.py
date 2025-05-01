import re
import csv
import argparse

def extract_date_region(input_log, output_csv):
    # Regular expression pattern to capture the date and region from the filename.
    file_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})_(\d+)\.npy')
    # This pattern looks for a region and date in the format YYYY-MM-DD in the error message
    error_pattern = re.compile(r'Error processing region (\d+) for date (\d{4}-\d{2}-\d{2})')
    
    with open(input_log, 'r') as infile, open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write CSV header
        csv_writer.writerow(['date', 'region'])
        
        for line in infile:
            # Process only error lines.
            if "[ERROR]" in line:
                file_match = file_pattern.search(line)
                error_match = error_pattern.search(line)
                if error_match:
                    region = error_match.group(1)
                    date = error_match.group(2)
                    csv_writer.writerow([date, region])
                elif file_match:
                    date = file_match.group(1)
                    region = file_match.group(2)
                    csv_writer.writerow([date, region])
                else:
                    print(f"Skipping line {line}")
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int, help="year for calculating missing data")
    args = parser.parse_args()
    year = args.year
    input_log = f"/data/wildfirets/download/download_{year}.log"
    output_csv = f"/data/wildfirets/download/missing_data/missing_{year}.csv"
    extract_date_region(input_log, output_csv)




