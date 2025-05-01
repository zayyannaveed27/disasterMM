"""
Asynchronous DisasterMM Image Processing with Google Earth Engine

This script asynchronously processes high-resolution, multimodal satellite imagery using
Google Earth Engine (GEE) and the DisasterMM client across U.S. regions from 2016–2023.
It is designed to support the creation of large-scale pretraining data and downstream inputs
for a wide range of natural disaster prediction tasks.

Supported Features:
    - Processes imagery from the DisasterMM dataset, including:
        - VIIRS bands (I1, I2, M4, M11, etc.)
        - Meteorological data (wind speed, precipitation, humidity, temperature)
        - Vegetation indices (NDVI, EVI2)
        - Topographic and drought features (PDSI, landcover type)
        - Forecast features (e.g., forecasted wind, temperature)
    - Downloads daily region-level features at 500-meter resolution in `.npy` format.
    - Fully asynchronous: uses `asyncio` and `aiohttp` to process all regions in parallel.
    - Automatically rotates across multiple GEE service accounts to avoid quota bottlenecks.
    - Supports configurable date ranges and output directories.

Usage:
    python script_name.py <year> 
                          [--start_month <1-12>] 
                          [--end_month <1-12>] 
                          [--output_dir <output_directory>] 
                          [--bandinfo_file <csv_path>]

Example:
    python script_name.py 2020 --start_month 4 --end_month 9 --output_dir data/disaster_images

Output:
    .npy feature files are saved in:
        <output_dir>/<year>/<YYYY-MM-DD>_<region_id>.npy

Environment Setup (.env required):
    Ensure you have the necessary Google Earth Engine credentials and .env file configured with:
    GEE_KEY_FILE=<path_to_key_file>
    GEE_SERVICE_ACCOUNT=<service_account_email>


Dependencies:
    - earthengine-api
    - aiohttp
    - python-dotenv
    - pandas

Required Files:
    - config/US_polygons.json: GeoJSON containing polygon boundaries for all U.S. subregions.
    - missing_values/500_column_names.csv: (Optional) to track any missing feature bands.
    - DisasterMM/DisasterMM.py: Local client module to interface with Earth Engine. 
"""

import os
import ee
import json
from disasterMM.DisasterMM import DisasterMM
from datetime import datetime, timedelta
from dotenv import load_dotenv
import argparse
import asyncio
import aiohttp
import pandas as pd
import logging


REQUEST_LIMIT = 35
GEOJSON_FILE = "config/US_polygons.json"

ALL_POSSIBLE_BANDS =  ['I1', 'M4', 'M3', 'M11', 'I2', 'I3', 'aerosol depth', 'NDVI_last', 'EVI2_last', 'total precipitation', 'wind speed', 'wind direction', 'minimum temperature', 'maximum temperature',
                        'energy release component', 'specific humidity', 
                        # 'slope', 'aspect', 'elevation',
                        'pdsi', 'LC_Type1', 
                       'forecast total precipitation', 'forecast wind speed', 'forecast wind direction', 'forecast temperature', 'forecast specific humidity']
columns = ["date", "region"] + ALL_POSSIBLE_BANDS

def update_env_variable(key, value, env_file=".env"):
    """Update a single environment variable in the .env file."""
    lines = []
    found = False

    # Read existing lines
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            lines = f.readlines()

    # Update the key if it exists
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            found = True
            break

    # If the key was not found, append it
    if not found:
        lines.append(f"{key}={value}\n")

    # Write back to the file
    with open(env_file, "w") as f:
        f.writelines(lines)

def prepare_daily_image(geometry, date_of_interest: str, region, time_stamp_start="00:00", time_stamp_end="23:59"):
    """Prepare a daily image collection from the DisasterMM satellite client."""
    satellite_client = DisasterMM()
    data = {
        "date": date_of_interest,
        "region": region,
    }
    # try:
    img = satellite_client.compute_daily_features(
        date_of_interest + 'T' + time_stamp_start,
        date_of_interest + 'T' + time_stamp_end,
        geometry
    )
    return img
    # except Exception as e:
    #     logging.error(f"Error preparing daily image for {date_of_interest} in region {region}: {e}")
    #     return None

async def download_image(session, url, output_filename):
    """
    Asynchronously download an image from the given URL and save it to a local file.
    """

    async with session.get(url) as response:
        if response.status == 200:
            with open(output_filename, 'wb') as f:
                while True:
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    f.write(chunk)
            # logging.info(f"Image successfully downloaded to {output_filename}")
        else:
            raise RuntimeError(f"Failed to download image {output_filename}. HTTP status code: {response.status}")
            # logging.error(f"Failed to download image {output_filename}. HTTP status code: {response.status}")

async def process_region_async(semaphore, sub_region_polygon, date_of_interest, session, output_dir, sub_region_number):
    """Asynchronously process a single region on Google Earth Engine with semaphore-controlled concurrency."""
    async with semaphore:
        try:
            feature_image = prepare_daily_image(sub_region_polygon, date_of_interest,sub_region_number)
            if feature_image:
                download_url = feature_image.getDownloadURL({
                'scale': 500, # 500 meters resolution
                'crs': 'EPSG:4326', # glocal lat/long projection
                'region': sub_region_polygon,
                'format': 'NPY',  # Use NPY format
                'maxPixels': 1e13  # Increase max pixels if needed
                })
            else:
                raise ValueError("feature_image is invalid or empty.")

            # Dynamically generate the output filename based on the current date
            output_filename = f"{output_dir}/{date_of_interest}_{sub_region_number}.npy"
            
            # logging.info(f"Downloading image {output_filename} from: {download_url}")
            await download_image(session, download_url, output_filename)

        except Exception as e:
            logging.error(f"Error processing region {sub_region_number} for date {date_of_interest}: {e}")
        # return data


async def process_day(region_geometries, date_of_interest, output_dir, bandinfo_filepath):
    """Asynchronously process all regions for a single day."""
    semaphore = asyncio.Semaphore(REQUEST_LIMIT)  # Limit concurrent requests to 30
    async with aiohttp.ClientSession() as session:

        logging.info(f"Processing data for all regions for {date_of_interest}.")

        # Create tasks for all sub-regions
        tasks = [
            process_region_async(semaphore, geometry, date_of_interest, session, output_dir, sub_region_number)
            for sub_region_number, geometry in enumerate(region_geometries, start=1)
        ]

        await asyncio.gather(*tasks, return_exceptions=True)


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int, help="Year for which to process data")
    parser.add_argument("--output_dir", default="/data/wildfirets/new", help="output directory for storing the downloaded images")
    parser.add_argument("--bandinfo_file", default="missing_values/500_column_names.csv", help="output csv for storing the missing column values")
    parser.add_argument("--end_month", type=int, default=12, help="Ending month (inclusive), default is December")
    parser.add_argument("--start_month", type=int, default=1, help="Starting month (inclusive), default is January")
    args = parser.parse_args()

    # Initialize the time range
    year = args.year
    start_month = args.start_month
    end_month = args.end_month

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"download_{year}.log"),
        ]
    )

    output_dir = args.output_dir + f"/{year}"
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")
    bandinfo_filepath = args.bandinfo_file

    # Load Google Earth Engine credentials
    load_dotenv(override=True)
    key_file = os.getenv(f"GEE_KEY_FILEPATH")
    service_account = os.getenv(f"GEE_SERVICE_ACCOUNT")


    # Initialize Earth Engine
    credentials = ee.ServiceAccountCredentials(service_account, key_file)
    ee.Initialize(credentials)

    # Load sub-regions in US
    with open(GEOJSON_FILE, 'r') as f:
        sub_regions = json.load(f)

    region_geometries = []
    # Iterate through the polygons and create ee.Geometry.Polygon objects
    for i, sub_region in enumerate(sub_regions['features']):  
        try:
            coordinates = sub_region['geometry']['coordinates']
            region_geometries.append(ee.Geometry.Polygon(coordinates))
        except KeyError as e:
            logging.warning(f"Skipping sub-region {i} due to missing key: {e}")
        except ee.EEException as e:
            logging.warning(f"Skipping invalid geometry for sub-region {i}: {e}")

    # Process each day in the time range
    for month in range(start_month, end_month + 1):
        month_start = datetime(year, month, 1)
        next_month = (month_start + timedelta(days=32)).replace(day=1)
        month_end = next_month - timedelta(days=1)

        logging.info(f"Processing data for {year}-{month:02d}")

        current_date = month_start
        while current_date <= month_end:
            date_of_interest = current_date.strftime('%Y-%m-%d')
            logging.info(f"Processing data for {date_of_interest}")
            asyncio.run(process_day(region_geometries, date_of_interest, output_dir, bandinfo_filepath))
            current_date += timedelta(days=1)


if __name__ == '__main__':
    main()
