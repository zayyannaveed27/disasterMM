"""
Asynchronous Missing Image Downloader for DisasterMM with Google Earth Engine

This script identifies and downloads missing daily multimodal satellite imagery from 
the DisasterMM dataset using Google Earth Engine (GEE). It is designed to recover 
missing `.npy` files based on a CSV log of (date, region) pairs that previously failed 
to download or were lost.

Key Features:
    - Loads sub-region polygons from a GeoJSON file (e.g., U.S. states/counties).
    - Accepts a CSV file listing (date, region) combinations with missing data.
    - Computes the daily multimodal features using the DisasterMM Earth Engine client.
    - Downloads image tiles in `.npy` format at 500m spatial resolution.
    - Uses asyncio with semaphore control to manage parallel downloads efficiently.
    - Supports GEE service account authentication using a `.env` file.

Usage:
    python missing_downloader.py <year> 
                                 [--start_month <1-12>] 
                                 [--end_month <1-12>] 
                                 [--output_dir <output_directory>]

Example:
    python missing_downloader.py 2020 --start_month 6 --end_month 8 --output_dir data/fixed

Expected CSV format:
    A file named `missing_data/missing_<year>.csv` must exist, with two columns:
        - `date` (in format YYYY-MM-DD)
        - `region` (integer sub-region ID corresponding to polygon index + 1)

Output:
    Each recovered file is saved to:
        <output_dir>/<year>/<YYYY-MM-DD>_<region_id>.npy

Environment Variables (.env):
    GEE_KEY_FILEPATH=<path_to_key.json>
    GEE_SERVICE_ACCOUNT=<email_for_service_account>

Dependencies:
    - earthengine-api
    - aiohttp
    - python-dotenv
    - pandas

Required Files:
    - config/US_polygons.json: GeoJSON defining U.S. sub-regions.
    - missing_data/missing_<year>.csv: List of failed (date, region) downloads to retry.
    - DisasterMM/DisasterMM.py: Client for feature extraction using GEE.

This script is typically run after the main downloader to patch missing data and ensure
complete temporal and spatial coverage for the DisasterMM dataset.
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

# limit for Google Earth Engine http requests
REQUEST_LIMIT = 30
# config file containing the feature collection of sub regions in US
GEOJSON_FILE = "config/US_polygons.json"

def prepare_daily_image(geometry, date_of_interest: str, region, time_stamp_start="00:00", time_stamp_end="23:59"):
    """Prepare a daily image collection from the DisasterMM satellite client."""
    satellite_client = DisasterMM()
    img = satellite_client.compute_daily_features(
        date_of_interest + 'T' + time_stamp_start,
        date_of_interest + 'T' + time_stamp_end,
        geometry
    )
    return img

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
            logging.info(f"Image successfully downloaded to {output_filename}")
        else:
            raise RuntimeError(f"Failed to download image {output_filename}. HTTP status code: {response.status}")

async def process_region_async(semaphore, sub_region_polygon, date_of_interest, session, output_dir, sub_region_number):
    """Asynchronously process a single region on Google Earth Engine with semaphore-controlled concurrency."""
    async with semaphore:
        try:
            print(f"Processing region {sub_region_number} for date {date_of_interest}.")
            feature_image = prepare_daily_image(sub_region_polygon, date_of_interest,sub_region_number)
            if feature_image:
                download_url = feature_image.getDownloadURL({
                'scale': 500,
                'crs': 'EPSG:4326',
                'region': sub_region_polygon,
                'format': 'NPY',  # Specify GeoTIFF format
                'maxPixels': 1e13  # Increase max pixels if needed
                })
            else:
                raise ValueError("feature_image is invalid or empty.")

            # Dynamically generate the output filename based on the current date
            output_filename = f"{output_dir}/{date_of_interest}_{sub_region_number}.npy"
            
            # print(f"Downloading image {output_filename} from: {download_url}")
            await download_image(session, download_url, output_filename)

        except Exception as e:
            logging.error(f"Error processing region {sub_region_number} for date {date_of_interest}: {e}")
        return None

async def process_missing(region_geometries, missing, output_dir):
    """Asynchronously process all regions for a single day."""
    semaphore = asyncio.Semaphore(REQUEST_LIMIT)  # Limit concurrent requests to 30
    async with aiohttp.ClientSession() as session:

        # Create tasks for all sub-regions
        tasks = [
            process_region_async(semaphore, region_geometries[sub_region_number-1], date_of_interest, session, output_dir, sub_region_number)
            for date_of_interest, sub_region_number in missing
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

def main():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int, help="year for downloading images")
    parser.add_argument("--output_dir", default="/data/wildfirets/new", help="output directory for storing the downloaded images")
    parser.add_argument("--end_month", type=int, default=12, help="Ending month (inclusive), default is December")
    parser.add_argument("--start_month", type=int, default=1, help="Starting month (inclusive), default is January")
    args = parser.parse_args()

    # Initialize the time range
    year = args.year
    output_dir = args.output_dir + f"/{year}"

    # Load Google Earth Engine credentials
    load_dotenv()
    key_file = os.getenv("GEE_KEY_FILEPATH_1")
    service_account = os.getenv("GEE_SERVICE_ACCOUNT_1")

    # Initialize Earth Engine
    credentials = ee.ServiceAccountCredentials(service_account, key_file)
    ee.Initialize(credentials)

        # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"download_missing_{year}.log"),
        ]
    )

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
            print(f"Skipping sub-region {i} due to missing key: {e}")
        except ee.EEException as e:
            print(f"Skipping invalid geometry for sub-region {i}: {e}")

    df = pd.read_csv(f"missing_data/missing_{year}.csv")
    # df = pd.read_csv(f"missing_data/missing.csv")
    date_region_list = list(df.itertuples(index=False, name=None))
    print(date_region_list[:5])
    asyncio.run(process_missing(region_geometries, date_region_list, output_dir))

if __name__ == '__main__':
    main()
