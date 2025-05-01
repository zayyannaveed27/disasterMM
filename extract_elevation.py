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


REQUEST_LIMIT = 1
GEOJSON_FILE = "config/US_polygons.json"

ALL_POSSIBLE_BANDS =  ['I1', 'M4', 'M3', 'M11', 'I2', 'I3', 'aerosol depth', 'NDVI_last', 'EVI2_last', 'total precipitation', 'wind speed', 'wind direction', 'minimum temperature', 'maximum temperature',
                        'energy release component', 'specific humidity', 
                        # 'slope', 'aspect', 'elevation',
                        'pdsi', 'LC_Type1', 
                       'forecast total precipitation', 'forecast wind speed', 'forecast wind direction', 'forecast temperature', 'forecast specific humidity']
columns = ["date", "region"] + ALL_POSSIBLE_BANDS

DF = pd.DataFrame(columns=columns)

def prepare_daily_image(geometry, date_of_interest: str, region, time_stamp_start="00:00", time_stamp_end="23:59"):
    """Prepare a daily image collection from the DisasterMM satellite client."""
    satellite_client = DisasterMM()
    data = {
        "date": date_of_interest,
        "region": region,
    }
    try:
        img = satellite_client.compute_elevation_features(
            date_of_interest + 'T' + time_stamp_start,
            date_of_interest + 'T' + time_stamp_end,
            geometry
        )
        return img
    except Exception as e:
        logging.error(f"Error preparing daily image for {date_of_interest} in region {region}: {e}")
        return None

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
            logging.error(f"Failed to download image {output_filename}. HTTP status code: {response.status}")

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
            output_filename = f"{output_dir}/{sub_region_number}.npy"
            
            logging.info(f"Downloading image {output_filename} from: {download_url}")
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
    # parser.add_argument("year", type=int, help="Year for which to process data")
    parser.add_argument("--output_dir", default="/data/wildfirets/new/elevation", help="output directory for storing the downloaded images")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"download_elevation.log"),
        ]
    )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")
    bandinfo_filepath = ""

    # Load Google Earth Engine credentials
    load_dotenv()
    key_file = os.getenv("GEE_KEY_FILEPATH_2")
    service_account = os.getenv("GEE_SERVICE_ACCOUNT_2")

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

    date_of_interest = datetime(2020, 1, 1).strftime('%Y-%m-%d')
    asyncio.run(process_day(region_geometries, date_of_interest, output_dir, bandinfo_filepath))


if __name__ == '__main__':
    main()
