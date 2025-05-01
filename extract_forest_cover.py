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


# limit for Google Earth Engine http requests
REQUEST_LIMIT = 25
# config file containing the feature collection of sub regions in US
GEOJSON_FILE = "config/US_polygons.json"


def prepare_yearly_image(geometry, year: int):
    """Prepare the image from the DisasterMM satellite client."""
    satellite_client = DisasterMM()
    img = satellite_client.compute_forest_cover(
        year,
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
            print(f"Image successfully downloaded to {output_filename}")
        else:
            print(f"Failed to download image {output_filename}. HTTP status code: {response.status}")


async def process_region_async(semaphore, sub_region_polygon, year, session, output_dir, sub_region_number):
    """Asynchronously process a single region on Google Earth Engine with semaphore-controlled concurrency."""
    async with semaphore:
        try:
            feature_image = prepare_yearly_image(sub_region_polygon, year)
            if feature_image:
                download_url = feature_image.getDownloadURL({
                'scale': 500,
                'crs': 'EPSG:4326', # glocal lat/long projection
                'region': sub_region_polygon,
                'format': 'GeoTIFF',  # Specify GeoTIFF format
                'maxPixels': 1e13  # Increase max pixels if needed
                })
            else:
                raise ValueError("feature_image is invalid or empty.")

            # Dynamically generate the output filename based on the current date
            output_filename = f"{output_dir}/{year}_{sub_region_number}.tif"
            
            # logging.info(f"Downloading image {output_filename} from: {download_url}")
            await download_image(session, download_url, output_filename)

        except Exception as e:
            print(f"Error processing region {sub_region_number} for date {year}: {e}")
        return None
    
async def process_year(region_geometries, year, output_dir):
    """Asynchronously process all regions for a single year"""
    semaphore = asyncio.Semaphore(REQUEST_LIMIT)  # Limit concurrent requests to 30
    async with aiohttp.ClientSession() as session:

        print(f"Processing data for all regions for {year}.")

        # Create tasks for all sub-regions
        tasks = [
            process_region_async(semaphore, geometry, year, session, output_dir, sub_region_number)
            for sub_region_number, geometry in enumerate(region_geometries, start=1)
        ]

        await asyncio.gather(*tasks, return_exceptions=True)


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/data/wildfirets/new/downstream", help="output directory for storing the downloaded images")
    parser.add_argument("--start_year", type=int, default=2016, help="Starting year (inclusive), default is 2024")
    parser.add_argument("--end_year", type=int, default=2023, help="Ending year (inclusive), default is 2016")
    args = parser.parse_args()

    # Initialize the time range
    start_year = args.start_year
    end_year = args.end_year

    output_dir = args.output_dir

    # Load Google Earth Engine credentials
    load_dotenv()
    key_file = os.getenv("GEE_KEY_FILEPATH_1")
    service_account = os.getenv("GEE_SERVICE_ACCOUNT_1")

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
            print(f"Skipping sub-region {i} due to missing key: {e}")
        except ee.EEException as e:
            print(f"Skipping invalid geometry for sub-region {i}: {e}")

    for year in range(start_year, end_year + 1):

        print(f"Processing data for {year}")
        asyncio.run(process_year(region_geometries, year, output_dir))

if __name__ == '__main__':
    main()
