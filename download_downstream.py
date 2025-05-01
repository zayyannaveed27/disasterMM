"""
DisasterMM Downstream Label Extraction Script (Google Earth Engine)

This script downloads daily binary masks (GeoTIFF) for specific downstream natural disaster tasks 
such as landslides, tornadoes, floods, debris flows, and severe thunderstorms. Each label image 
indicates the presence (1) or absence (0) of a disaster in each subregion of the U.S. for a given day.

It uses Google Earth Engine (GEE) to access curated event polygon datasets and reduces them to 
region-specific raster images aligned with the DisasterMM dataset’s spatial resolution and coverage.

Supported Tasks:
    - landslide
    - debris_flow
    - floods
    - tornadoes
    - severe-thunderstorms

Features:
    - Dynamically loads task-specific FeatureCollections from GEE assets.
    - Applies task-specific filtering logic to select disaster events for each day.
    - Buffers geometries where needed (e.g., tornado width or thunderstorm wind spread).
    - Reduces filtered features to binary presence/absence masks.
    - Downloads and saves per-day, per-region GeoTIFF images at 500m resolution.
    - Skips regions/dates with no disaster events.

Usage:
    python download_downstream_labels.py <task> <start_year> <end_year>

Example:
    python download_downstream_labels.py tornadoes 2018 2021

Output:
    Saved to:
        /media/data1/derived/wildfire_ts/downstream/<task>/<YYYY-MM-DD>_<region_id>.tif

Environment Variables (.env required):
    GEE_KEY_FILEPATH=<path_to_key.json>
    GEE_SERVICE_ACCOUNT=<service_account_email>

Dependencies:
    - earthengine-api
    - python-dotenv
    - argparse
    - requests, urllib
    - logging
    - datetime, calendar

Required Files:
    - config/US_polygons.json: GeoJSON defining the rectangular sub-regions across the U.S.

Note:
    Region indices in filenames correspond to the order of polygons in the US_polygons.json file (1-indexed).
    Disaster presence is defined based on polygon intersection with region boundaries and timestamp filtering.

Example Output Filename:
    2020-07-14_3.tif → Indicates a binary presence mask for region 3 on July 14, 2020.

This script helps generate ground-truth supervision for evaluating downstream disaster prediction models.
"""

import ee
import json
import os
import datetime
import requests
import urllib.request
import zipfile
import calendar
from dotenv import load_dotenv
import argparse
import logging


def get_region_geometries():
    # Load rectangular sub-regions from geojson
    GEOJSON_FILE = "config/US_polygons.json"
    with open(GEOJSON_FILE, 'r') as f:
        sub_regions = json.load(f)

    # Prepare the list of ee.Geometry.Polygon objects
    region_geometries = []
    for i, sub_region in enumerate(sub_regions['features']):  
        coordinates = sub_region['geometry']['coordinates']
        region_geometries.append(ee.Geometry.Polygon(coordinates))
    
    return region_geometries

def download_region_image(image, region_idx, region_geom, current_date, output_dir):
    try:
        path = image.getDownloadURL({
            'scale': 500,
            'crs': 'EPSG:4326',
            'region': region_geom,
            'format': 'GeoTIFF',
            'maxPixels': 1e13
        })

        filename = f"{output_dir}/{current_date.isoformat()}_{region_idx}.tif"
        logging.info(f"Downloading {filename}...")
        urllib.request.urlretrieve(path, filename)
        logging.info(f"Saved {filename}")

    except Exception as e:
        logging.warning(f"Failed to download {filename}: {e}")

def debris_filter_func(collection, year, month, day):
    begin_yearmonth = year * 100 + month
    return collection \
        .filter(ee.Filter.eq('BEGIN_DAY', day)) \
        .filter(ee.Filter.eq('BEGIN_YEARMONTH', begin_yearmonth))

def thunderstorm_filter_func(collection, year, month, day):
    begin_yearmonth = year * 100 + month
    return collection \
        .filter(ee.Filter.eq('BEGIN_DAY', day)) \
        .filter(ee.Filter.eq('BEGIN_YEARMONTH', begin_yearmonth)) \
        .map(buffer_thunderstorm_feature)
    
def buffer_thunderstorm_feature(feature):
    buffered_geom = feature.geometry().buffer(1000)
    return feature.setGeometry(buffered_geom)
    

def landslide_filter_func(collection, year, month, day):
    curr_datetime = datetime.datetime(year, month, day)
    next_datetime = curr_datetime + datetime.timedelta(days=1)
    start_timestamp = int(curr_datetime.timestamp()) * 1000
    end_timestamp = int(next_datetime.timestamp()) * 1000

    return collection \
        .filter(ee.Filter.gte('ev_date', start_timestamp)) \
        .filter(ee.Filter.lt('ev_date', end_timestamp))

def flood_filter_func(collection, year, month, day):
    return collection.filter(
        ee.Filter.And(
            ee.Filter.Or(
                ee.Filter.lt('BEGIN_YEARMONTH', year * 100 + month),
                ee.Filter.And(
                    ee.Filter.eq('BEGIN_YEARMONTH', year * 100 + month),
                    ee.Filter.lte('BEGIN_DAY', day)
                )
            ),
            ee.Filter.Or(
                ee.Filter.gt('END_YEARMONTH', year * 100 + month),
                ee.Filter.And(
                    ee.Filter.eq('END_YEARMONTH', year * 100 + month),
                    ee.Filter.gte('END_DAY', day)
                )
            )
        )
    )

def tornado_filter_func(collection, year, month, day):
    begin_yearmonth = year * 100 + month
    filtered_collection = collection \
        .filter(ee.Filter.eq('BEGIN_DAY', day)) \
        .filter(ee.Filter.eq('BEGIN_YEARMONTH', begin_yearmonth))
    return filtered_collection.map(buffer_tornado_feature)

def buffer_tornado_feature(feature):
    width_m = ee.Number(feature.get('TOR_WIDTH')).multiply(0.9144)
    # logging.info(f"Width in meters: {width_m}")
    # Buffer the geometry by half the width
    buffered_geom = feature.geometry().buffer(width_m.divide(2))
    # logging.info(f"Feature info: {feature.getInfo()}")
    return feature.setGeometry(buffered_geom)

def compute_daily_disaster_images(filter_func, collection, region_geometries, year, month, day, output_dir):
    current_date = datetime.date(year, month, day)

    # Apply dataset-specific daily filtering
    daily_filtered = filter_func(collection, year, month, day)

    if daily_filtered.size().getInfo() == 0:
        print(f"Skipping {current_date.isoformat()} — no data anywhere")
        return

    for region_num, region_geom in enumerate(region_geometries):
        region_idx = region_num + 1
        print(f'{current_date.isoformat()} | Region {region_idx}...')

        region_filtered = daily_filtered.filterBounds(region_geom)

        if region_filtered.size().getInfo() == 0:
            print(f"No data in region {region_idx} on {current_date.isoformat()}")
            continue

        image = region_filtered.map(lambda f: f.set('value', 1)).reduceToImage(
            properties=['value'],
            reducer=ee.Reducer.first()
        ).unmask(0).clip(region_geom)

        download_region_image(image, region_idx, region_geom, current_date, output_dir)

def main():
    # Load Google Earth Engine credentials
    load_dotenv()
    key_file = os.getenv("GEE_KEY_FILEPATH")
    service_account = os.getenv("GEE_SERVICE_ACCOUNT")

    # Initialize Earth Engine
    credentials = ee.ServiceAccountCredentials(service_account, key_file)
    ee.Initialize(credentials)

    parser = argparse.ArgumentParser()
    # TODO: update the help
    parser.add_argument("downstream_task", help="landslide, debris_flow, etc....")
    parser.add_argument("start_year", help="starting year YYYY")
    parser.add_argument("end_year", help="ending year YYYY")
    args = parser.parse_args()

    task = args.downstream_task
    output_dir = f"/media/data1/derived/wildfire_ts/downstream/{task}"
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    region_geometries = get_region_geometries()

        # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"downsteam_{task}.log"),
        ]
    )

    if task == "landslide":
        collection = ee.FeatureCollection('projects/ai-refire/assets/landslides-final')
        filter_func = landslide_filter_func
    elif task == "debris_flow":
        collection = ee.FeatureCollection('projects/ai-refire/assets/debris-polygons')
        filter_func = debris_filter_func
    elif task == "floods":
        collection = ee.FeatureCollection('projects/ai-refire/assets/flood-poly')
        filter_func = flood_filter_func
    elif task == "tornadoes":
        collection = ee.FeatureCollection('projects/ai-refire/assets/tornado-poly')
        filter_func = tornado_filter_func
    elif task == "severe-thunderstorms":
        collection = ee.FeatureCollection('projects/ai-refire/assets/thunderstorm-poly')
        filter_func = thunderstorm_filter_func
    else:
        raise ValueError(f"Unsupported task: {task}")

    for year in range(int(args.start_year), int(args.end_year) + 1):
        for month in range(1, 13):
            days_in_month = calendar.monthrange(year, month)[1]
            for day in range(1, days_in_month + 1):
                compute_daily_disaster_images(filter_func, collection, region_geometries, year, month, day, output_dir)

    
if __name__=='__main__':
    main()