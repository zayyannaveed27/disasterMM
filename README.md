# DisasterMM

This repository contains the code used to recreate the dataset for the senior thesis submission:

**“DisasterMM: A Multimodal Dataset for Pretraining Disaster Foundation Models”**

## 🌎 Overview

**DisasterMM** is a large-scale, high-resolution multimodal dataset designed for both foundation model pretraining and downstream disaster prediction tasks. It includes:

- **2.8 million** daily observations across the continental U.S.
- Spatial resolution of **500 meters** on a **1° lat/lon grid**
- Integrated modalities:
  - Satellite imagery (e.g., VIIRS bands)
  - Meteorological data (temperature, wind, precipitation, humidity)
  - Topographic and drought indicators
  - Vegetation indices (NDVI, EVI2)
  - Land cover classification

In addition to pretraining data, **binary supervision labels** are available for 7 critical natural hazard types:
- Wildfires
- Forest cover loss (deforestation)
- Landslides
- Tornadoes
- Floods
- Thunderstorm winds
- Droughts

---

## ⚙️ Requirements

To recreate the dataset, you will need:

- Access to **Google Earth Engine (GEE)** and **Google Cloud**
- Python ≥ 3.7
- Required packages in `requirements.txt`

---

## 🛠️ Setup Instructions

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Authenticate with Google Cloud**:
   ```bash
   gcloud auth login
   ```

3. **Authenticate with Google Earth Engine**:
   ```bash
   earthengine authenticate
   ```

4. **Create a `.env` file** with your service account credentials:
   ```env
   GEE_KEY_FILEPATH=path/to/key.json
   GEE_SERVICE_ACCOUNT=your-service-account@project.iam.gserviceaccount.com
   ```

   (Add additional keys `GEE_KEY_FILEPATH_2`, etc., for credential rotation if needed)

5. **Configure U.S. sub-regions** using:
   ```
   config/US_polygons.json
   ```
   This file should contain the rectangular polygons covering the U.S.

---

## 📦 Scripts

### 1. `extract_pretrain_images.py`
Generates daily `.npy` files for each region with all multimodal features. Uses asynchronous downloads and rotates GEE credentials to stay under quota limits.

**Example**:
```bash
python extract_pretrain_images.py 2020 --start_month 6 --end_month 8 --output_dir data/fire_images
```

---

### 2. `download_missing_images.py`
Retries downloads for specific (date, region) pairs listed in a CSV (e.g., due to earlier download failures).

**Expected input**: `missing_data/missing_<year>.csv`

**Example**:
```bash
python download_missing_images.py 2020 --output_dir data/fire_images
```

---

### 3. `download_downstream.py`
Downloads downstream binary label masks (`.tif`) for disasters such as landslides, floods, tornadoes, etc.

**Supported tasks**:
- `landslide`
- `debris_flow`
- `floods`
- `tornadoes`
- `severe-thunderstorms`

**Example**:
```bash
python download_downstream.py tornadoes 2018 2021
```

Output files will be saved to:
```
/media/data1/derived/wildfire_ts/downstream/<task>/<YYYY-MM-DD>_<region>.tif
```

---

## 📝 Notes

- Make sure your `.env` file is complete and all credentials have Earth Engine access enabled.
- Downloaded data can consume significant disk space. Monitor your storage usage and optionally filter by time range or region.
- Logs are saved per run in the root directory.

---

## 📫 Contact

For any issues or questions, please contact the repository maintainer.
