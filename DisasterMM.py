'''
Code based on https://github.com/SebastianGer/WildfireSpreadTSCreateDataset.
'''

import datetime
import ee
import math


class DisasterMM:
    def __init__(self):
        """_summary_ This class describes which data to extract how from Google Earth Engine. 
        The init defines the different source data products to use. 
        """
        self.name = "DisasterMM:"
        # Digital elevation model
        self.srtm = ee.Image("USGS/SRTMGL1_003") # 30 meters
        self.landcover = ee.ImageCollection("MODIS/061/MCD12Q1") # 500 meters
        self.weather = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET") # 4638.3 meters
        self.weather_forecast = ee.ImageCollection('NOAA/GFS0P25') # 27830 meters
        self.drought = ee.ImageCollection("GRIDMET/DROUGHT") # 4638.3 meters
        # VIIRS surface reflectance
        self.viirs = ee.ImageCollection("NASA/VIIRS/002/VNP09GA") # 500m or 1km
        # VIIRS vegetation index
        self.viirs_veg_idx = ee.ImageCollection("NASA/VIIRS/002/VNP13A1") # 500m
        self.aerosol_depth = ee.ImageCollection("MODIS/061/MCD19A2_GRANULES") # 1km
        # TODO:
        # FIX dataset scale - 375m to 500m
        # resampling - nearest neighbor or bilinear?
        # Remove forecast data? or add HRRR forecast data?
        #  add viirs M4 , M3 (green, blue I1 is red) + I3
        #  possibly add snow cover daily

        # extract vegetation fields yearly for deforestation
        # extract data for floods

        #* Downstream Tasks

        # Completed
        self.forest_cover = ee.ImageCollection("MODIS/061/MOD44B") # 250 meters
        self.landslides = ee.FeatureCollection('projects/ai-refire/assets/landslide_report_pts')
        # Remaining
        self.floods = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1") # 30 meters
        # self.fire_modis = ee.ImageCollection("FIRMS") # 1000 meters
        # self.fire_viirs = ee.ImageCollection("NASA/LANCE/NOAA20_VIIRS/C2") # 375 meters
        self.active_fire = ee.FeatureCollection('projects/ai-refire/assets/viirs_all')
        self.snow_cover = ee.ImageCollection("MODIS/061/MYD10A1") # 500 meters

    def compute_fire_data(self, start_time:str, end_time:str, geometry:ee.Geometry):
        pass
        # # Active Fire Data
        # # remove low confidence interval data
        # #  add this for binary value .map(lambda f: f.set('active fire', 1))
        # active_fire = self.active_fire \
        #                 .filterBounds(geometry) \
        #                 .filter(ee.Filter.neq('CONFIDENCE', 'low')) \
        #                 .filter(ee.Filter.gte('ACQ_DATE', today_timestamp)) \
        #                 .filter(ee.Filter.lt('ACQ_DATE', end_timestamp)) \
        #                 .map(self.get_buffer_af) \
        #                 .reduceToImage(properties=['BRIGHT_TI4'], reducer=ee.Reducer.max()) \
        #                 .unmask(0)

    def compute_landslide_data(self, start_time:str, end_time:str, geometry:ee.Geometry):
        start = datetime.datetime.strptime(start_time[:-6], '%Y-%m-%d')
        start_timestamp = int(datetime.datetime.timestamp(start)) * 1000
        end_timestamp = int(datetime.datetime.timestamp(start + datetime.timedelta(days=1))) * 1000

        landslides = self.landslides.filterBounds(geometry) \
        .filter(ee.Filter.inList('loc_accu', ['1km', 'exact'])) \
        .filter(ee.Filter.gte('ev_date', start_timestamp)) \
        .filter(ee.Filter.lt('ev_date', end_timestamp)) \
        .map(lambda f: f.set('landslide', 1)) \
        .map(self.get_buffer_landslide) \
        .reduceToImage(properties=['landslide'], reducer=ee.Reducer.max()) \
        .unmask(0)
    
        return landslides.clip(geometry)

    def compute_forest_cover(self, year:int, geometry:ee.Geometry):
        """_summary_ Computer forest cover from Google Earth Engine.

        Args:
            start_time (str): _description_
            end_time (str): _description_
            geometry (ee.Geometry): _description_

        Returns:
            ee.Imagen: _description_ Image containing forest cover band inside the given geometry
        """

        # Forest Cover Data
        forest_year = year
        forest_cover = self.forest_cover.filterDate(f"{forest_year}-01-01", f"{forest_year}-12-31").filterBounds(geometry).select(['Percent_Tree_Cover']).median()

        # Return the clipped image
        return forest_cover.clip(geometry)

    def compute_daily_features(self, start_time:str, end_time:str, geometry:ee.Geometry):
        """_summary_ Compute the daily features in Google Earth Engine.

        Args:
            start_time (str): _description_
            end_time (str): _description_
            geometry (ee.Geometry): _description_

        Returns:
            ee.ImageCollection: _description_ ImageCollection containing one image, 
            with all desired features for the given day, inside the given geometry.
        """


        # Time objects we need later. We add "000" to timestamps, because GEE has timestamps with miliseconds,
        # but datetime doesn't by default
        today_string = start_time[:-6].replace("-", "")
        today = datetime.datetime.strptime(start_time[:-6], '%Y-%m-%d')
        today_timestamp = int(datetime.datetime.timestamp(today)) * 1000
        end_timestamp = int(datetime.datetime.timestamp(today + datetime.timedelta(days=1))) * 1000

        # Weather Data
        # Median is used to turn ee.ImageCollection into a single ee.Image.
        # Each ImageCollection should only contain a single image at this point.
        weather = self.weather.filterDate(start_time, end_time).filterBounds(geometry)
        precipitation = weather.select('pr').median().rename("total precipitation")
        wind_direction = weather.select('th').median().rename("wind direction")
        temperature_min = weather.select('tmmn').median().rename("minimum temperature")
        temperature_max = weather.select('tmmx').median().rename("maximum temperature")
        energy_release_component = weather.select('erc').median().rename("energy release component")
        specific_humidity = weather.select('sph').median().rename("specific humidity")
        wind_velocity = weather.select('vs').median().rename("wind speed")

        # Weather Forecast Data
        # Take forecasts made at midnight (00), and that tell us something about the hours between 01 and 24.
        # Important: The forecasts at 00 contain six features instead of nine, like all others.
        weather_forecast = self.weather_forecast.filter(
            ee.Filter.gte("system:index", today_string + "00F01")).filter(
            ee.Filter.lte("system:index", today_string + "00F24")
        ).filterBounds(geometry)
        forecast_temperature = weather_forecast.select("temperature_2m_above_ground").mean().rename(
            "forecast temperature")
        forecast_specific_humidity = weather_forecast.select("specific_humidity_2m_above_ground").mean().rename(
            "forecast specific humidity")
        forecast_u_wind = weather_forecast.select("u_component_of_wind_10m_above_ground").mean()
        forecast_v_wind = weather_forecast.select("v_component_of_wind_10m_above_ground").mean()

        if forecast_u_wind.bandNames().size().getInfo() > 0 and forecast_v_wind.bandNames().size().getInfo() > 0:
            # Transform from u/v to direction and speed, to align with GRIDMET and DEM data
            forecast_wind_speed = forecast_u_wind.multiply(forecast_u_wind).add(
                forecast_v_wind.multiply(forecast_v_wind)).sqrt().rename("forecast wind speed")
            forecast_wind_direction = forecast_v_wind.divide(forecast_u_wind).atan()
            forecast_wind_direction = forecast_wind_direction.divide(2 * math.pi).multiply(360).rename(
                "forecast wind direction")
        else:
            forecast_wind_speed = ee.Image([])
            forecast_wind_direction =  ee.Image([])

        # Rain forecasts were changed: From rain within the one-hour interval to cumulative rain during the day so far
        forecast_rain_change_date = datetime.datetime.strptime("2019-11-07T06:00:00", '%Y-%m-%dT%H:%M:%S')
        forecast_rain = weather_forecast.select("total_precipitation_surface")
        if today <= forecast_rain_change_date:
            forecast_rain = forecast_rain.reduce(ee.Reducer.sum())
        else:
            forecast_rain = forecast_rain.reduce(ee.Reducer.last())
        forecast_rain = forecast_rain.rename("forecast total precipitation")
        
        # Elevation Data
        # elevation = self.srtm.select('elevation')
        # reduce resolution to 500 to match other datasets
        # elevation = self.srtm.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=60000) \
        #             .reproject(crs='EPSG:4326', scale=500)
        # slope = ee.Terrain.slope(elevation)
        # aspect = ee.Terrain.aspect(elevation)

        # Drought Data
        # Only available every fifth day, but we can find the valid entry via time_start and time_end
        drought_index = self.drought \
            .filter(ee.Filter.lte("system:time_start", today_timestamp)) \
            .filter(ee.Filter.gte("system:time_end", today_timestamp)) \
            .select('pdsi').median()
        
        # Land Cover Data
        igbp_land_cover = self.landcover.filterDate(start_time[:4] + '-01-01', start_time[:4] + '-12-31').filterBounds(
            geometry).select('LC_Type1').median()
        
        # self.landcover = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        # igbp_land_cover = self.landcover.filterDate(start_time, end_time).filterBounds(geometry) \
        #             .select('label').reduce(ee.Reducer.last()) \
        #             .setDefaultProjection('EPSG:4326', None, 10) \
        #             .reduceResolution(reducer=ee.Reducer.mode(), maxPixels=60000) \
        #             .reproject(crs='EPSG:4326', scale=500)

        # VIIRS Satellite Data
        satellite_img = self.viirs.filterDate(start_time, end_time).filterBounds(geometry).select(
            ['I1', 'M4', 'M3', 'M11', 'I2', 'I3']).median()
            # ['M11', 'I2', 'I1']).median()

        # VIIRS Vegetation Index Data
        viirs_veg_idc = self.viirs_veg_idx.filterDate((
                datetime.datetime.strptime(end_time[:-6], '%Y-%m-%d') + datetime.timedelta(-15)).strftime(
            '%Y-%m-%d'), end_time).filterBounds(geometry).select(['NDVI', 'EVI2']).reduce(
            ee.Reducer.last())
        

        # MODIS Aerosol Depth Data
        aerosol_depth = self.aerosol_depth.filterDate(start_time, end_time).filterBounds(geometry) \
            .select('Optical_Depth_047').median().rename("aerosol depth")
    
        
        combined_img = ee.Image(
            [satellite_img, aerosol_depth, viirs_veg_idc, precipitation, wind_velocity, wind_direction, temperature_min, temperature_max,
             energy_release_component, specific_humidity, 
             # slope, aspect, elevation, 
             drought_index, igbp_land_cover,
             forecast_rain, forecast_wind_speed, forecast_wind_direction, forecast_temperature,
             forecast_specific_humidity,
             ])


        # Clip the combined image to the input geometry
        clipped_img = combined_img.clip(geometry)
    
        # Return an ImageCollection containing the clipped image
        return clipped_img
    
    def compute_elevation_features(self, start_time:str, end_time:str, geometry:ee.Geometry):
        """_summary_ Compute the daily features in Google Earth Engine.

        Args:
            start_time (str): _description_
            end_time (str): _description_
            geometry (ee.Geometry): _description_

        Returns:
            ee.ImageCollection: _description_ ImageCollection containing one image, 
            with all desired features for the given day, inside the given geometry.
        """      
        # Elevation Data
        elevation = self.srtm.select('elevation')
        # reduce resolution to 500 to match other datasets
        elevation = self.srtm.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=60000) \
                    .reproject(crs='EPSG:4326', scale=500)
        slope = ee.Terrain.slope(elevation)
        aspect = ee.Terrain.aspect(elevation)


        combined_img = ee.Image(
            [slope, aspect, elevation, ]
        )

        # Clip the combined image to the input geometry
        clipped_img = combined_img.clip(geometry)
    
        # Return an ImageCollection containing the clipped image
        return clipped_img

    def get_buffer_af(self, feature):
        # TODO: check if buffer size matches with resolution of the dataset
        return feature.buffer(375 / 2).bounds()
    
    def get_buffer_landslide(self, feature):
        return feature.buffer(1000 / 2).bounds()
