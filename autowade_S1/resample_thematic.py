# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:31:15 2024

@author: Francesca Trevisiol

The resample_thematic function resamples a thematic raster dataset (e.g., land cover, slope) to match the spatial resolution, 
extent, and projection of a given reference Sentinel image. It is designed to extract and align specific thematic masks 
(e.g., permanent water, urban areas, steep slopes) required for flood mapping algorithms.

The function:

-Supports multiple resampling methods (nearest, bilinear, cubic, average, etc.)
-Applies masking based on class labels or threshold values
-Works with datasets in different coordinate reference systems and resolutions
-Is tailored for use in flood extent detection chains based on Sentinel-1 or Sentinel-2 imagery on the WASDI platform

The function returns a np.array (same shape as the reference image) representing a binary mask with the following values:
-0: background (non-target areas)
-out_value: pixels matching the selected label_mask class from the thematic raster
  For example:
  - label_mask="water" → pixels where land cover class = water (DN=80)
  - label_mask="slope" → pixels with slope ≥ threshold (e.g., 5°)

"""

import rasterio
import os
import numpy as np
import wasdi
from osgeo import gdal

def write_geotiff(data, transform, crs, output_path):
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def resample_thematic(InputRefImagePath, input_LC_path, label_mask, resample_method, out_value):
    '''
    Parameters
    ----------
    InputRefImagePath : 
        path to the raster that is the reference
    input_LC_path : 
        path to the raster that is the one to be resample/reprojected and a specific vale needs to be extracted from it
    resample_method:
        "nearest", "mode"
    label_mask : string
        "Ocean","urban", "water","Tree", "Shrubland", "Grassland", "Cropland", "Bare/Sparse Vegetation", "Snow and Ice", "Herbaceous Wetlands", "Mangroves"
    out_value : int
        DN you want as output in the mask, such as 1 or 10

    Returns
    -------
    permanent_water : TYPE
        raster mask with velue 0 and 1 (or the value you assigned as out_value)
    '''

    '''--------------READ INPUT DATA-----------------------------'''
    
    # Define thematic labels and corresponding DNs
    legend_dict = {
        "0": 0,
        "Tree": 10,
        "Shrubland": 20,
        "Grassland": 30,
        "Cropland": 40,
        "urban": 50,
        "Bare/Sparse Vegetation": 60,
        "Snow and Ice": 70,
        "water": 80,
        "Herbaceous Wetlands": 90,
        "Mangroves": 95,
        "ocean": 1,
        "slope": 5
    }
    
    # Define resampling algorithms
    resampling_methods = {
        'nearest': gdal.GRA_NearestNeighbour,
        'bilinear': gdal.GRA_Bilinear,
        'cubic': gdal.GRA_Cubic,
        'cubicspline': gdal.GRA_CubicSpline,
        'average': gdal.GRA_Average,
        'mode': gdal.GRA_Mode
    }


    mask_value = legend_dict[label_mask]
    resample_alg = resampling_methods[resample_method]
    
    #----->> SENTINEL IMAGE
    
    # Open the reference raster to get target parameters
    input_image_path = InputRefImagePath
    # Open reference raster (target grid)
    # read and open the reference image
    with rasterio.open(input_image_path) as src:
        wasdi.wasdiLog(f'\nreading image input: {os.path.basename(input_image_path)}')
        # with rasterio.open(os.path.join(data_folder, "S2_output.tif")) as src:
        src_bounds = src.bounds
        src_transform = src.transform
        src_width = src.width
        src_height = src.height
        src_resolution = src.res
        src_crs = src.crs
    # ------------------------------->> LC
    LC_path = input_LC_path

    # read the mosaic LC, clip and resample it to meet S2 extent and grid
    with rasterio.open(LC_path) as LC:
        wasdi.wasdiLog(f'\nreading thematic input: {os.path.basename(LC_path)}')
        LCover = LC.read(1)
        crs = LC.crs.to_wkt()  # Get CRS in WKT format
        transform = LC.transform  # Get the transform

    # Step 3: Create an in-memory GDAL dataset from the clipped `permanent_water` array
    mem_driver = gdal.GetDriverByName('MEM')
    in_mem_ds = mem_driver.Create('', LCover.shape[1], LCover.shape[0], 1,
                                      gdal.GDT_Byte)
    in_mem_ds.GetRasterBand(1).WriteArray(LCover)
    in_mem_ds.SetProjection(crs)
    in_mem_ds.SetGeoTransform(transform.to_gdal())  # Convert rasterio transform to GDAL transform

    # Step 4: Use `gdal.Warp` to resample the clipped dataset to match the `src` resolution and extent
    warped_mem_ds = gdal.Warp('',in_mem_ds, format='MEM',
                                  xRes=src_resolution[0],
                                  yRes=src_resolution[1],
                                  outputBounds=src_bounds,
                                  dstSRS=src_crs, resampleAlg=resample_alg)

    # Step 5: Read the result into a NumPy array
    reprojected_clipped_water = warped_mem_ds.GetRasterBand(1).ReadAsArray()

    permanent_water = np.zeros_like(reprojected_clipped_water, dtype=np.uint8)

    if label_mask == "slope":
        permanent_water[reprojected_clipped_water >= mask_value] = out_value
    else:
        permanent_water[reprojected_clipped_water == mask_value] = out_value

    wasdi.wasdiLog(f'\ncreating output: resampled and reprojected mask for --> {label_mask}')
    wasdi.wasdiLog(f'\noutput resolution: {warped_mem_ds.GetGeoTransform()[1]} px\noutput CRS: EPSG {warped_mem_ds.GetProjection()[-9:-2]}')

    # Clean up datasets
    in_mem_ds = None
    warped_mem_ds = None

    return permanent_water
