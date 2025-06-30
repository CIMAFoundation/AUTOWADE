# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:27:32 2024
@author: Francesca Trevisiol

__________________________________________________________
                AUTOWADE_S2
----------------------------------------------------------

This file contains all necessary functions to perform automatic flood mapping using Sentinel-2 L2A optical imagery,
ESA Land Cover (e.g., WorldCover), and auxiliary DEM-based water body mask (e.g., CopDEM-derived sea/ocean mask).
This AUTOWADE_S2 Python implementation is based on the optical workflow developed by CIMA Research Foundation [1],
and is based on spectral index (MNDWI), clustering, thresholding, and region growing.

The workflow includes the following main steps:

1. Data Preparation:
   - Reading input Sentinel-2 L2A image (bands: Red, Green, Blue, NIR, SWIR) and Scene Classification Layer (SCL)
   - Resampling thematic layers (permanent water and urban classes from Land Cover, ocean mask from DEM) to match image resolution
   - Masking clouds, urban areas, and ocean using SCL and thematic layers
   - Spectral index computation

2. Clustering and Thresholding:
   a. Unsupervised clustering on valid MNDWI values to identify water-like pixels
   b. Selection of target cluster(s) based on statistical ranking and user option (OPTION_1)
   c. Adaptive buffer size optimization to support robust thresholding (using edge-based method)
   d. Otsu thresholding to classify water vs non-water within the buffered region

3. Flood Water Refinement:
   - Combining permanent water edges and thresholded flood pixels
   - Executing seeded region growing on MNDWI using computed threshold and tolerance

4. Final Map Composition.

Main Output:
- A classified GeoTIFF flood map with the same spatial resolution of 20 m, where:
  - DN=0: Non-water
  - DN=1: Flooded area
  - DN=2: Permanent water (from Land Cover)

Reference:
[1] Pulvirenti, L., Squicciarino, G., & Fiori, E. (2020). A Method to Automatically Detect Changes in Multitemporal Spectral Indices: Application to Natural Disaster Damage Assessment. Remote Sensing, 12(17), 2681. https://doi.org/10.3390/rs12172681
[2] Stéfan van der Walt, Johannes L. Schönberger, Juan Nunez-Iglesias, François Boulogne, Joshua D. Warner, Neil Yager, Emmanuelle Gouillart, Tony Yu, and the scikit-image contributors. scikit-image: Image processing in Python. PeerJ 2:e453 (2014). https://doi.org/10.7717/peerj.453
[3] https://github.com/scikit-image/scikit-image/blob/main/skimage/filters/thresholding.py


"""

from resample_thematic import resample_thematic
import wasdi
import os
import rasterio
from glob import glob
import numpy as np
from osgeo import gdal
import zipfile
from scipy.ndimage import binary_dilation, binary_erosion
from scipy import ndimage 
from scipy.stats import norm, skew, kurtosis

import math

from sklearn.cluster import KMeans

from skimage.filters import threshold_otsu
from skimage._shared.utils import  warn
from skimage.exposure import histogram 

import time

from collections import deque

'''************************************ PARAMETER TO SET ******************************************************************************************'''


start_time_proce=time.time()

'''************************************ FUNCTION ******************************************************************************************'''

'''----------------------------------- Extract Band --------------------------------------------------------------------------'''

filter_size = 3

def extractBands(sFile, RES=20.0):  # RES: resolution of the output file (using bands at 20m resample to 20m)
    try:
        # Create output file names
        sOutputVrtFileBands = sFile.replace(".zip", "_bands.vrt")
        sOutputTiffFileBands = sFile.replace(".zip", "_bands.tif")
        sOutputVrtFileSCL = sFile.replace(".zip", "_SCL.vrt")
        sOutputTiffFileSCL = sFile.replace(".zip", "_SCL.tif")

        # Get the local file path of the input zip file
        sLocalFilePath = wasdi.getPath(sFile)

        # Log the local file path
        wasdi.wasdiLog(sLocalFilePath)

        # Get the local file paths for the output VRT and TIFF files
        sOutputVrtPathBands = wasdi.getPath(sOutputVrtFileBands)
        sOutputTiffPathBands = wasdi.getPath(sOutputTiffFileBands)
        sOutputVrtPathSCL = wasdi.getPath(sOutputVrtFileSCL)
        sOutputTiffPathSCL = wasdi.getPath(sOutputTiffFileSCL)

        # Log the output TIFF file paths
        wasdi.wasdiLog(sOutputTiffPathBands)
        wasdi.wasdiLog(sOutputTiffPathSCL)

        # Define the band names for Sentinel-2 Level-2A products
        asBandsJp2 = ['SCL_20m.jp2', 'B11_20m.jp2', 'B8A_20m.jp2', 'B04_20m.jp2', 'B03_20m.jp2', 'B02_20m.jp2']

        # Open the zip file and get the list of files inside it
        with zipfile.ZipFile(sLocalFilePath, 'r') as sZipFiles:
            asZipNameList = sZipFiles.namelist()

            # Log the list of files inside the zip
            #wasdi.wasdiLog(asZipNameList)

            # Filter the list to get the paths of the desired bands
            asBandsS2 = [name for name in asZipNameList for band in asBandsJp2 if band in name]
            asBandsZip = ['/vsizip/' + sLocalFilePath + '/' + band for band in asBandsS2]

            # Separate the paths for SCL and other bands
            asOrderedZipBands = []
            asOrderedZipSCL = []

            for sBand in ['SCL', 'B02', 'B03', 'B04', 'B8A', 'B11']:
                for sZipBand in asBandsZip:
                    if sBand in sZipBand:
                        if sBand == 'SCL':
                            asOrderedZipSCL.append(sZipBand)
                        else:
                            asOrderedZipBands.append(sZipBand)
                        break

            # Build VRT and translate to GeoTIFF for the bands
            gdal.BuildVRT(sOutputVrtPathBands, asOrderedZipBands, separate=True)
            gdal.Translate(sOutputTiffPathBands, sOutputVrtPathBands, xRes=RES, yRes=RES)
            os.remove(sOutputVrtPathBands)

            # Build VRT and translate to GeoTIFF for the SCL band
            gdal.BuildVRT(sOutputVrtPathSCL, asOrderedZipSCL, separate=True)
            gdal.Translate(sOutputTiffPathSCL, sOutputVrtPathSCL, xRes=RES, yRes=RES)
            os.remove(sOutputVrtPathSCL)

            # Return the paths to the output TIFF files
            return sOutputTiffFileBands, sOutputTiffFileSCL

    # If any exception occurs, log the exception message
    except Exception as oEx:
        wasdi.wasdiLog(f'extractBands EXCEPTION: {repr(oEx)}')

    # Return an empty string if an exception occurs
    return "", ""

'''----------------------------------- scale factor --------------------------------------------------------------------------'''

def mask_scale_factor(band, scale_factor, offset, no_data_value): #applico fattori correzione alle singole bande ed escludo no data
    band_mask = np.where(band == no_data_value, np.nan, band)
    band_mask = (band_mask + offset) * scale_factor
    band_mask = np.where(band_mask < 0, 0, band_mask)
    return band_mask


'''----------------------------------- index computation ---------------------------------------------------------------------'''
def compute_norm_diff (nir, red):
    ndvi = (nir - red) / (nir + red)
    ndvi = np.where(np.isfinite(ndvi), ndvi, np.nan)  # Handle NaN and infinite values
    return ndvi

'''----------------------------------- cluster implementation ----------------------------------------------------------------'''

def optimized_isodata_clustering2(data,n_clusters=6):
    start_time = time.time()
    
    valid_mask = np.ma.masked_invalid(data)
    valid_data = valid_mask.compressed().reshape(-1, 1)

    # Control that there are unmasked valid pixel in the image
    if valid_data.shape[0] == 0:
        wasdi.wasdiLog("No valid data available for clustering after masking. Skipping clustering.")
        return None, None, None

    kmeans = KMeans(n_clusters, init='random', n_init=1, max_iter=2, random_state=1)
    kmeans.fit(valid_data)
    #labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # Create an empty array to store the clustered image
    ndvi_clusters = np.full(data.shape, np.nan)

    # Assign cluster labels to the valid pixels
    ndvi_clusters[~valid_mask.mask] = kmeans.labels_

    return ndvi_clusters, centroids, n_clusters


'''----------------------------------- compute statistics --------------------------------------------------------------------'''
def compute_statistics(ndvi, clusters, n_clusters):
    """
    Compute statistics for NDVI values in each cluster.
    
    Parameters:
    ndvi (numpy.ndarray): NDVI array.
    clusters (numpy.ndarray): Clustered NDVI array.
    n_clusters (int): Number of clusters.
    
    Returns:
    dict: Dictionary of statistics for each cluster.
    """
    stats_dict = {}
    for cluster in range(n_clusters):
        cluster_mask = clusters == cluster
        cluster_values = ndvi[cluster_mask]
        
        # Filter out NaN values if any
        cluster_values = cluster_values[~np.isnan(cluster_values)]
        
        # Calculate statistics
        stats_dict[cluster] = {
            'mean': np.mean(cluster_values),
            'median': np.median(cluster_values),
            'std': np.std(cluster_values),
            'min': np.min(cluster_values),
            'max': np.max(cluster_values),
            'percentile_25': np.percentile(cluster_values, 25),
            'percentile_75': np.percentile(cluster_values, 75)
        }
    return stats_dict



'''----------------------------------- creating buffer from max median cluster -----------------------------------------------'''
def create_buffer (label, cluster, buffer_size): #CLUSTER is the label corresponding to the claster with the desired stat (es. max median)
    target_mask= label==cluster
    buffered_mask= binary_dilation(target_mask, iterations=buffer_size)
    return buffered_mask

'''----------------------------------- applying Roberts Edge filter -----------------------------------------------'''

def RobFilter(layer):
    
    #check that input is a compatible format for opencv
    layer=layer.astype(np.uint8)

    # Define the Robert's cross operator kernels
    roberts_cross_v = np.array([[1, 0], 
                                [0, -1]], dtype=int)
    
    roberts_cross_h = np.array([[0, 1], 
                                [-1, 0]], dtype=int)
    
    # Apply the vertical and horizontal Robert's cross
    vertical = ndimage.convolve( layer, roberts_cross_v ) 
    horizontal = ndimage.convolve( layer, roberts_cross_h )
    
    # Combine the two edges
    edged_img = np.sqrt( np.square(horizontal) + np.square(vertical)) 
    
    #print(f'Roberts edge filter \nMAX: {roberts_edge.max()} \nMIN: {roberts_edge.min()}')
    
    # Normalize the result
    roberts_edge = np.uint8(edged_img / edged_img.max() * 255)
    
    #create a mask where roberts edge is greater then 1 --> eq. 1 
    mask=np.where(roberts_edge==0, 0, 1)
    
    return mask


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
        
def _validate_image_histogram(image, hist, nbins=None, normalize=False):
    """Ensure that either image or hist were given, return valid histogram.

    If hist is given, image is ignored.

    Parameters
    ----------
    image : array or None
        Grayscale image.
    hist : array, 2-tuple of array, or None
        Histogram, either a 1D counts array, or an array of counts together
        with an array of bin centers.
    nbins : int, optional
        The number of bins with which to compute the histogram, if `hist` is
        None.
    normalize : bool
        If hist is not given, it will be computed by this function. This
        parameter determines whether the computed histogram is normalized
        (i.e. entries sum up to 1) or not.

    Returns
    -------
    counts : 1D array of float
        Each element is the number of pixels falling in each intensity bin.
    bin_centers : 1D array
        Each element is the value corresponding to the center of each intensity
        bin.

    Raises
    ------
    ValueError : if image and hist are both None
    """
    if image is None and hist is None:
        raise Exception("Either image or hist must be provided.")

    if hist is not None:
        if isinstance(hist, (tuple, list)):
            counts, bin_centers = hist
        else:
            counts = hist
            bin_centers = np.arange(counts.size)

        if counts[0] == 0 or counts[-1] == 0:
            # Trim histogram from both ends by removing starting and
            # ending zeroes as in histogram(..., source_range="image")
            cond = counts > 0
            start = np.argmax(cond)
            end = cond.size - np.argmax(cond[::-1])
            counts, bin_centers = counts[start:end], bin_centers[start:end]
    else:
        counts, bin_centers = histogram(
            image.reshape(-1), nbins, source_range='image', normalize=normalize
        )
    return counts.astype('float32', copy=False), bin_centers

        
def threshold_otsu_mod(image=None, nbins=256, *, hist=None):
    """Return threshold value based on Otsu's method.

    Either image or hist must be provided. If hist is provided, the actual
    histogram of the image is ignored.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray, optional
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    hist : array, or 2-tuple of arrays, optional
        Histogram from which to determine the threshold, and optionally a
        corresponding array of bin center intensities. If no hist provided,
        this function will compute it from the image.


    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.

    References
    ----------
    .. [1] Wikipedia, https://en.wikipedia.org/wiki/Otsu's_Method

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_otsu(image)
    >>> binary = image <= thresh

    Notes
    -----
    The input image must be grayscale.
    """
    if image is not None and image.ndim > 2 and image.shape[-1] in (3, 4):
        warn(
            f'threshold_otsu is expected to work correctly only for '
            f'grayscale images; image shape {image.shape} looks like '
            f'that of an RGB image.'
        )

    # Check if the image has more than one intensity value; if not, return that
    # value
    if image is not None:
        first_pixel = image.reshape(-1)[0]
        if np.all(image == first_pixel):
            return first_pixel

    counts, bin_centers = _validate_image_histogram(image, hist, nbins)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[idx]
    mean_1=mean1[idx]
    mean_2=mean2[idx]
    

    return threshold, mean_1, mean_2


def region_growing(image, seeds, value_threshold):
    """
    Perform region growing based on seeds, adding neighboring pixels if their value
    is greater than or equal to a specified threshold (tolerance).

    Parameters:
    - image: 2D numpy array, the original image (mndwi).
    - seeds: 2D numpy array of boolean, the seed pixels.
    - value_threshold: float, the minimum value for adding pixels to the region.

    Returns:
    - grown_region: 2D numpy array of boolean, the grown region mask.
    """
    start_time = time.time()

    seeds_bool = seeds == 1

    # Initialize the region mask with seeds
    grown_region = np.zeros_like(image, dtype=bool)
    grown_region[seeds_bool] = True

    # Create a queue for the region growing process
    queue = deque(np.argwhere(seeds))

    # Define neighbor connectivity (4-connectivity)
    # neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Define neighbor connectivity (8-connectivity)
    neighbors = [(-1, 0), (-1, -1), (1, 0), (1, 1), (-1, 1), (0, -1), (0, 1), (1, -1)]

    while queue:
        y, x = queue.popleft()

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx

            # Ensure we are within image bounds
            if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:
                # Check if neighboring pixel is not already in the region
                # and satisfies the value threshold
                if not grown_region[ny, nx] and image[ny, nx] >= value_threshold:
                    grown_region[ny, nx] = True
                    queue.append((ny, nx))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Region growing completed in {elapsed_time:.2f} seconds.")

    return grown_region
        
'''---------------------------- Bimodality check & buffer --------'''
def BM_check_buffer(target_mask, MNDWI, permanent_water):
    
    buff_size=[4,6,8,10,12,15,20]
    for i in buff_size:
        
        buffered= binary_dilation(target_mask, iterations=i)
    
        #trovo i valori validi in corrispondenza della maschera che ho calcolato
        buff_ok=np.full(MNDWI.shape, np.nan)
        buff_ok[buffered] = MNDWI[buffered]
               
        #calcolo array mascherato in corrispondenza del cluster, del buffer e della loro differenza
        masked_arr = np.ma.masked_array(MNDWI, mask=~buffered) #cluster + buffer
        valid_1=masked_arr.compressed()
        
        T_masked_buffer= np.ma.masked_array(masked_arr, mask=target_mask) # buffer - cluster
        T_masked_cluster= np.ma.masked_array(masked_arr, mask=~target_mask) # cluster
        
        #------TEST 1:  BC - BIMODALITY COEFFICIENT TEST
        print('\n*****BC - BIMODALITY COEFFICIENT TEST*****')
        gamma_val=skew(valid_1)
        k_val=kurtosis(valid_1)
        p_val=float(len(valid_1))
        BC=float(gamma_val**2 +1)/float((k_val)+((3*(p_val-1)**2)/((p_val-2)*(p_val-3))))
        print(f"\nbimodality coefficient: {BC:.4f}")
        
        
        #------TEST 2:  d - Ashman's D TEST
        print('\n*****Ashmans D TEST*****')
        # option1: apply otsu and gather stats from it
        print('-----With stats based on otsu')
        T_thresh, T_m1, T_m2 = threshold_otsu_mod(valid_1)
        class_1=valid_1[valid_1<T_thresh]
        class_2=valid_1[valid_1>T_thresh]
        Ad_otsu=math.sqrt(2*(abs(class_1.mean()-class_2.mean()))/math.sqrt(class_1.std()**2+class_2.std()**2))
        print(f"Ashman's D coefficient (using otsu)': {Ad_otsu:.2f}")
        
        #verify class_1 (no water) population is at least 10% of class_2 (water)
        #if len(class_1)<len(class_2):
        pop_pecent_ot=len(class_1)*100/len(class_2)
        print(f"class_1 (no water) population is {pop_pecent_ot:.2f}% of class_2 (water)")
        
        # option2: gather stats from cluster and (buffer-cluster) pixels
        print('\n-----With stats based on cluster and buffer')
        #stat from (buffer-cluster)
        mean_1 = T_masked_buffer.compressed().mean() 
        std_1 = T_masked_buffer.compressed().std()
        #stat from cluster
        mean_2 = T_masked_cluster.compressed().mean() #from cluster
        std_2 = T_masked_cluster.compressed().std() 
        
        Ad=math.sqrt(2*(abs(mean_1-mean_2))/math.sqrt(std_1**2+std_2**2))
        print(f"Ashman's D coefficient': {Ad:.2f}")
        
        #verify class_1 (no water) population is at least 10% of class_2 (water)
        if len(T_masked_buffer.compressed())<len(T_masked_cluster.compressed()):
            pop_pecent=len(T_masked_buffer.compressed())*100/len(T_masked_cluster.compressed())
            print(f"class_1 (no water) population is {pop_pecent:.2f}% of class_2 (water)")

        # Check if the BC exceeds the threshold
        if (BC > 0.55) and (Ad_otsu > 2) and (pop_pecent_ot > 10):
        #if (BC > 0.55) and ((Ad_otsu > 2) or (Ad > 2)) and ((pop_pecent_ot>10) or (pop_pecent>5)):
            print(f"\nBuffer size {i} meets the bimodality criterion with BC={BC:.2f}")
            print(f"\nBuffer size {i} meets the bimodality criterion with Ashamn's D={Ad_otsu:.2f}")
            print(f"\nBuffer size {i} meets the bimodality criterion with Population of non water class beeing = {pop_pecent_ot:.2f}% of water class\n")
            return buffered, buff_ok, i  # Stop and return if the criterion is met

        # If the first loop does not meet the condition, move to the second set of buffer sizes
        print("No buffer size met the bimodality coefficient threshold. Proceeding to secondary check.")

        # Second set of buffer sizes
        buff_size_2 = [5, 10, 15, 20, 25, 35, 40]
        intersection = (target_mask == 1) & (permanent_water == 1)  # Compute intersection

        for i in buff_size_2:
            # Placeholder for additional processing
            buffered = binary_dilation(intersection, iterations=i)

            # trovo i valori validi in corrispondenza della maschera che ho calcolato
            buff_ok = np.full(MNDWI.shape, np.nan)
            buff_ok[buffered] = MNDWI[buffered]

            # calcolo array mascherato in corrispondenza del cluster, del buffer e della loro differenza
            masked_arr = np.ma.masked_array(MNDWI, mask=~buffered)  # cluster + buffer
            valid_1 = masked_arr.compressed()

            T_masked_buffer = np.ma.masked_array(masked_arr, mask=intersection)  # buffer - intersection
            T_masked_cluster = np.ma.masked_array(masked_arr, mask=~intersection)  # intersection

            # ------TEST 1:  BC - BIMODALITY COEFFICIENT TEST
            print('\n*****BC - BIMODALITY COEFFICIENT TEST*****')
            gamma_val = skew(valid_1)
            k_val = kurtosis(valid_1)
            p_val = float(len(valid_1))
            BC = float(gamma_val ** 2 + 1) / float((k_val) + ((3 * (p_val - 1) ** 2) / ((p_val - 2) * (p_val - 3))))
            print(f"\nbimodality coefficient: {BC:.4f}")

            # ------TEST 2:  d - Ashman's D TEST
            print('\n*****Ashmans D TEST*****')
            # option1: apply otsu and gather stats from it
            print('\n-----With stats based on otsu')
            T_thresh, T_m1, T_m2 = threshold_otsu_mod(valid_1)
            class_1 = valid_1[valid_1 < T_thresh]
            class_2 = valid_1[valid_1 > T_thresh]
            Ad_otsu = math.sqrt(
                2 * (abs(class_1.mean() - class_2.mean())) / math.sqrt(class_1.std() ** 2 + class_2.std() ** 2))
            print(f"Ashman's D coefficient (using otsu)': {Ad_otsu:.2f}")

            # verify class_1 (no water) population is at least 10% of class_2 (water)
            # if len(class_1)<len(class_2):
            pop_pecent_ot = len(class_1) * 100 / len(class_2)
            print(f"class_1 (no water) population is {pop_pecent_ot:.2f}% of class_2 (water)")

            # option2: gather stats from cluster and (buffer-cluster) pixels
            print('\n-----With stats based on cluster and buffer')
            # stat from (buffer-cluster)
            mean_1 = T_masked_buffer.compressed().mean()
            std_1 = T_masked_buffer.compressed().std()
            # stat from cluster
            mean_2 = T_masked_cluster.compressed().mean()  # from cluster
            std_2 = T_masked_cluster.compressed().std()

            Ad = math.sqrt(2 * (abs(mean_1 - mean_2)) / math.sqrt(std_1 ** 2 + std_2 ** 2))
            print(f"Ashman's D coefficient': {Ad:.2f}")

            # verify class_1 (no water) population is at least 10% of class_2 (water)
            if len(T_masked_buffer.compressed()) < len(T_masked_cluster.compressed()):
                pop_pecent = len(T_masked_buffer.compressed()) * 100 / len(T_masked_cluster.compressed())
                print(f"class_1 (no water) population is {pop_pecent:.2f}% of class_2 (water)")

                # Check if the BC exceeds the threshold
            if (BC > 0.55) and (Ad_otsu > 2) and (pop_pecent_ot > 10):
                print(f"\nBuffer size {i} meets the bimodality criterion with BC={BC:.2f}")
                print(f"\nBuffer size {i} meets the bimodality criterion with Ashamn's D={Ad_otsu:.2f}")
                print(f"\nBuffer size {i} meets the bimodality criterion with Population of non water class beeing = {pop_pecent_ot:.2f}% of water class\n")
                return buffered, buff_ok, i  # Stop and return if the criterion is met

    # If no buffer size meets the criterion, return None or another indicator
    print("No buffer size met the bimodality coefficient threshold.")
    return None


def AUTOWADE_S2 (ImageToProcess, LC_path, DEM_auxWBM_path, OPTION_1, buff_size2, scale_factor, offset, no_data_value, output_path):
    '''************************************ IMPORT DATA ***************************************************************************************'''
    # Read bands, apply scale factor, mask out clouds using QA

    sTiffFile, SCL_band = extractBands(ImageToProcess)

    with rasterio.open(wasdi.getPath(sTiffFile)) as src:
        with rasterio.open(wasdi.getPath(SCL_band)) as scl_src:
            # Read the Red, Green, and Blue bands (assuming B4, B3, B2 for Landsat-8/9)
            # apply scale and offset
            red_band = src.read(1).astype(np.float32)  # B4
            green_band = src.read(2).astype(np.float32)  # B3
            blue_band = src.read(3).astype(np.float32)  # B2
            nir_band = src.read(4).astype(np.float32)  # B8
            swir_band = src.read(5).astype(np.float32)  # B11

            wasdi.wasdiLog(f"RED min: {np.nanmin(red_band)}, max: {np.nanmax(red_band)}")

            qa_band = scl_src.read(1)   #SCL

            # Get the transform and CRS from the source dataset
            transform = src.transform
            crs = src.crs

            # Get the transform and CRS from the SCL dataset
            transform_scl = scl_src.transform
            crs_scl = scl_src.crs
    # Check if the transforms and CRS are consistent
    if transform != transform_scl or crs != crs_scl:
        wasdi.wasdiLog("Warning: Transforms or CRS do not match between SCL and bands.")

    '''************************************ LAND COVER - URBAN PERMANENT WATER ******************************************************************'''

    # extract reference water extent and urban from LC by ESA
    water_from_LC = resample_thematic(wasdi.getPath(sTiffFile), LC_path, 'mode', 'water', 1)
    urban_from_LC = resample_thematic(wasdi.getPath(sTiffFile), LC_path, 'mode', 'urban', 1)
    sea_mask_from_dem = resample_thematic(wasdi.getPath(sTiffFile), DEM_auxWBM_path, 'mode', 'ocean', 1)

    '''----------------------------SCALE FACTOR and MASKING-------------------------------------------------------------'''

    # List of bands
    bands = [red_band, green_band, blue_band, nir_band, swir_band]

    wasdi.wasdiLog(f"Starting preprocessing - clouds & urban masking")

    # Create masks for Cirrus (10) , Clouds High Probability (9), Clouds Medium Probability (8), Cloud Shadows (3)
    cloud_mask = (qa_band == 10) | (qa_band == 9) | (qa_band == 8) | (qa_band == 3)
    # Buffer to the cloud mask
    cloud_mask_buffered = binary_dilation(cloud_mask, iterations=4)

    # Create mask for urban areas
    urban_mask = (urban_from_LC == 1)

    # Create mask for ocean
    sea_mask = (sea_mask_from_dem == 1)
    # Reduce the sea mask
    ocean_mask = binary_erosion(sea_mask, iterations=3)

    # Combine the cloud mask and urban mask
    final_mask = cloud_mask_buffered | urban_mask | ocean_mask

    # Apply scale, mask, and mask invalid values to each band
    for i in range(len(bands)):
        bands[i] = mask_scale_factor(bands[i], scale_factor, offset, no_data_value)  # Apply scale factor and offset
        bands[i][final_mask] = np.nan  # Apply cloud & urban  mask
        bands[i] = np.ma.masked_invalid(bands[i])  # Mask invalid values (NaN)

    # Unpack back to original variable names
    red_band, green_band, blue_band, nir_band, swir_band = bands

    '''************************************ COMPUTE INDEX ***********************************************'''
    #compute index and mask out not valid values

    MNDWI = compute_norm_diff(green_band, swir_band) #
    print(f"MNDWI min: {np.nanmin(MNDWI)}, max: {np.nanmax(MNDWI)}")
    #Mask out of range values
    mask_mNDWI = np.isfinite(MNDWI) & (MNDWI >= -1.0) & (MNDWI <= 1.0)
    MNDWI[~mask_mNDWI ] = np.nan
    MNDWI = np.ma.masked_invalid(MNDWI)

    print(f"MNDWI min: {np.nanmin(MNDWI)}, max: {np.nanmax(MNDWI)}")

    '''************************************ CLUSTERING and STATISTICS ***********************************************'''

    # Apply optimized ISODATA clustering
    labels, centroids, opt_clusters = optimized_isodata_clustering2(MNDWI)

    if labels is None or centroids is None or opt_clusters is None:
        wasdi.wasdiLog("Clustering failed due to insufficient valid data (area maske due to slope or urban). Exiting AUTOWADE_S2 early.")
        return

    #compute statistics
    Mndwi_stat = compute_statistics(MNDWI, labels, opt_clusters)
    #sort clusters by median value
    a=sorted(Mndwi_stat, key=lambda k: Mndwi_stat[k]['median'])

    #Identifiy clusters of interest with high value of MNDWI

    if OPTION_1 and ((Mndwi_stat[a[-2]]['percentile_25'])>0) :
        print('option 1 set - the two clusters with the highest median have positive median')
        print('selecting the two clusters with the highest MNDWI median')
        #get the two clusters with the highest median
        target_mask = (labels == a[-1])|(labels == a[-2])
    else:
        # Identify the cluster with the max median MNDWI
        print('selecting the max median cluster')
        max_median_cluster = a[-1] #MAX MEDIAN CLUSTER
        #create mask for values corresponding to the max median cluster
        target_mask = labels == max_median_cluster
        print(f'max median cluster:{ max_median_cluster}')

    MNDWI_flood = np.full(MNDWI.shape, np.nan)
    # Mask out MDWI values that do not belong to the identified cluster
    MNDWI_flood[target_mask] = MNDWI[target_mask]

    '''************************************ BUFFER ************************************************************************'''
    #check buffer size to achieve bimodal distribution, starting from the max median cluster
    buffered, buff_ok, buff_size = BM_check_buffer(target_mask, MNDWI, water_from_LC)

    '''************************************ THRESHOLDING *********************************************************************'''

    masked_flood= np.ma.masked_array(MNDWI, mask=~buffered)
    valid_data = masked_flood.compressed() #get valid data in a 2D array
    thresh = threshold_otsu(valid_data)

    print (f"Otsu threshold: {thresh: .4f}")

    #apply threshold
    water_mask = np.zeros(MNDWI.shape)
    water_mask[~masked_flood.mask] = np.where(masked_flood[~masked_flood.mask] >= thresh, 1, 0)

    '''************************************ CREATE THE MCDWA  ******************************************************************'''

    #compute land\water edges with Roberts filter
    edge=RobFilter(water_mask)

    #union of edges and the perm_water
    union=edge+water_mask
    p_water_edge = np.where(union>=1, 1, 0)

    #create ----->  BUFFER2
    buffered_mask = binary_dilation(p_water_edge, iterations=buff_size2)

    #masking MNDWI with the MCDWA mask + buffer
    MNDWI_pWATER = np.full(MNDWI.shape, np.nan)
    MNDWI_pWATER = np.where(buffered_mask==1, MNDWI, np.nan)

    P_masked_flood= np.ma.masked_array(MNDWI, mask=~buffered_mask)

    P_valid_data = P_masked_flood.compressed() #get valid data in a 2D array
    P_thresh = threshold_otsu(P_valid_data)

    print (f"Otsu threshold, 2: {P_thresh: .4f}")

    #apply threshold
    P_water_mask = np.full(MNDWI.shape, np.nan)
    P_water_mask = np.where(MNDWI_pWATER >= P_thresh, 1, 0)

    # Step 2: Filter the data to include only values above the threshold
    filtered_data = MNDWI_pWATER[MNDWI_pWATER > P_thresh]

    # Step 3: Fit a Gaussian distribution to the filtered data
    mean, std_dev = norm.fit(filtered_data)

    print(f'mean: {mean: .4f}; \nstd dev:{std_dev: .4f}')

    '''----------------------------------------- REGION GROWING --------------------------------'''

    # Assign seed and tolerance values
    seed_threshold = thresh#0.5 * (P_thresh + mean)
    tolerance = P_thresh #2*std_dev#

    wasdi.wasdiLog(f"Mean: {mean:.4f}, Std Dev: {std_dev:.4f}, Tolerance: {tolerance:.4f}")
    #create seeds mask
    seeds = MNDWI_pWATER >= seed_threshold

    if seed_threshold > tolerance:
        # median filter to remove isolated pixels
        seeds_filt = ndimage.median_filter(seeds, size=filter_size)
        grown_region = region_growing(MNDWI, seeds_filt, tolerance)
        wasdi.wasdiLog(f"Seed Threshold: {seed_threshold:.4f}, Tolerance: {tolerance:.4f} ")
    else:
        seeds_tol = MNDWI_pWATER >= tolerance
        seeds_filt = ndimage.median_filter(seeds_tol, size=filter_size)
        grown_region = region_growing(MNDWI, seeds_filt, seed_threshold)
        wasdi.wasdiLog(f"Seed (tol) Threshold: {tolerance:.4f}, Tolerance: {seed_threshold:.4f} ")

    '''************************************ CREATE THE FINAL MAP  ******************************************************************'''
    # assign 1 to flood - 2 to permanent water - Intersection of permanent water from reference and water mask derived from S2 clustering
    all_water = np.where((grown_region == 1) & (water_from_LC == 1), 2,  # Condition 1: Both are 1 → PERMANENT WATER
                np.where((grown_region == 1) & (water_from_LC == 0), 1, # Condition 2: Only grown_region is 1 → FLOOD
                np.where((grown_region == 0) & (water_from_LC == 1), 0, # Condition 3: Only water_from_LC is 1 → it means it's permanent water from reference but not water in the image
                                           0))).astype(np.uint8)  # Else → 0
    end_time_proce=time.time()
    wasdi.wasdiLog(f"\nProcess completed in {(end_time_proce - start_time_proce)/60:.2f} minutes.")


    wasdi.wasdiLog("Writing output")
    write_geotiff(all_water, transform, crs, output_path)


        
