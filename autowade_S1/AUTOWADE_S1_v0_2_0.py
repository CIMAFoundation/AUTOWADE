# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:14:45 2024
Last update 10.06.2025
@author: Francesca Trevisiol

__________________________________________________________
                AUTOWADE_S1 v0.2.0
----------------------------------------------------------

This file contains all necessary functions to performs automatic flood mapping using pre- and post-event Sentinel-1 GRD imagery,
Land Cover (WorldCover), and DEM data (CopDEM). This autowade_S1 python implementation is based on Pulvirenti et al. (2021) [1].
The workflow includes the following main steps:

1. Data Preparation:
   - Reading input files (pre- and post-event image, WorldCover)
   - Extracting Copernicus DEM over the footprint of the input images (with wasdi dem_extractor function) and computing slope
   - Cropping, Resampling, Reprojecting thematic layers (urban areas and permanent water from WorldCover, slope) to meet the post event grid and extent
   - Masking out urban and steep areas

2. Post-Event Image Processing:
   a. Clustering to extract "first guess" water class
   b. Iterative bimodality check to determine optimal buffer size for edge-based thresholding (using functions in [2][3])
   c. Thresholding: Automatic classification of water/non-water using Otsu on median cluster + buffer,
      including statistical analysis for defining seeds and tolerance values
   d. Region growing to expand water areas using previously computed seeds and tolerance

3. Post-Pre Difference Analysis:
   - Repetition of the above steps (clustering, thresholding, region growing) on the delta (post-pre) image

4. Final Flood Map Composition:
   - Combination of permanent water, flood water, and refinement using buffers and contextual rules

Main Output:
- A flood map GeoTIFF, with a spatial resolution of 20 meters, classified as:
  - DN=0: Maked
  - DN=1: No flood
  - DN=2: Permanent water
  - DN=3: Flood water
The minimum mapping unit is 10 hectares.

Reference:
[1] Pulvirenti, L., Squicciarino, G., Fiori, E., Ferraris, L., & Puca, S. (2021). *A Tool for Pre-Operational Daily Mapping of Floods and Permanent Water Using Sentinel-1 Data*. Remote Sensing, 13(7), 1342.
https://doi.org/10.3390/rs13071342
[2] Stéfan van der Walt, Johannes L. Schönberger, Juan Nunez-Iglesias, François Boulogne, Joshua D. Warner, Neil Yager, Emmanuelle Gouillart, Tony Yu, and the scikit-image contributors. scikit-image: Image processing in Python. PeerJ 2:e453 (2014) https://doi.org/10.7717/peerj.453
[3] https://github.com/scikit-image/scikit-image/blob/main/skimage/filters/thresholding.py
"""

from resample_thematic import resample_thematic
import wasdi
import os
import scipy.ndimage as ndi
from rasterio.warp import transform
from scipy import ndimage
from sklearn.cluster import KMeans
from scipy.ndimage import binary_dilation
from scipy.stats import norm, skew, kurtosis
import time
from collections import deque
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from osgeo import gdal
import math
from skimage.filters import threshold_otsu
from skimage._shared.utils import warn
from skimage.exposure import histogram 


'''************************************FUNCTIONs*********************************************************************************************'''

def compute_statistics(index, clusters, n_clusters):
    """
    Compute statistics for backscatter values in each cluster.
    
    Parameters:
    index (numpy.ndarray): image.
    clusters (numpy.ndarray): Clustered VV array.
    n_clusters (int): Number of clusters.
    
    Returns:
    dict: Dictionary of statistics for each cluster.
    """
    stats_dict = {}
    for cluster in range(n_clusters):
        cluster_mask = clusters == cluster
        cluster_values = index[cluster_mask]
        
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

def region_growing(image, seeds, value_threshold):
    """
    Perform region growing based on seeds, adding neighboring pixels if their value
    is lower than or equal to a specified threshold (tolerance).
    
    Parameters:
    - image: 2D numpy array, the original image.
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
                if not grown_region[ny, nx] and image[ny, nx] <= value_threshold:
                    grown_region[ny, nx] = True
                    queue.append((ny, nx))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Region growing completed in {elapsed_time:.2f} seconds.")

    return grown_region

def clustering (VV_post, combined_mask, min_clust_num, count_pixel_pw):

    index = VV_post
    
    start_time = time.time()
    
    #Mask out of range values
    mask_mNDWI = np.isfinite(index) & (index >= -30) & (index <=  10)
    index[~mask_mNDWI] = np.nan
    index = np.ma.masked_invalid(index)
   
    #mask out urban pixels and slope higher then a certain threshold
    
    index_masked = np.ma.masked_array(index, mask=(combined_mask))
    
    print(f"Cluster DN values min: {np.nanmin(index_masked)}, max: {np.nanmax(index_masked)}")
    print(index_masked.shape)
    
    valid_mask = np.ma.masked_invalid(index_masked)
    valid_data = valid_mask.compressed().reshape(-1, 1)
    
    # Apply optimized ISODATA clustering
    kmeans = KMeans(n_clusters=min_clust_num, init='random', n_init=1, max_iter=2, random_state=1).fit(valid_data)

    end_time = time.time()
    print(f"Clustering completed in {(end_time - start_time)/60:.2f} minutes.")
    
    # Create an empty array to store the clustered image
    vv_clusters = np.full(VV_post.shape, np.nan)
    # Assign cluster labels to the valid pixels
    vv_clusters[~valid_mask.mask] = kmeans.labels_
    
    Mndwi_stat = compute_statistics(VV_post, vv_clusters, min_clust_num)
    #sort clusters by median value
    a = sorted(Mndwi_stat, key=lambda k: Mndwi_stat[k]['median'])
        
    #Identifiy clusters of interest with high value of MNDWI
    
    # Identify the cluster with the min median backscatter
    print('selecting the min median cluster')
    MIN_median_cluster = a[0] # MIN MEDIAN CLUSTER
    # Create mask for values corresponding to the MIN median cluster
    target_mask = vv_clusters == MIN_median_cluster
        
    index_water = np.full(index_masked.shape, np.nan)
    # Mask out backscatter values that do not belong to the identified cluster
    index_water[target_mask] = index_masked[target_mask]
    return index_water, target_mask

def clustering_2 (index, combined_mask, min_clust_num):

    index
    
    start_time = time.time()
    #Mask out of range values
    mask_mNDWI = np.isfinite(index) & (index >= -30) & (index <=  10)
    index[~mask_mNDWI ] = np.nan
    index = np.ma.masked_invalid(index)
   
    #mask out urban pixels and slope higher then a certain threshold
    
    MNDWI= np.ma.masked_array(index, mask=(combined_mask))
    
    # print(f"Cluster DN values min: {np.nanmin(MNDWI)}, max: {np.nanmax(MNDWI)}")
    # print(MNDWI.shape)
    
    valid_mask = np.ma.masked_invalid(MNDWI)
    valid_data = valid_mask.compressed().reshape(-1, 1)

    # Control that there are unmasked valid pixel in the image
    if valid_data.shape[0] == 0:
        wasdi.wasdiLog("No valid data available for clustering after masking. Skipping clustering.")
        return None, None

    # Apply optimized ISODATA clustering
    kmeans = KMeans(n_clusters=min_clust_num, init='random', n_init=1, max_iter=2, random_state=1).fit(valid_data)

    end_time = time.time()
    wasdi.wasdiLog(f"Clustering completed in {(end_time - start_time)/60:.2f} minutes.")
    
    # Create an empty array to store the clustered image
    ndvi_clusters = np.full(index.shape, np.nan)
    # Assign cluster labels to the valid pixels
    ndvi_clusters[~valid_mask.mask] = kmeans.labels_
    
    # Count valid pixels per cluster
    valid_cluster_pixels = ndvi_clusters[~np.isnan(ndvi_clusters)].astype(int)
    unique, counts = np.unique(valid_cluster_pixels, return_counts=True)
    
    # print("Pixel count per cluster:")
    # for cluster_id, count in zip(unique, counts):
    #     print(f"Cluster {cluster_id}: {count} valid pixels")
    
    Mndwi_stat = compute_statistics(index, ndvi_clusters, min_clust_num)
    #sort clusters by median value
    a=sorted(Mndwi_stat, key=lambda k: Mndwi_stat[k]['median'])
    
    # Identify the cluster with the min median backscatter
    print('selecting the min median cluster')

    # Count pixels per cluster
    cluster_counts = dict(zip(unique, counts))
    
    # Select cluster with minimum median
    MIN_median_cluster = a[0]
    min_count = cluster_counts.get(MIN_median_cluster, 0)
    
    # Threshold for valid pixel count
    min_pixel_threshold = 1000  # <-- Set your own threshold here
    
    # Decide if one or two clusters to use
    if min_count < min_pixel_threshold:
        wasdi.wasdiLog(f"Cluster {MIN_median_cluster} has only {min_count} valid pixels < {min_pixel_threshold}. Including second min-median cluster.")
        SECOND_median_cluster = a[1]
        target_mask = np.logical_or(
            ndvi_clusters == MIN_median_cluster,
            ndvi_clusters == SECOND_median_cluster
        )
    else:
        target_mask = ndvi_clusters == MIN_median_cluster
    
    print(f"Selected cluster(s): {MIN_median_cluster}" + (f", {SECOND_median_cluster}" if min_count < min_pixel_threshold else ""))    
    vv_flood = np.full(MNDWI.shape, np.nan)
    # Mask out MDWI values that do not belong to the identified cluster
    vv_flood[target_mask] = MNDWI[target_mask]
    return vv_flood, target_mask

def RobFilter(layer):
    #check that input is a compatible format for opencv
    layer = layer.astype(np.uint8)
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
    # Normalize the result
    roberts_edge = np.uint8(edged_img / edged_img.max() * 255)
    
    # create a mask where roberts edge is greater then 1 --> eq. 1
    mask = np.where(roberts_edge == 0, 0, 1)
    
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

# --- Functions for slope computation

def get_utm_crs_from_dem(dem_path):
    """Detect the correct UTM zone from DEM center coordinates"""
    with rasterio.open(dem_path) as src:
        wasdi.wasdiLog(src.crs)
        bounds = src.bounds
        lon_center = (bounds.left + bounds.right) / 2
        lat_center = (bounds.top + bounds.bottom) / 2

    utm_zone = int((lon_center + 180) / 6) + 1
    if lat_center >= 0:
        epsg_code = 32600 + utm_zone  # Northern hemisphere
    else:
        epsg_code = 32700 + utm_zone  # Southern hemisphere

    return f"EPSG:{epsg_code}"


def reproject_raster_gdal(src_path, dst_crs):
    """Reproject a raster to a new CRS using GDAL's Warp."""

    # Open the source raster with GDAL
    src_ds = gdal.Open(src_path)

    # If dst_crs is a string (e.g., 'EPSG:4326'), use it directly.
    if isinstance(dst_crs, str):
        dst_srs = dst_crs
        wasdi.wasdiLog("dst_crs is the following string:")
        wasdi.wasdiLog(dst_srs)
    else:
        # If dst_crs is a pyproj CRS object, convert it to WKT
        dst_srs = dst_crs.to_wkt() if hasattr(dst_crs, 'to_wkt') else dst_crs

    # Define the Warp options as a dictionary
    options = gdal.WarpOptions(
        dstSRS=dst_srs,  # Destination CRS (can be EPSG code or WKT)
        format='MEM',  # Store the result in memory
        resampleAlg=gdal.GRA_Bilinear,  # Resampling algorithm (bilinear in this case)
    )

    # Perform the reproject operation using GDAL's Warp
    reprojected_ds = gdal.Warp('', src_ds, options=options)

    # Close the source dataset
    src_ds = None

    print(f"Raster reprojected to {dst_crs} and stored in memory")

    return reprojected_ds  # Return the in-memory dataset


def compute_slope_gdal(input_dem_path, output_slope_path):
    """Compute slope from DEM using GDAL DEMProcessing."""
    gdal.DEMProcessing(
        output_slope_path,
        input_dem_path,
        'slope',
        computeEdges=True,
        slopeFormat='degree'  # Output slope in degrees
    )

def SLOPE_COMPUTATION(input_dem, output_tmp_path):
    """ Computes terrain slope from a DEM in EPSG:4326!!.

    The DEM is reprojected to its appropriate UTM zone, and slope is calculated using GDAL.
    The result is saved as 'tmp_slope_utm.tif' in the same folder as `output_tmp_path`.

    Parameters:
    :param input_dem (str): Path to input DEM (CopDEM) in lat/lon EPSG:4326.
    :param output_tmp_path (str): Path to define the output directory.

    :return:
    - str: Path to the computed slope GeoTIFF (in UTM).
    """

    folder_path = os.path.dirname(output_tmp_path)
    slope_utm = os.path.join(folder_path, "tmp_slope_utm.tif") # Temporary slope in UTM

    # Step 1: Detect UTM CRS
    utm_crs = get_utm_crs_from_dem(input_dem)
    wasdi.wasdiLog(f"Selected UTM CRS: {utm_crs}")

    # Step 2: Reproject DEM to UTM
    reprojected_dem = reproject_raster_gdal(input_dem, utm_crs)

    # Step 3: Compute slope using GDAL
    compute_slope_gdal(reprojected_dem, slope_utm)
    wasdi.wasdiLog("slope computed")

    # Step 4: Cleanup temporary files
    # os.remove(tmp_slope_utm)

    print(f"Final slope layer saved as: {slope_utm}")
    return slope_utm

def _validate_image_histogram(image, hist, nbins=None, normalize=False):
    """
    Source: https://github.com/scikit-image/scikit-image/blob/main/skimage/filters/thresholding.py

    Ensure that either image or hist were given, return valid histogram.


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
    """
    Modified from https://github.com/scikit-image/scikit-image/blob/main/skimage/filters/thresholding.py
    ---
    Return threshold value based on Otsu's method.

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
        
'''---------------------------- Bimodality check & buffer --------'''

def BM_check_buffer_2(target_mask, MNDWI, buffer_option, safe_thresh=-14, max_pop_percent_ot=150):
    """
    Search for the optimal buffer size around a binary mask (target_mask from clustering) to evaluate
    local bimodality in VV_img values and determine a suitable threshold for water detection.

    This function iteratively tests different buffer sizes around the input mask and evaluates
    the statistical separation of VV_img values in the buffered region using Ashman's D, and a modified Otsu threshold. It returns the configuration
    that best satisfies bimodality and distributional separation criteria.
    :param target_mask: np.ndarray (bool or binary)
        Binary mask of the initial target area (e.g., flood candidate region).
    :param MNDWI: np.ndarray (float)
        Original image to classify - e.g. VV_post
    :param buffer_option: str
        Defines the range of buffer sizes to test:
        - If 'post': uses np.arange(10, 100, 5)
        - If 'delta': uses np.arange(1, 50, 1)
    :param safe_thresh: float, optional (default=-14)
        Safe minimum threshold value for water; used to correct overestimated thresholds
        from the Otsu algorithm if necessary
    :param max_pop_percent_ot: float, optional (default=150)
        Maximum allowable ratio (%) between the population of higher and lower Otsu classes
        for a configuration to be considered valid.
    :return:
        tuple or None
        If a valid buffer configuration is found, returns:
        (buffered_mask, valid_MNDWI_values, buffer_size, threshold, mean_below_thresh, std_below_thresh)

        - buffered_mask : np.ndarray (bool)
            Binary mask representing the optimal buffered region.
        - valid_MNDWI_values : np.ndarray
            MNDWI values within the optimal buffer region.
        - buffer_size : int
            Number of iterations (pixels) used for binary dilation of the original mask.
        - threshold : float
            Otsu-derived (or adjusted) threshold for separating water from non-water.
        - mean_below_thresh : float
            Mean of MNDWI values below the threshold (assumed to be water).
        - std_below_thresh : float
            Standard deviation of MNDWI values below the threshold.

        Returns None if no valid buffer configuration is found.
    """

    if isinstance(buffer_option, str):
        if buffer_option == "post":
            buff_size = np.arange(10, 100, 5)
        elif buffer_option == "delta":
            buff_size = np.arange(1, 50, 1)
        else:
            raise ValueError("Unknown buffer_option string. Use 'post', 'delta', or pass a custom array.")
    else:
        buff_size = np.array(buffer_option)

    best_score = -np.inf
    best_result = None

    def check_buffer(i):
        nonlocal best_score, best_result

        buffered = binary_dilation(target_mask, iterations=i)
        buff_ok = np.full(MNDWI.shape, np.nan)
        buff_ok[buffered] = MNDWI[buffered]
        masked_arr = np.ma.masked_array(MNDWI, mask=(~buffered | np.isnan(MNDWI)))
        valid_1 = masked_arr.compressed()

        if valid_1.size == 0 or np.isnan(valid_1).all():
            return None

        T_masked_buffer = np.ma.masked_array(masked_arr, mask=target_mask)
        T_masked_cluster = np.ma.masked_array(masked_arr, mask=~target_mask)
        buffer_vals = T_masked_buffer.compressed()
        cluster_vals = T_masked_cluster.compressed()

        if buffer_vals.size == 0 or cluster_vals.size == 0:
            return None

        gamma_val = skew(valid_1)
        k_val = kurtosis(valid_1)
        p_val = len(valid_1)

        try:
            BC = (gamma_val ** 2 + 1) / (k_val + ((3 * (p_val - 1)**2) / ((p_val - 2)*(p_val - 3))))
        except ZeroDivisionError:
            return None

        try:
            T_thresh, T_m1, T_m2 = threshold_otsu_mod(valid_1)
        except Exception:
            return None

        class_1 = valid_1[valid_1 > T_thresh]
        class_2 = valid_1[valid_1 < T_thresh]

        if class_1.size == 0 or class_2.size == 0:
            return None

        T_std = class_2.std()
        Ad_otsu = math.sqrt(2 * abs(class_1.mean() - class_2.mean()) / math.sqrt(class_1.std()**2 + class_2.std()**2))
        pop_percent_ot = (len(class_1) * 100 / len(class_2)) if len(class_2) > 0 else 0

        mean_1 = buffer_vals.mean()
        std_1 = buffer_vals.std()
        mean_2 = cluster_vals.mean()
        std_2 = cluster_vals.std()
        Ad = math.sqrt(2 * abs(mean_1 - mean_2) / math.sqrt(std_1**2 + std_2**2))

        combined_score = BC * Ad_otsu
        if combined_score > best_score:
            best_score = combined_score
            best_result = {
                "buffered": buffered,
                "buff_ok": buff_ok,
                "buffer_size": i,
                "T_thresh": T_thresh,
                "T_m1": T_m1,
                "T_std": T_std,
                "BC": BC,
                "Ad_otsu": Ad_otsu,
                "pop_percent_ot": pop_percent_ot,
                "cluster_vals": cluster_vals,
                "buffer_vals": buffer_vals,
                "valid_1": valid_1
            }

        if (BC > 0.55) and (Ad_otsu > 2) and (10 < pop_percent_ot < max_pop_percent_ot):
            print(f"\nBuffer size {i} meets bimodality criteria.")
            if T_thresh > safe_thresh:
                print(f"[INFO] Otsu threshold {T_thresh:.3f} exceeds safe threshold {safe_thresh}. Recomputing stats.")
                class_safe = valid_1[valid_1 < safe_thresh]
                if class_safe.size > 0:
                    T_thresh = safe_thresh
                    T_m1 = class_safe.mean()
                    T_std = class_safe.std()
            return buffered, buff_ok, i, T_thresh, T_m1, T_std
        return None

    # Step 1: Try first buffer only
    result = check_buffer(buff_size[0])
    if result is None and best_result and best_result["pop_percent_ot"] > max_pop_percent_ot:
        print(f"[INFO] First buffer pop_percent_ot too high ({best_result['pop_percent_ot']:.2f}). Trying smaller buffers.")
        for i in np.arange(2, 10, 1):
            result = check_buffer(i)
            if result:
                return result

    # Step 2: Continue with rest of buff_size (skip the first one to avoid repeat)
    for i in buff_size[1:]:
        result = check_buffer(i)
        if result:
            return result

    # Final fallback
    if best_result:
        print("\nNo strict bimodality found. Returning best buffer based on highest BC * Ashman’s D.")
        if best_result["T_thresh"] > safe_thresh:
            print(f"\n[INFO] Otsu threshold {best_result['T_thresh']:.3f} exceeds safe threshold {safe_thresh}. Recomputing stats.")
            valid_safe = best_result["valid_1"][best_result["valid_1"] < safe_thresh]
            if valid_safe.size > 0:
                best_result["T_thresh"] = safe_thresh
                best_result["T_m1"] = valid_safe.mean()
                best_result["T_std"] = valid_safe.std()
        return (best_result["buffered"], best_result["buff_ok"], best_result["buffer_size"],
                best_result["T_thresh"], best_result["T_m1"], best_result["T_std"])
    else:
        print("No valid buffer configuration found.")
        return None

def BM_check_buffer_delta_2(target_mask, MNDWI, buffer_option="post", safe_thresh=-3):
    """
    Determine buffer zone with best bimodality for thresholding analysis (delta_VV).
    
    Parameters
    ----------
    target_mask : np.ndarray
        Boolean mask of the region (result of cluster).
    MNDWI : np.ndarray
        Input image data (MNDWI, VV, delta VV).
    buffer_option : str or array
        Use 'post', 'delta', or custom buffer size array.
    safe_thresh : float
        Upper cap for Otsu threshold if it's too high.
    
    Returns
    -------
    buffered, buff_ok, buffer_size, T_thresh, T_m1, T_std
    """

    if isinstance(buffer_option, str):
        if buffer_option == "post":
            buff_size = np.arange(10, 100, 4)
        elif buffer_option == "delta":
            buff_size = np.arange(2, 12, 1)
        else:
            raise ValueError("Invalid buffer_option string. Use 'post', 'delta', or pass custom range.")
    else:
        buff_size = np.array(buffer_option)

    best_score = -np.inf
    best_result = None

    for i in buff_size:
        #print(f"\n=== Testing buffer size: {i} ===")

        buffered = binary_dilation(target_mask, iterations=i)
        buff_ok = np.full(MNDWI.shape, np.nan)
        buff_ok[buffered] = MNDWI[buffered]
        masked_arr = np.ma.masked_array(MNDWI, mask=(~buffered | np.isnan(MNDWI)))
        valid_1 = masked_arr.compressed()

        if valid_1.size == 0:
            print(f"Skipping buffer size {i} due to no valid data.")
            continue

        T_masked_buffer = np.ma.masked_array(masked_arr, mask=target_mask)
        T_masked_cluster = np.ma.masked_array(masked_arr, mask=~target_mask)
        buffer_vals = T_masked_buffer.compressed()
        cluster_vals = T_masked_cluster.compressed()

        if buffer_vals.size == 0 or cluster_vals.size == 0:
            print(f"Skipping buffer size {i} due to empty buffer or cluster.")
            continue

        try:
            gamma_val = skew(valid_1)
            k_val = kurtosis(valid_1)
            p_val = len(valid_1)
            BC = (gamma_val ** 2 + 1) / (k_val + ((3 * (p_val - 1)**2) / ((p_val - 2)*(p_val - 3))))
        except ZeroDivisionError:
            print(f"Skipping buffer size {i} due to BC error.")
            continue

        try:
            T_thresh, T_m1, T_m2 = threshold_otsu_mod(valid_1)
            class_1 = valid_1[valid_1 > T_thresh]
            class_2 = valid_1[valid_1 < T_thresh]
            if class_1.size == 0 or class_2.size == 0:
                continue
            T_std = class_2.std()
            Ad_otsu = math.sqrt(2 * abs(class_1.mean() - class_2.mean()) / math.sqrt(class_1.std()**2 + class_2.std()**2))
            pop_percent_ot = (len(class_1) * 100 / len(class_2)) if len(class_2) > 0 else 0
        except Exception as e:
            print(f"Otsu failed at buffer {i}: {e}")
            continue

        #print(f"BC: {BC:.3f}, Ad: {Ad_otsu:.3f}, Thresh: {T_thresh:.3f}, µ1: {T_m1:.3f}, σ: {T_std:.3f}")


        combined_score = BC * Ad_otsu
        if combined_score > best_score:
            best_score = combined_score
            best_result = {
                "buffered": buffered,
                "buff_ok": buff_ok,
                "buffer_size": i,
                "T_thresh": T_thresh,
                "T_m1": T_m1,
                "T_std": T_std,
                "BC": BC,
                "Ad_otsu": Ad_otsu,
                "pop_percent_ot": pop_percent_ot,
                "cluster_vals": cluster_vals,
                "buffer_vals": buffer_vals,
                "valid_1": valid_1
            }

        if (BC > 0.55) and (Ad_otsu > 2) and (pop_percent_ot > 10):
            print(f"\nBuffer size {i} meets bimodality criteria. Exiting loop.")
            return buffered, buff_ok, i, T_thresh, T_m1, T_std

    if best_result:
        # print("\nNo strict bimodality found. Returning best buffer based on highest BC * Ashman’s D.")
        # print(f"\n=== SELECTED BUFFER SIZE: {best_result['buffer_size']} ===")
        # print(f"Bimodality Coefficient (BC): {best_result['BC']:.3f}")
        # print(f"Ashman's D: {best_result['Ad_otsu']:.3f}")
        # print(f"Otsu Threshold: {best_result['T_thresh']:.3f}")
        # print(f"Otsu Mean (1): {best_result['T_m1']:.3f}")
        # print(f"Std (class < threshold): {best_result['T_std']:.3f}")

        # Check if threshold needs to be capped
        if best_result["T_thresh"] > safe_thresh:
            print(f"\n[INFO] Otsu threshold {best_result['T_thresh']:.3f} exceeds safe threshold {safe_thresh}. Recomputing stats.")
            valid_safe = best_result["valid_1"][best_result["valid_1"] < safe_thresh]
            if valid_safe.size > 0:
                best_result["T_thresh"] = safe_thresh
                best_result["T_m1"] = valid_safe.mean()
                best_result["T_std"] = valid_safe.std()

        return (best_result["buffered"], best_result["buff_ok"], best_result["buffer_size"],
                best_result["T_thresh"], best_result["T_m1"], best_result["T_std"])
    else:
        print("No valid buffer configuration found.")
        return None
    

def remove_small_patches(flood_mask, min_pixel_area=50):
    """
    Removes small connected patches from a binary flood mask.

    Parameters
    ----------
    flood_mask : np.ndarray (bool or 0/1)
        Binary mask of flooded pixels.
    min_pixel_area : int
        Minimum number of pixels to retain (default: 50 pixels = 20,000 m²).

    Returns
    -------
    cleaned_mask : np.ndarray (bool)
        Cleaned flood mask with small patches removed.
    """
    # Label connected components
    labeled_array, num_features = ndi.label(flood_mask)
    
    # Count pixels in each component
    sizes = ndi.sum(flood_mask, labeled_array, range(1, num_features + 1))

    # Create mask of valid component labels
    mask_sizes = sizes >= min_pixel_area

    # Create cleaned mask
    # cleaned_mask = mask_sizes[labeled_array - 1]
    # cleaned_mask[labeled_array == 0] = 0  # Keep background as 0

    # Get labels to keep
    labels_to_keep = np.arange(1, num_features + 1)[sizes >= min_pixel_area]

    # Create cleaned mask
    cleaned_mask = np.isin(labeled_array, labels_to_keep)

    return cleaned_mask.astype(bool)

def region_growing_from_seed_to_tolerance(seed_mask, tolerance_mask):
    """
    Grows a region starting from seed pixels and only includes neighboring pixels
    that are part of the tolerance mask (==2), using 8-connectivity.

    Parameters:
    - seed_mask: 2D numpy array (binary), 1 where seeds are, 0 elsewhere.
    - tolerance_mask: 2D numpy array (values 0 or 2), 2 where growing is allowed.

    Returns:
    - grown_region: 2D boolean numpy array, True for grown region.
    """
    start_time = time.time()

    # Initialize masks
    seeds_bool = seed_mask == 1
    tolerance_bool = tolerance_mask == 2

    grown_region = np.zeros_like(seed_mask, dtype=bool)
    grown_region[seeds_bool] = True

    queue = deque(np.argwhere(seeds_bool))

    # 8-connectivity
    neighbors = [(-1, 0), (-1, -1), (1, 0), (1, 1), (-1, 1), (0, -1), (0, 1), (1, -1)]

    while queue:
        y, x = queue.popleft()

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx

            if 0 <= ny < seed_mask.shape[0] and 0 <= nx < seed_mask.shape[1]:
                if not grown_region[ny, nx] and tolerance_bool[ny, nx]:
                    grown_region[ny, nx] = True
                    queue.append((ny, nx))

    elapsed_time = time.time() - start_time
    print(f"Region growing completed in {elapsed_time:.2f} seconds.")

    return grown_region

'''************************************ IMPORT DATA ***************************************************************************************'''
#import the singleband layer

def AUTOWADE_S1(img_path, img_path_pre,LC_path, output_path, min_clust_n, filter_size):

    filter_size = 5
    print ('AutoWADE v.0.2.0')
    '''
    :param img_path: post event Sentinel-1 GRD preprocessed path
    :param img_path_pre: pre event Sentinel-1 GRD preprocessed image path
    :param LC_path: WorldCover geotiff path
    :param dem_image_path: WorldCover geotiff path
    :param output_path: path of the folder where the output are written
    :param min_clust_n: number of clusters
    :param filter_size:
    :return: flood map as geotiff with DN:2 permanent water, DN:3 flood water
    '''
    
    #----------------READING INPUTS-----------------------------------------------
    
    NoData=-9999
    
    # -- Sentinel-1 post event
    with rasterio.open(img_path) as src:
        
        VV_after = src.read(1).astype(np.float32)
        no_data_value = NoData #src.nodata  # Get the NoData value from metadata
        # Explicitly mask pixels with a value of no_data
        VV_post = np.where(VV_after == no_data_value, np.nan, VV_after)

        # Get raster bounds
        bounds = src.bounds
        src_crs = src.crs
        # Extract corners in source CRS
        left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top

        # Check if CRS is WGS84 (EPSG:4326)
        if src_crs.to_string() != 'EPSG:4326':
            # Transform bounds coordinates to WGS84
            lon, lat = transform(
                src_crs,
                'EPSG:4326',
                [left, right],
                [bottom, top]
            )
            SW_LON, NE_LON = lon[0], lon[1]
            SW_LAT, NE_LAT = lat[0], lat[1]
        else:
            # Already in WGS84
            SW_LON, SW_LAT = left, bottom
            NE_LON, NE_LAT = right, top

        # COMPUTE SLOPE FROM DEM

        # 1. get parameters to extract the copdem

        # Create the dictionary with the parameters to pass to the application
        # aoApplicationParameters = {"name": sName}
        img_file_name = os.path.basename(img_path)
        name_output_dem = "dem_"+str(img_file_name)
        aoApplicationParameters = {
            "BBOX": {
                "northEast": {
                    "lat": NE_LAT,
                    "lng": NE_LON
                },
                "southWest": {
                    "lat": SW_LAT,
                    "lng": SW_LON
                }
            },
            "OUTPUT": name_output_dem ,
            "DEM_RES": "DEM_30M",
            "DELETE": True
        }
        wasdi.wasdiLog("Starting DEM extractor")
        sProcessId = wasdi.executeProcessor("dem_extractor", aoApplicationParameters)
        # Here you are free to do what you want
        wasdi.wasdiLog("Extracting DEM over the AOI")
        # Call this if you need to wait for it to finish
        wasdi.waitProcess(sProcessId)
        wasdi.wasdiLog(f"dem written to file: {name_output_dem}")
        dem_image_path = wasdi.getPath(name_output_dem)
        wasdi.wasdiLog(f"dem written to file: {dem_image_path}")
        # Prepare masks

        # a. Urban mask from LandCover
        urban_from_LC = resample_thematic (img_path, LC_path, 'urban','mode',  1)
        # b. Permanent water from LandCover
        water_from_LC = resample_thematic (img_path, LC_path, 'water','mode', 1)
        # c. Wetlands from LandCover
        wetlands_from_LC = resample_thematic(img_path, LC_path, 'Herbaceous Wetlands', 'mode', 1)
        # b. Slope from DEM 
        #print('---Computing SLOPE')
        slope_image_path = SLOPE_COMPUTATION(dem_image_path, output_path)
        slope_mask = resample_thematic (img_path, slope_image_path, 'slope','bilinear', 1) # mask with value 1 for pixel with slope<5degrees

        # Remove temporal file
        #os.remove(dem_image_path)
        #wasdi.deleteProduct (name_output_dem)
        os.remove(slope_image_path)
        
        # Applying mask to post event image
        VV_post = np.where((VV_post == 0) | (urban_from_LC == 1) | (slope_mask == 1), np.nan, VV_post)

        urban_slope_mask = urban_from_LC | slope_mask

        transform = src.transform
        crs = src.crs

        src_crs = src.crs
        src_bounds = src.bounds
        src_resolution = src.res

    # Sentinel-1 pre event (VV_pre) as a GDAL dataset
    VV_pre_ds = gdal.Open(img_path_pre)
    
    # Resample VV_pre to match VV_post grid
    VV_pre_resampled_ds = gdal.Warp('', VV_pre_ds, format='MEM',
                                    xRes=src_resolution[0],
                                    yRes=src_resolution[1],
                                    outputBounds=src_bounds,
                                    dstSRS=src_crs.to_wkt(),  # Convert CRS to WKT
                                    resampleAlg=gdal.GRA_Bilinear)  # Bilinear resampling
    

    # Read the resampled raster as an array
    VV_pre_stack = VV_pre_resampled_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    VV_pre_stack = np.where(VV_pre_stack == NoData, np.nan, VV_pre_stack)
    
    
    # Compute Delta image (post - pre)
    VV_delta = VV_post - VV_pre_stack
    
    # '''************************************ POST EVENT IMAGE ANALYSIS *********************************************************************'''
    # '''************************************ clustering *********************************************************************'''

    wasdi.wasdiLog (" ***** Starting Clustering on POST event image  ***** ")
    
    # Clustering Derived Water AREA - CDWA
    clustering_DWM, target_mask = clustering_2(VV_post, urban_slope_mask, min_clust_n)

    if clustering_DWM is None or target_mask is None:
        wasdi.wasdiLog("Clustering failed due to insufficient valid data (area maske due to slope or urban). Exiting AUTOWADE_S1 early.")
        return
    
    # intersect water delineation (post) with permanet water --> Modified Clustering Derived Water Area MCDWA
    #MCDWMask
    intersect_post_perm = np.full(VV_post.shape, np.nan)
    intersect_post_perm = np.where((water_from_LC == 1) & (target_mask == 1), 1, 0)
    
    edge_post = RobFilter(intersect_post_perm)

    # Get valid value, unmasked pixels of MCDWA
    # Masked 2D array
    masked_VV_intersect = np.where(intersect_post_perm == 1, VV_post, np.nan)
    # Remove nan values
    masked_values = masked_VV_intersect[~np.isnan(masked_VV_intersect)]

    # ----------------------STATISTICS FOR REGION GROWING
    mean_v = masked_values.mean()
    median_v = np.median(masked_values)
    dev_v = masked_values.std()
    tol_v = median_v + 2 * dev_v

    print(f"mean value {mean_v}")
    print(f"median value {median_v}")
    print(f"dev std value {dev_v}")
    print("++++++++++++++++REGION GROWING")
    print(f"seeds VV_post < {median_v} and tolerance {median_v + 2 * dev_v}")


    #-------------------------------------------------------------------
    
    # '''************************************ THRESHOLDING *********************************************************************'''
    
    wasdi.wasdiLog ("\n ***** Checking Bimodal distribution and finding Otsu thresholds on POST event  ***** ")
    
    #*************************** test otsu
    # perform the bimodality check and buffer on the CDWM - clustering derived water mask
    result = BM_check_buffer_2(edge_post.astype(bool), VV_post, 'post')
    if result is None:
        wasdi.wasdiLog("BM_check_buffer_2 failed: No valid buffer size met bimodality criteria.")
        return  # or return False
    else:
        buffered, buff_ok, buff_size, otsu_thresh, otsu_mean, otsu_std = result

    #buffered, buff_ok, buff_size, otsu_thresh, otsu_mean, otsu_std = BM_check_buffer_2(edge_post.astype(bool), VV_post, 'post')
    
    wasdi.wasdiLog('\n    --- BM CHECK ---> done ******\n')
    # Get stats for region grwing - seeds threshold and tolerance
    #print(f"mean value from otsu {otsu_mean}")
    #print(f"std from otsu value {otsu_std}")

    # threshold for seeds = 0.5 * (otsu + mean_otsu)
    seed_otsu_old = (otsu_mean + otsu_thresh) / 2

    if mean_v < seed_otsu_old:
        seed_otsu = ((otsu_mean + otsu_thresh) / 2 + mean_v) / 2
    else:
        seed_otsu = seed_otsu_old

    # tolerance is otsu
    tolerance_old = otsu_thresh
    tol = otsu_mean + 2 * otsu_std

    if tol < tolerance_old:
        tolerance = tol
    else:
        tolerance = tolerance_old

    wasdi.wasdiLog (f"Otsu threshold (tolerance): {tolerance: .4f}")
    wasdi.wasdiLog (f"Otsu threshold (seeds): {seed_otsu: .4f}")
    
    # '''************************************ REGION GROWING *********************************************************************'''
    
    seed_post = np.full(VV_post.shape, np.nan)
    seed_post = np.where(VV_post<=seed_otsu, 1, 0)
    
    seed_post_filt = ndimage.median_filter(seed_post, size=filter_size)
        
    wasdi.wasdiLog (' +++ Starting Region growing on POST event image')
    # Thresholding Derived Water Mask 
    grown_region_post = region_growing(VV_post, seed_post_filt, tolerance)
    
    # Create open water mask
    
    # 1. create buffer of 1 km from the grown region - TDWM
    buffer_pixels = int(1000/20) #500 M per side
    #grown_mask = grown_region_post.astype(bool)
    grown_mask = ((grown_region_post==1) & (water_from_LC==1)).astype(bool)
    one_KM_mask = binary_dilation(grown_mask, iterations=buffer_pixels)

    # --create open water
    open_water = np.where((((grown_region_post==1) & (target_mask==1))|((grown_region_post==1) & (one_KM_mask==1) ) ), 1,0)

    
    # '''************************************ POST-PRE EVENT DIFFERENCE IMAGE ANALYSIS *********************************************************************'''
    
    # '''************************************ clustering *********************************************************************'''
    
    wasdi.wasdiLog ("Starting Clustering on POST-PRE difference")
    
    MNDWI_delta, target_mask_2 = clustering_2(VV_delta, urban_slope_mask, min_clust_n)

    # intersect delta cluster with open water but exclude permanent water --> Modified Clustering Derived Flood Area MCDWA
    # MCDFMask
    MCDF_Mask = np.full(VV_post.shape, np.nan)
    MCDF_Mask = np.where((water_from_LC == 0) & (target_mask_2 == 1) & (open_water ==1 ), 1, 0)

    # '''************************************ THRESHOLDING *********************************************************************'''
    
    wasdi.wasdiLog ("\n ***** Checking Bimodal distribution and finding Otsu thresholds on Delta *****\n ")
    
    #*************************** test otsu
    # perform the bimodality check and buffer on the CDWM - clustering derived water mask
    buffered_D, buff_ok_D, buff_size_D, otsu_thresh_D, otsu_mean_D, otsu_std_D = BM_check_buffer_delta_2(MCDF_Mask.astype(bool), VV_delta, 'delta')
       
    wasdi.wasdiLog('\n    --- BM CHECK ---> done ******\n')
    # Get stats for region grwing - seeds threshold and tolerance
    #print(f"mean value from otsu {otsu_mean_D}")
    #print(f"std from otsu value {otsu_std_D}")
    
    # threshold for seeds = 0.5 * (otsu + mean_otsu)
    seed_otsu_D = (otsu_mean_D + otsu_thresh_D)/2
    # tolerance is otsu
    tolerance_D = otsu_thresh_D

    wasdi.wasdiLog (f"Otsu threshold (tolerance): {tolerance_D: .4f}")
    wasdi.wasdiLog(f"Otsu threshold (seeds): {seed_otsu_D: .4f}")
    
    # '''************************************ REGION GROWING *********************************************************************'''
    
    seed_delta = np.full(VV_delta.shape, np.nan)
    seed_delta = np.where(VV_delta<=seed_otsu_D, 1, 0)
    seed_delta_filt = ndimage.median_filter(seed_delta, size=filter_size)

    wasdi.wasdiLog (' +++ Starting Region growing on Delta Post - Pre image')
    # Thresholding Derived Water Mask 
    grown_region_Delta = region_growing(VV_delta, seed_delta_filt, tolerance_D)

    # Computing Flood Mask
    one_KM_mask_D = binary_dilation(((grown_region_Delta==1) & (target_mask_2==1)), iterations=buffer_pixels)

    flood_mask = np.where(
        (
            ((grown_region_Delta==1) & (target_mask_2==1)) |((grown_region_Delta==1) & (one_KM_mask_D==1) ) ), 1,0)

    '''--------------- Final map ------------------------------'''
    # assign 3 to flood - 2 to permanent water - 1 masked pixel

    # Cleaning results and final map preparation

    #--- 1. Permanent water
    open_water_clean = ndimage.median_filter(open_water, size=filter_size)
    # compute minimun mapping unit of 10 ha --> 10000/400
    open_water_clean = remove_small_patches(open_water_clean, min_pixel_area=25)

    # ****** Final Permanent water class --> DN 2
    p_Water = np.where((open_water == 1) & (water_from_LC == 1), 1, 0)

    # --- 2. Flood water
    flood_class = np.where((open_water == 1) & (flood_mask == 1), 1, 0)

    # Final Food water --> DN 3
    # compute minimun mapping unit of 10 ha --> 10000/400
    flood_class_clean = remove_small_patches(flood_class, min_pixel_area=25)

    # --- 3. Computing and refining residual water --> flood

    # Refining classification to exclude areas too far from main water bodies
    residual_open_water = np.where((open_water_clean == 1) & (water_from_LC == 0) & (flood_class_clean == 0), 2, 0)
    residual_grown = region_growing_from_seed_to_tolerance(p_Water, residual_open_water)
    possible_Water = np.where((open_water == 1) & (wetlands_from_LC == 1), 1, 0)
    possible_grown = region_growing_from_seed_to_tolerance(possible_Water, residual_open_water)

    # ****** Final residual flood --> DN 3
    # 2.5 km buffer around permanent water
    two_KM_mask = binary_dilation(p_Water, iterations=int(2500 / 20))
    residual_flood = ((residual_open_water == 2) & (two_KM_mask == 1)) | (
                                                           (residual_open_water == 2) & (residual_grown == 1)) | (
                                                                    (residual_open_water == 2) & (possible_grown == 1))

    overall_water = np.where(urban_slope_mask, 0,
                             np.where((p_Water == 1), 2,
                                      np.where(flood_class_clean, 3,
                                               np.where((residual_flood==1),
                                                        3, 1)))).astype(np.uint8)

    wasdi.wasdiLog ("Writing output")
    write_geotiff(overall_water, transform, crs, output_path)