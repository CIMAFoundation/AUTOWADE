"""
AutoWADE S1 Processor with Slope Computation

This processor launches, within the WASDI environment, the AUTOmatic Water Areas DEtector (AUTOWADE) [1]
algorithm for Sentinel-1 data (`AUTOWADE_S1`).

The algorithm uses Synthetic Aperture Radar (SAR) data by analyzing a pair of Sentinel-1 images: one
acquired before and one after a flood event. These images must have the same relative orbit and cover
the same geographic area.

The algorithm also uses ancillary data to improve flood detection accuracy:
- The Copernicus Digital Elevation Model (DEM) [2]
- ESA WorldCover Land Cover Map [3]

Note:
- The DEM is **not required as an input parameter**, as it is automatically extracted over the
  Sentinel-1 image footprint using WASDI’s built-in `dem_extractor` function.

Inputs (from WASDI parameters):
- PRE_IMAGE:    Name of the pre-event (pre-processed) Sentinel-1 image.
- POST_IMAGE:   Name of the post-event (pre-processed) Sentinel-1 image.
- LULC_IMAGE:   Name of the land use/land cover image (ESA WorldCover).
- OUTPUT:       Name of the output flooded area map (default: "output.tif").
- MIN_CLUST_N:  Minimum number of pixels to form a cluster (default: 6).
- FILTER_SIZE:  Size of the filtering kernel for smoothing (default: 3).

Outputs:
- A flood extent map (UInt8 GeoTIFF file) with the following classes:
    0 - Masked pixels
    1 - No flood
    2 - Permanent water
    3 - Flooded areas

References:
[1] Pulvirenti, L., Squicciarino, G., Fiori, E., Ferraris, L., & Puca, S. (2021).
    “A Tool for Pre-Operational Daily Mapping of Floods and Permanent Water Using Sentinel-1 Data”.
    Remote Sensing, 13(7), 1342. https://doi.org/10.3390/rs13071342

[2] Copernicus DEM – Global Digital Elevation Model - COP-DEM_GLO-30.
    https://doi.org/10.5270/ESA-c5d3d65

[3] ESA WorldCover 10 m 2021 v200.
    Zanaga, D. et al., 2022. https://doi.org/10.5281/zenodo.7254221

Author: Francesca Trevisiol
"""
import wasdi
from AUTOWADE_S1_v0_2_0 import *

def run():
    wasdi.wasdiLog("AutoWADE S1 - with slope computation")

    # Read the input parameters
    aoInputParameters = wasdi.getParametersDict()

    # Declare the payload
    aoPayload = {}

    # Add the inputs as a member of the payload
    aoPayload["inputs"] = aoInputParameters
    wasdi.setPayload(aoPayload)

    pre_image_name = wasdi.getParameter("PRE_IMAGE", "")
    post_image_name = wasdi.getParameter("POST_IMAGE", "")
    lulc_image_name = wasdi.getParameter("LULC_IMAGE", "")
    output_name = wasdi.getParameter("OUTPUT", "output.tif")
    min_clust_n = wasdi.getParameter("MIN_CLUST_N", 6)
    filter_size = wasdi.getParameter("FILTER_SIZE", 3)

    pre_image_full_path = wasdi.getPath(pre_image_name)
    post_image_full_path = wasdi.getPath(post_image_name)
    lulc_image_path = wasdi.getPath(lulc_image_name)
    output_name_path = wasdi.getPath(output_name)

    wasdi.wasdiLog("Starting AUTOWADE")
    AUTOWADE_S1(post_image_full_path, pre_image_full_path, lulc_image_path, output_name_path,
                      min_clust_n, filter_size)
    wasdi.addFileToWASDI(output_name, sStyle="DDS_FLOODED_AREAS")


if __name__ == '__main__':
    wasdi.init("./config.json")
    run()