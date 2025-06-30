import wasdi
from AUTOWADE_S2 import *

from datetime import datetime
from datetime import timedelta

import numpy
import zipfile
import os
from osgeo import gdal

def run():
    wasdi.wasdiLog("AutoWADE S2 v.0.0.5")

    # Read the input parameters
    aoInputParameters = wasdi.getParametersDict()

    # Declare the payload
    aoPayload = {}

    # Add the inputs as a member of the payload
    aoPayload["inputs"] = aoInputParameters
    wasdi.setPayload(aoPayload)


    image_name = wasdi.getParameter("S2_ImageToProcess", "")
    lulc_image_name = wasdi.getParameter("LULC_IMAGE", "")
    dem_aux_wbm_name = wasdi.getParameter("DEM_AUX_WBM","")
    output_name = wasdi.getParameter("OUTPUT", "output.tif")
    OPTION_1 = wasdi.getParameter("OPTION_1", False)
    buffer_size = wasdi.getParameter("BUFFER_SIZE", 10)
    scale_factor = wasdi.getParameter("SCALE_FACTOR", "")
    offset = wasdi.getParameter("OFFSET", "")
    no_data_value = wasdi.getParameter("NO_DATA", "0")

    #image_full_path=wasdi.getPath(image_name)
    lulc_image_path= wasdi.getPath(lulc_image_name)
    dem_aux_image_path = wasdi.getPath(dem_aux_wbm_name)
    output_name_path=wasdi.getPath(output_name)

    wasdi.wasdiLog("Starting AUTOWADE")
    AUTOWADE_S2(image_name, lulc_image_path, dem_aux_image_path, OPTION_1, buffer_size, scale_factor, offset, no_data_value, output_name_path)
    wasdi.addFileToWASDI(output_name, sStyle="autowade_s1_flood_permanent")



if __name__ == '__main__':
    wasdi.init("./config.json")
    run()