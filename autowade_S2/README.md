# Autowade S2  
CIMA's Flood Detection algorithm for Sentinel-2 optical images.

The application executes the AUTOmatic WAter DEtector (AUTOWADE) algorithm for flood detection using Sentinel-2 (S2) optical data [1], inside the WASDI platform (https://www.wasdi.net).  
The algorithm analyzes a single post-event Sentinel-2 Level 2A image and uses MNDWI index to detect flooded areas.

Ancillary data sources are used to improve the detection quality:  
- **ESA WorldCover Land Cover map** [2]: used to mask out urban areas and identify permanent water.  
- **Water Body Mask from Copernicus Digital Elevation Model (CopDEM)** [3]: used to generate a sea/ocean mask, excluding marine pixels from analysis.  
- **Scene Classification Layer (SCL)** from Sentinel-2: used to mask clouds, shadows, and other invalid pixels.

The output is a water extent map with a spatial resolution of 10 meters, classified as:  
- `0` = Non-water (background/land)  
- `1` = Flooded areas  
- `2` = Permanent water

---

## Input Parameters

- **S2_ImageToProcess**: Sentinel-2 L2A image (post-event), as a `.zip` SAFE product.   
- **LULC_IMAGE**: ESA WorldCover map over the Area of Interest (AOI), in `.tif` format. Can be extracted using the WASDI `world_cover_extractor` application.  
- **DEM_AUX_WBM**: Water Body Mask (WBM) - auxiliary tif file delivered with the CopDEM-30, available in the WASDI catalog THEMATIC>CopDEM30m_wbm_global_mosaic    
- **OUTPUT**: Name of the output flood extent map (GeoTIFF format).

### Advanced Parameters (default values are typically recommended)
These parameters fine-tune the detection algorithm and should only be changed by advanced users:
- **BUFFER_SIZE**: Integer buffer (in pixels) used during contextual post-processing. *(Default: 10)*  
- **SCALE_FACTOR**: Scaling applied to reflectance bands. *(Default: 0.0001)*  
- **OFFSET**: Offset applied to band values. *(Default: -1000)*  
- **NO_DATA**: No-data value to use during processing. *(Default: 0)*

---

## Output
- A classified flood map in unsigned 8-bit GeoTIFF format (20 m resolution).
- The output is uploaded to WASDI and styled using `DDS_FLOODED_AREAS_OPTICAL`.

---

## Example  
Example input configuration in JSON format:

```json
{
  "S2_ImageToProcess": "S2A_MSIL2A_20231212T153621_N0509_R068_T18PVR_20231212T204250.zip",
  "LULC_IMAGE": "ESA_WorldCover_10m_2021_v200_aoi_32618.tif",
  "DEM_AUX_WBM": "Copernicus_DSM_10_N09_00_W076_00_WBM.tif",
  "OUTPUT": "S2_20231212T153621_FLOOD_MAP.tif",
  "BUFFER_SIZE": 10,
  "SCALE_FACTOR": 0.0001,
  "OFFSET": -1000,
  "NO_DATA": 0
}
```

---

## Code Structure

- **myProcessor.py**  
  Contains the main function which calls the core functions from the `autowade_s2` module.

- **AUTOWADE_S2.py**  
  Contains the functions and implementation of the AUTOWADE_S2 algorithm.

- **resample_themathic.py**  
  Provides utilities for resampling and aligning thematic layers (e.g., land cover, DEM) to the Sentinel-2 image.

---

## References
[1] Pulvirenti, L., Squicciarino, G., Fiori, E., Ferraris, L., & Puca, S. (2021). *A Tool for Pre-Operational Daily Mapping of Floods and Permanent Water Using Sentinel-1 Data*. Remote Sensing, 13(7), 1342. 
https://doi.org/10.3390/rs13071342  

[2] ESA WorldCover 10 m 2021 v200. Zanaga, D. et al., 2022. 
https://doi.org/10.5281/zenodo.7254221

[3] Copernicus DEM – Global Digital Elevation Model - COP-DEM_GLO-30  
https://doi.org/10.5270/ESA-c5d3d65  


