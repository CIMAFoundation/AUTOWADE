## Autowade S1
CIMA's Flood Detection algorithm for Sentinel-1 pair of images.

The application executes the AUTOmatic Water Areas DEtector (AUTOWADE) [<a href="#ref-1">1</a>] algorithm for flood detection using Sentinel-1 (S1) SAR data, inside the WASDI platform (https://www.wasdi.net).
The algorithm analyzes a pair of S1 images: one acquired before and one after a flood event. These images must have the same relative orbit and cover the same geographic area.

The algorithm also uses ancillary data to improve flood detection accuracy:  
-The Copernicus Digital Elevation Model (CopDEM) [<a href="#ref-2">2</a>]: used to compute slope and exclude steep areas from the analysis.     
-ESA WorldCover Map [<a href="#ref-3">3</a>]: used to exclude urban areas and integrated into the permanent water extraction process.    

The output is a water extent (flooded and permanent water) map, with a spatial resolution of 20 meters, classified as:  
`0` = Masked pixels  
`1` = No flood  
`2` = Permanent water  
`3` = Flooded areas.  
The minimum mapping unit is 10 hectares. 

---

## Input Parameters

- **PRE_IMAGE**: Preprocessed Sentinel-1 GRD VV polarization image acquired *before* the flood event, as a `.tif` file. This can be produced using the WASDI processor `coplac_autowade_preprocessing_s1`.                                           
- **POST_IMAGE**: Preprocessed Sentinel-1 GRD VV polarization image acquired *after* the flood event, as a `.tif` file. Use the same preprocessing workflow as above.                                                                                 
- **LULC_IMAGE**: ESA WorldCover Land Use / Land Cover map over the Area of Interest (AOI), provided as a `.tif` file. It can be generated using the `world_cover_extractor` WASDI app.  
- **OUTPUT**: Name of the output flood extent map (GeoTIFF format).                                                                                                                                            

### Advanced Parameters (keep default values)
These parameters control internal behavior and should only be modified by experienced users only:
- **MIN_CLUST_N**: Minimum number of pixels to form a valid flood cluster. *(Default: 6)*
- **FILTER_SIZE**: Size of the spatial filter kernel to smooth classification output. *(Default: 3)*

## Output
- A flood extent map as an unsigned 8-bit GeoTIFF (20m spatial resolution).
- The output is uploaded to WASDI and styled using `DDS_FLOODED_AREAS`.

## Notes
- The **Copernicus DEM** is automatically extracted using WASDI’s `dem_extractor` based on the image footprint. **No DEM input is required**.
- Both Sentinel-1 images must be acquired over the same relative orbit.

## Example  
Example of the input parameter file:  

```json
{
  "PRE_IMAGE": "subset_of_S1_20240507.tif",
  "POST_IMAGE": "subset_of_S1_20240519.tif",
  "LULC_IMAGE": "ESA_WorldCover_10m_2021_v200_aoi_res.tif",
  "OUTPUT": "S1_20240519_flood_map.tif",
  "MIN_CLUST_N": 6,
  "FILTER_SIZE": 3
}
```

---

## Code Structure

- **MyProcessor.py**  
  Contains the main function which calls the core functions from the `autowade_s2` module.

- **Autowade_s2.py**  
  Contains the functions and implementation of the AUTOWADE_S2 algorithm.

- **Resample_themathic.py**  
  Provides utilities for resampling and aligning thematic layers (e.g., land cover, DEM) to the Sentinel-2 image.

---

## References
[1] <a id="ref-1"></a> Pulvirenti, L., Squicciarino, G., Fiori, E., Ferraris, L., & Puca, S. (2021). *A Tool for Pre-Operational Daily Mapping of Floods and Permanent Water Using Sentinel-1 Data*. Remote Sensing, 13(7), 1342. 
https://doi.org/10.3390/rs13071342  

[2] <a id="ref-2"></a> Copernicus DEM – Global Digital Elevation Model - COP-DEM_GLO-30. https://doi.org/10.5270/ESA-c5d3d65  

[3] <a id="ref-3"></a> ESA WorldCover 10 m 2021 v200. Zanaga, D. et al., 2022. https://doi.org/10.5281/zenodo.7254221
