## AUTOWADE algorithms

This repository contains the **Python implementation** of the **AUTOWADE** (AUTOmatic WAter DEtector) algorithms for automatic flood mapping, developed by **CIMA Research Foundation**, for optical and Syntethic Aperture Radar (SAR) EO data [REF-01].  
Two tailored and independently developed modules automatically detect flooded areas from Copernicus satellite imagery:

- **`autowade_s1`** – tailored for Sentinel-1 Synthetic Aperture Radar (SAR) data, works with a pair of pre and post event image [REF-01]  
- **`autowade_s2`** – tailored for Sentinel-2 optical data, works with single image [REF-02]

The two algorithms have been developed for the **Copernicus LAC** program.
These tools are developed in Python and has been deployed on the **Copernicus Specialized Processing Environment (PE)** powered by WASDI Sarl (https://wasdi.net/).    
The fully automated SAR and optical flood mapping chain integrating `autowade_s1` and `autowade_s2` is available in the Flood Extent Mapping Service available in the Copernicus-LAC Specialized PE (https://coplac.wasdi.net/#/cl_flood_extent/appDetails).


## Repository Structure

```AUTOWADE/
├── autowade_s1/ # Code for Sentinel-1 flood mapping
├── autowade_s2/ # Code for Sentinel-2 flood mapping
└── README.md # This file
```

## References

- [REF-01] Pulvirenti, L., Squicciarino, G., Fiori, E., Ferraris, L., & Puca, S. (2021).  
  *A Tool for Pre-Operational Daily Mapping of Floods and Permanent Water Using Sentinel-1 Data.* Remote Sensing, 13(7), 1342. [https://doi.org/10.3390/rs13071342](https://doi.org/10.3390/rs13071342)

- [REF-02] Pulvirenti, L., Squicciarino, G., & Fiori, E. (2020).  
  *A method to automatically detect changes in multitemporal spectral indices: Application to natural disaster damage assessment.*  
  Remote Sensing, 12(17), 2681. [https://doi.org/10.3390/rs12172681](https://doi.org/10.3390/rs12172681)

