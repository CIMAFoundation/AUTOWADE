## AUTOWADE algorithms

This repository contains the **Python implementation** of the **AUTOWADE** (AUTOmatic WAter DEtector) algorithms for automatic flood mapping, developed by **CIMA Research Foundation**, for optical and Syntethic Aperture Radar (SAR) EO data [REF-01].  
Two tailored and independently developed modules automatically detect flooded areas from Copernicus satellite imagery:

- **`autowade_S1`** – tailored for Sentinel-1 Synthetic Aperture Radar (SAR) data, works with a pair of pre and post event image [REF-01]  
- **`autowade_S2`** – tailored for Sentinel-2 optical data, works with single image [REF-02]

The two algorithms have been developed for the **CopernicusLAC** Service Development project and have been deployed on the **CopernicusLAC Specialized Processing Environment (PE)** powered by WASDI Sarl (https://wasdi.net/).    
The fully automated SAR and optical flood mapping chain integrating `autowade_s1` and `autowade_s2` is available in the Flood Extent Mapping Service on the CopernicusLAC Specialized PE (https://coplac.wasdi.net/#/cl_flood_extent/appDetails).
Detailed description of the service can be found at: [Flood Extent Mapping Details - Hydromet Hazards documentation](https://coplac-hydromet-hazards.readthedocs.io/en/latest/detailed_description/flood_event.html).

## Repository Structure

```AUTOWADE/
├── autowade_S1/ # Code for Sentinel-1 flood mapping
├── autowade_S2/ # Code for Sentinel-2 flood mapping
└── README.md # This file
```

## References

- [REF-01] Pulvirenti, L., Squicciarino, G., Fiori, E., Ferraris, L., & Puca, S. (2021).  
  *A Tool for Pre-Operational Daily Mapping of Floods and Permanent Water Using Sentinel-1 Data.* Remote Sensing, 13(7), 1342. [https://doi.org/10.3390/rs13071342](https://doi.org/10.3390/rs13071342)

- [REF-02] Pulvirenti, L., Squicciarino, G., & Fiori, E. (2020).  
  *A method to automatically detect changes in multitemporal spectral indices: Application to natural disaster damage assessment.* Remote Sensing, 12(17), 2681. [https://doi.org/10.3390/rs12172681](https://doi.org/10.3390/rs12172681)

