import numpy as np
from dataclasses import dataclass

@dataclass
class BAOBin:
    z_min: float
    z_max: float
    n_gal: float
    sigma_z: float
    bias: float = 1.6

def GetSurvey(survey):

    # -----------------------------------------------------
    # Survey definitions
    # -----------------------------------------------------
    
    if survey == "DES_Y1":
        z_min, z_max, dz = 0.6, 1.0, 0.1
        area = 1336
        nz = [386057, 353789, 330959, 229395]
        sz = [0.023, 0.028, 0.029, 0.036]

    elif survey == "DES_Y3":
        z_min, z_max, dz = 0.6, 1.1, 0.1
        area = 4108.47
        nz = [1478178, 1632805, 1727646, 1315604, 877760]
        sz = [0.021, 0.025, 0.029, 0.030, 0.040]

    elif survey == "DES_Y6":
        z_min, z_max, dz = 0.6, 1.2, 0.1
        area = 4273.42
        nz = [2854542, 3266097, 3898672, 3404744, 1752169, 761332]
        sz = [0.0232, 0.0254, 0.0292, 0.0358, 0.0403, 0.0415]

    elif survey == "LSST_Y1":
        z_min, z_max, dz = 0.2, 1.2, 0.2
        area = 12300
        nz = np.array([3.11, 4.29, 4.25, 3.59, 2.76]) * (area * 60**2)
        sz = [0.03] * 5

    elif survey == "LSST_Y10":
        z_min, z_max, dz = 0.2, 1.2, 0.1
        area = 14300
        nz = np.array([2.89, 4.09, 4.94, 5.45, 5.65, 5.61, 5.40, 5.07, 4.67, 4.23]) * (area * 60**2)
        sz = [0.03] * 10

    else:
        raise ValueError(
            f"Unknown survey '{survey}'. "
            f"Available: DES_Y1, DES_Y3, DES_Y6, LSST_Y1, LSST_Y10"
        )

    # -----------------------------------------------------
    # Build bins
    # -----------------------------------------------------
    
    z_edges = np.arange(z_min, z_max + 1e-12, dz)

    bins = [
        BAOBin(
            z_min=z_edges[i],
            z_max=z_edges[i + 1],
            n_gal=nz[i],
            sigma_z=sz[i],
        )
        for i in range(len(nz))
    ]

    return bins, area
    