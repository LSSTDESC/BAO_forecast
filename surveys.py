import numpy as np
from dataclasses import dataclass

@dataclass
class BAOBin:
    z_min: float
    z_max: float
    n_gal: float
    sigma_z: float
    bias: float

def GetSurvey(survey):

    # -----------------------------------------------------
    # Survey definitions
    # -----------------------------------------------------

    if survey == "DES_Y1_BAO":
        z_edges = {0: (0.60, 0.70), 1: (0.70, 0.80), 2: (0.80, 0.90), 3: (0.90, 1.00)}
        area = 1336
        nz = [386057, 353789, 330959, 229395]
        sz = [0.023, 0.028, 0.029, 0.036]

    elif survey == "DES_Y3_BAO":
        z_edges = {0: (0.60, 0.70), 1: (0.70, 0.80), 2: (0.80, 0.90), 3: (0.90, 1.00), 4: (1.00, 1.10)}
        area = 4108.47
        nz = [1478178, 1632805, 1727646, 1315604, 877760]
        sz = [0.021, 0.025, 0.029, 0.030, 0.040]

    elif survey == "DES_Y6_BAO":
        z_edges = {0: (0.60, 0.70), 1: (0.70, 0.80), 2: (0.80, 0.90), 3: (0.90, 1.00), 4: (1.00, 1.10), 5: (1.10, 1.20)}
        area = 4273.42
        nz = [2854542, 3266097, 3898672, 3404744, 1752169, 761332]
        sz = [0.0232, 0.0254, 0.0292, 0.0358, 0.0403, 0.0415]

    elif survey == "DES_Y6_maglim":
        z_edges = {0: (0.20, 0.40), 1: (0.40, 0.55), 2: (0.55, 0.70), 3: (0.70, 0.85), 4: (0.85, 0.95), 5: (0.95, 1.05)}
        area = 4031
        nz = [1852538, 1335294, 1413738, 1783834, 1391521, 1409280]
        sz = [0.0323, 0.0272, 0.0180, 0.0206, 0.0277, 0.0372]

    elif survey == "LSST_Y1_lens":
        z_edges = {0: (0.20, 0.40), 1: (0.40, 0.60), 2: (0.60, 0.80), 3: (0.80, 1.00), 4: (1.00, 1.20)}
        area = 12300
        nz = np.array([3.11, 4.29, 4.25, 3.59, 2.76]) * (area * 60**2)
        sz = [0.03] * len(nz)

    elif survey == "LSST_Y10_lens":
        z_edges = {0: (0.20, 0.30), 1: (0.30, 0.40), 2: (0.40, 0.50), 3: (0.50, 0.60), 4: (0.60, 0.70),
                   5: (0.70, 0.80), 6: (0.80, 0.90), 7: (0.90, 1.00), 8: (1.00, 1.10), 9: (1.10, 1.20)}
        area = 14300
        nz = np.array([2.89, 4.09, 4.94, 5.45, 5.65, 5.61, 5.40, 5.07, 4.67, 4.23]) * (area * 60**2)
        sz = [0.03] * len(nz)

    else:
        raise ValueError(
            f"Unknown survey '{survey}'. "
            f"Available: DES_Y1, DES_Y3, DES_Y6, LSST_Y1, LSST_Y10"
        )

    bias = [1.6] * len(nz) # same value for all redshift bins, but it could be adjusted

    # -----------------------------------------------------
    # Build bins
    # -----------------------------------------------------

    bins = [
        BAOBin(
            z_min=z_edges[i][0],
            z_max=z_edges[i][1],
            n_gal=nz[i],
            sigma_z=sz[i],
            bias=bias[i],
        )
        for i in range(len(nz))
    ]

    return bins, area
    