from dataclasses import dataclass
import numpy as np
import pyccl as ccl

# ---------------------------------------------------------
# BAO template constants
# ---------------------------------------------------------

pk_BAO = np.array([
    9.034, 14.52, 12.63, 9.481, 7.409, 6.397, 5.688, 4.804,
    3.841, 3.108, 2.707, 2.503, 2.300, 2.014, 1.707, 1.473,
    1.338, 1.259, 1.174, 1.061, 0.9409, 0.8435, 0.7792,
    0.7351, 0.6915, 0.6398, 0.5851, 0.5376, 0.5018, 0.4741,
    0.4484, 0.4210, 0.3929, 0.3671, 0.3456, 0.3276, 0.3112,
    0.2950, 0.2788, 0.2635, 0.2499, 0.2379, 0.2270, 0.2165,
    0.2062, 0.1965, 0.1876, 0.1794, 0.1718, 0.1646
])

k_step = 0.01
k_vector = k_step * (np.arange(len(pk_BAO)) + 0.5)

power_BAO = 1896.1 # The power spectrum at k=0.2h Mpc^-1 for sigma8=0.8 and Planck cosmo
silk_BAO = 7.76
amp_BAO = 0.5 # Approximate, see Seo & Eisenstein for details

mu_step = 0.0001
mu_vector = np.arange(0.0, 1.0, mu_step)

# ---------------------------------------------------------
# Survey bin definition
# ---------------------------------------------------------

@dataclass
class BAOBin:
    z_min: float
    z_max: float
    n_gal: float
    sigma_z: float
    bias: float = 1.6

# ---------------------------------------------------------
# Individual-bin BAO error calculator
# ---------------------------------------------------------

def IndividualBAOError(z_min, z_max, n_gal, sigma_z, bias, area_deg2, cosmo, recon_factor=1.0, sigma8_fid=0.8):

    z = 0.5 * (z_min + z_max)

    # -----------------------------------------------------
    # Survey volume
    # -----------------------------------------------------

    fsky = area_deg2 / (4 * np.pi * (180 / np.pi)**2)
    chi_min = cosmo.comoving_radial_distance(1.0 / (1.0 + z_min)) * cosmo["h"]
    chi_max = cosmo.comoving_radial_distance(1.0 / (1.0 + z_max)) * cosmo["h"]
    volume = (4.0 * np.pi / 3.0 * fsky * (chi_max**3 - chi_min**3)) / 1e9

    # -----------------------------------------------------
    # Growth + RSD
    # -----------------------------------------------------

    D = cosmo.growth_factor(1.0 / (1.0 + z))
    f = cosmo.growth_rate(1.0 / (1.0 + z))
    beta = f / bias

    power = (bias * D)**2 * power_BAO

    nbar = n_gal / volume / 1e9
    nP = nbar * power

    # -----------------------------------------------------
    # BAO damping from non-linear evolution
    # -----------------------------------------------------

    sigma0 = (12.4 * sigma8_fid / 0.9 * D * 0.758 * recon_factor)
    sigma_par = sigma0# * (1 + f)
    sigma_per = sigma0
    sigma_z_dist = 2997.92458 / cosmo.h_over_h0(1.0 / (1.0 + z)) * sigma_z
    sigma_z_bao = sigma_z_dist / (cosmo.comoving_radial_distance(1.0 / (1.0 + z)) * cosmo["h"]) * 105.0
    sigma2_tot = (sigma_par**2 * mu_vector**2 + sigma_per**2 * (1.0 - mu_vector**2) + 0.5 * sigma_z_bao**2)

    # -----------------------------------------------------
    # BAO damping from photo-z smearing
    # -----------------------------------------------------

    sigzdampl = BAOdampsigz(z, sigma_z, cosmo)
    # sigzdampl = np.ones(len(k_vector))

    # Ensure same shape as k_vector/pk_BAO
    if len(sigzdampl) != len(k_vector):
        raise ValueError(
            f"sigzdampl has length {len(sigzdampl)} "
            f"but k_vector has length {len(k_vector)}"
        )

    # -----------------------------------------------------
    # Silk damping
    # -----------------------------------------------------

    silk = np.exp(-2.0 * (k_vector * silk_BAO)**1.4)
    
    # -----------------------------------------------------
    # Double integral
    # -----------------------------------------------------

    rsd = (1.0 + beta * mu_vector**2)**2
    tmp = (pk_BAO[:, None] + np.exp(k_vector[:, None]**2 * sigma_z_dist**2 * mu_vector**2) / (nP * rsd))
    
    fisher_integrand = (k_vector[:, None]**2 * sigzdampl[:, None]**2 * silk[:, None] * np.exp(-k_vector[:, None]**2 * sigma2_tot) / tmp**2)
    fisher = np.trapz(np.trapz(fisher_integrand, mu_vector), k_vector)
    fisher *= (amp_BAO**2 * 1e9 * volume / 2.0)

    return 1.0 / np.sqrt(fisher)

# ---------------------------------------------------------
# BAO error combiner
# ---------------------------------------------------------

def CombinedBAOError(bins, area_deg2, cosmo, recon_factor=1.0, sigma8_fid=0.8):

    fisher_sum = 0.0

    for b in bins:

        err = IndividualBAOError(
            z_min=b.z_min,
            z_max=b.z_max,
            n_gal=b.n_gal,
            sigma_z=b.sigma_z,
            bias=b.bias,
            area_deg2=area_deg2,
            cosmo=cosmo,
            recon_factor=recon_factor,
            sigma8_fid=sigma8_fid,
        )

        fisher_sum += 1.0 / err**2

    return np.sqrt(1.0 / fisher_sum)

# ---------------------------------------------------------
# Photo-z smearing damping
# ---------------------------------------------------------

def BAOdampsigz(z, sigma_z, cosmo, nthbin=50, dz=0.01, z_max=3.0):

    chi_z = cosmo.comoving_radial_distance(1.0 / (1.0 + z)) * cosmo["h"]

    # angular grid
    theta = (np.arange(nthbin) + 0.5) * (16.0 * np.pi / nthbin)

    # redshift grid
    zj = (np.arange(int(z_max / dz)) + 0.5) * dz

    # Gaussian redshift kernel
    gz = (
        dz / np.sqrt(2.0 * np.pi * sigma_z**2)
        * np.exp(-(z - zj) ** 2 / (2.0 * sigma_z**2))
    )

    chi_j = cosmo.comoving_radial_distance(1.0 / (1.0 + zj)) * cosmo["h"]

    # projection
    cl = np.sum(
        gz[None, :] * np.cos(theta[:, None] * (chi_j / chi_z)),
        axis=1
    )

    cl0 = np.cos(theta)
    norm = np.sum(cl**2) * (16.0 * np.pi / nthbin) / np.pi / 3.0

    out = cl / cl0

    if sigma_z < 0.01 and norm < 0.1:
        return np.ones(nthbin)

    return out
    