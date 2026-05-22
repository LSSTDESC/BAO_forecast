import numpy as np
from cosmoprimo import Cosmology

# ---------------------------------------------------------
# Cosmological parameters
# ---------------------------------------------------------

cosmo = Cosmology(
    h=0.7,
    Omega_cdm=0.25,
    Omega_b=0.05,
    sigma8=0.8,
    n_s=0.96,
    engine="class"
)

# ---------------------------------------------------------
# BAO template
# ---------------------------------------------------------

pk_implementation = "new"

if pk_implementation == "old":

    power_BAO = 1896.1 # power spectrum at 0.2 h/Mpc for sigma8 = 0.8 and Planck cosmology. In (Mpc/h)^3
    
    pk_BAO = np.array([
        9.034, 14.52, 12.63, 9.481, 7.409, 6.397, 5.688, 4.804,
        3.841, 3.108, 2.707, 2.503, 2.300, 2.014, 1.707, 1.473,
        1.338, 1.259, 1.174, 1.061, 0.9409, 0.8435, 0.7792,
        0.7351, 0.6915, 0.6398, 0.5851, 0.5376, 0.5018, 0.4741,
        0.4484, 0.4210, 0.3929, 0.3671, 0.3456, 0.3276, 0.3112,
        0.2950, 0.2788, 0.2635, 0.2499, 0.2379, 0.2270, 0.2165,
        0.2062, 0.1965, 0.1876, 0.1794, 0.1718, 0.1646
    ]) # dimensionless

    kh_step = 0.01
    kh_vector = kh_step * (np.arange(len(pk_BAO)) + 0.5) # in h/Mpc

    silk_BAO = 7.76 # in Mpc/h

elif pk_implementation == "new":

    pkz = cosmo.get_fourier(engine="class").pk_interpolator()
    pk = pkz.to_1d(z=0)
    power_BAO = pk(0.2) # at 0.2 h/Mpc. In (Mpc/h)^3

    kh_step = 0.01
    kh_vector = kh_step * (np.arange(50 * 0.01 / kh_step) + 0.5) # in h/Mpc. k binning used in https://arxiv.org/pdf/astro-ph/0701079 (see page 9)
    # kh_vector = 10**np.linspace(np.log10(10**-4 / cosmo.h), np.log10(10**2), 5*10**2) # in h/Mpc. Will give the same result as the previous kh_vector
    
    pk_BAO = pk(kh_vector) / power_BAO # dimensionless

    silk_BAO = 1.0 / (1.6 * (cosmo.Omega0_b * cosmo.h**2)**0.52 * (cosmo.Omega0_m * cosmo.h**2)**0.73 * 
                      (1.0 + (10.4 * cosmo.Omega0_m * cosmo.h**2)**-0.95) / cosmo.h) # page 9 of https://arxiv.org/pdf/astro-ph/0701079. In Mpc/h
    
amp_BAO = 0.5 # approximate, see page 10 of https://arxiv.org/pdf/astro-ph/0701079 for details

mu_vector = np.linspace(0.0, 1.0, 10**4)

# ---------------------------------------------------------
# Individual-bin BAO error calculator
# ---------------------------------------------------------

def IndividualBAOError(z_min, z_max, n_gal, sigma_z, bias, area, recon_factor=1.0):

    z = 0.5 * (z_min + z_max)

    # -----------------------------------------------------
    # Survey volume
    # -----------------------------------------------------

    fsky = area / (4 * np.pi * (180 / np.pi)**2)
    chi_min = cosmo.comoving_radial_distance(z_min) # in Mpc/h
    chi_max = cosmo.comoving_radial_distance(z_max) # in Mpc/h
    volume = (4.0 * np.pi / 3.0 * fsky * (chi_max**3 - chi_min**3)) # in (Mpc/h)^3

    # -----------------------------------------------------
    # Growth + RSD
    # -----------------------------------------------------

    D = cosmo.growth_factor(z)
    f = cosmo.growth_rate(z)
    beta = f / bias

    power = (bias * D)**2 * power_BAO # in (Mpc/h)^3

    nbar = n_gal / volume # in (h/Mpc)^3
    nP = nbar * power # dimensionless

    # -----------------------------------------------------
    # BAO damping from non-linear evolution
    # -----------------------------------------------------

    sigma0 = (12.4 * cosmo.sigma8_m / 0.9 * D * 0.758 * recon_factor) # the 12.4 is in Mpc/h. The 0.9 is sigma_8. See page 4 of https://arxiv.org/pdf/astro-ph/0701079
    sigma_par = sigma0 * (1 + f)
    sigma_per = sigma0
    sigma_z_dist = 2997.92458 / cosmo.efunc(z) * sigma_z # in Mpc/h
    sigma_z_bao = sigma_z_dist / cosmo.comoving_radial_distance(z) * cosmo.rs_drag # in Mpc/h
    sigma2_tot = (sigma_par**2 * mu_vector**2 + sigma_per**2 * (1.0 - mu_vector**2) + 0.5 * sigma_z_bao**2) # in (Mpc/h)^2

    # -----------------------------------------------------
    # BAO damping from photo-z smearing
    # -----------------------------------------------------

    # sigzdampl = BAOdampsigz(z, sigma_z, nthbin=len(kh_vector)) # This was in Ashley's original code, but it was never used

    # -----------------------------------------------------
    # Silk damping
    # -----------------------------------------------------

    silk = np.exp(-2.0 * (kh_vector * silk_BAO)**1.4)
    
    # -----------------------------------------------------
    # Double integral (with BAO component split)
    # -----------------------------------------------------
    
    R = (1.0 + beta * mu_vector**2)**2 * np.exp(-kh_vector[:, None]**2 * mu_vector**2 * sigma_z_dist**2) # eq. 27 from https://arxiv.org/pdf/astro-ph/0701079. Dimensionless
    denom = pk_BAO[:, None] + 1.0 / (nP * R) # dimensionless
    
    # base = (kh_vector[:, None]**2 * silk[:, None] * np.exp(-kh_vector[:, None]**2 * sigma2_tot * sigzdampl[:, None]**2) / denom**2)
    base = (kh_vector[:, None]**2 * silk[:, None] * np.exp(-kh_vector[:, None]**2 * sigma2_tot) / denom**2) # in (h/Mpc)^2
    
    # -----------------------------------------------------
    # Angular Fisher decompositions
    # -----------------------------------------------------
    
    Fdd_integrand = base * (1.0 - mu_vector**2)**2 # in (h/Mpc)^2
    Fhh_integrand = base * mu_vector**4 # in (h/Mpc)^2
    Fdh_integrand = base * (1.0 - mu_vector**2) * mu_vector**2 # in (h/Mpc)^2
    
    Fdd = np.trapz(np.trapz(Fdd_integrand, mu_vector), kh_vector) # in (h/Mpc)^3
    Fhh = np.trapz(np.trapz(Fhh_integrand, mu_vector), kh_vector) # in (h/Mpc)^3
    Fdh = np.trapz(np.trapz(Fdh_integrand, mu_vector), kh_vector) # in (h/Mpc)^3
    
    prefactor = (amp_BAO**2 * volume / 2.0) # in (Mpc/h)^3. I'm not sure about the 2.0 factor... I don't think it should be there
    
    Fdd *= prefactor # dimensionless
    Fhh *= prefactor # dimensionless
    Fdh *= prefactor # dimensionless
    
    r = Fdh / np.sqrt(Fhh * Fdd)
    
    Drms = 1.0 / np.sqrt(Fdd * (1.0 - r**2)) # dimensionless
    Hrms = 1.0 / np.sqrt(Fhh * (1.0 - r**2)) # dimensionless
    Rrms = Drms * np.sqrt((1.0 - r**2) / (1.0 + (Drms / Hrms) * (2.0 * r + (Drms / Hrms)))) # dimensionless

    # -----------------------------------------------------
    # Full Fisher
    # -----------------------------------------------------
    
    fisher = np.trapz(np.trapz(base, mu_vector), kh_vector) # in (h/Mpc)^3. This should be equivalent to Rrms
    fisher *= prefactor # dimensionless

    # -----------------------------------------------------
    # Print
    # -----------------------------------------------------

    print(f"z={z:.3f} | Drms={Drms:.4e}, Hrms={Hrms:.4e}, Rrms={Rrms:.4e}, r={r:.4f} | Fisher={1.0 / np.sqrt(fisher):.4e}")

    # return 1.0 / np.sqrt(fisher)
    return Drms # we want to forecast D_M. This is returninig sigma(log(D_M))

# ---------------------------------------------------------
# BAO error combiner
# ---------------------------------------------------------

def CombinedBAOError(bins, area, recon_factor=1.0):

    fisher_sum = 0.0

    for b in bins:

        err = IndividualBAOError(
            z_min=b.z_min,
            z_max=b.z_max,
            n_gal=b.n_gal,
            sigma_z=b.sigma_z,
            bias=b.bias,
            area=area,
            recon_factor=recon_factor,
        )

        fisher_sum += 1.0 / err**2

    return np.sqrt(1.0 / fisher_sum)

# # ---------------------------------------------------------
# # Photo-z smearing damping
# # ---------------------------------------------------------

# def BAOdampsigz(z, sigma_z, nthbin=50, dz=0.01, z_max=3.0):

#     chi_z = cosmo.comoving_radial_distance(z)

#     # angular grid
#     theta = (np.arange(nthbin) + 0.5) * (16.0 * np.pi / nthbin)

#     # redshift grid
#     zj = (np.arange(int(z_max / dz)) + 0.5) * dz

#     # Gaussian redshift kernel
#     gz = (
#         dz / np.sqrt(2.0 * np.pi * sigma_z**2)
#         * np.exp(-(z - zj) ** 2 / (2.0 * sigma_z**2))
#     )

#     chi_j = cosmo.comoving_radial_distance(zj)

#     # projection
#     cl = np.sum(
#         gz[None, :] * np.cos(theta[:, None] * (chi_j / chi_z)),
#         axis=1
#     )

#     cl0 = np.cos(theta)
#     norm = np.sum(cl**2) * (16.0 * np.pi / nthbin) / np.pi / 3.0

#     out = cl / cl0

#     if sigma_z < 0.01 and norm < 0.1:
#         return np.ones(nthbin)

#     return out
    