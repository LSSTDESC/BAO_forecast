# BAO_forecast

Forecasting Baryon Acoustic Oscillation (BAO) distance measurement precision for photometric galaxy surveys using Fisher matrix methods based on Seo & Eisenstein (2007).
> Seo & Eisenstein (2007)
> *Improved Forecasts for the Baryon Acoustic Oscillations and Cosmological Distance Scale*
> https://arxiv.org/abs/astro-ph/0701079

---

# Features

- BAO Fisher forecasting in individual redshift bins and combined precision.
- Automatic cosmology handling via `cosmoprimo`.
- Includes:
  - redshift-space distortions (RSD),
  - Silk damping,
  - non-linear BAO damping,
  - photo-z smearing.
- Returns:
  - transverse distance precision: $$\sigma_D\equiv\sigma(\ln D_M).$$

---

# Inputs

The main input to the forecast consists of:

- redshift-bin edges $(z_{\min}, z_{\max})$.
- galaxy number in each bin $N_{\rm gal}$.
- photometric redshift uncertainty $\sigma_z$.
- survey area $A$ (in square degrees).

---

# Forecast Formalism

The BAO Fisher matrix is computed as a double integral over wavenumber $k$ and angle cosine $\mu$:

$$F_{ij} = \frac{A_{\rm BAO}^2 V}{2} \int d\mu \int dk  \frac{k^2\ f_i(\mu)\ f_j(\mu)\ \exp[-2(k\Sigma_{\rm Silk})^{1.4}]\ \exp[-k^2 \Sigma_{\rm nl}^2(\mu)]}{\left[P_{\rm BAO}(k) + \frac{1}{nP\ R(k,\mu)}\right]^2},$$

where 

- the angular response functions are

$$f_D(\mu)=1-\mu^2,\quad f_H(\mu)=\mu^2.$$

- the volume is given by

$$V = \frac{4\pi}{3} f_{\rm sky} \left(\chi_{\rm max}^3 - \chi_{\rm min}^3\right),\quad\text{where}\quad f_{\rm sky} = \frac{A(\text{deg}^2)}{4\pi (180/\pi)^2}.$$

- the shotnoise parameter $nP$ is given by

$$nP = \bar{n} P_{\rm eff},\quad\text{where}\quad\bar{n} = \frac{N_{\rm gal}}{V}\quad\text{and}\quad P_{\rm eff} = (bD)^2 P(k=0.2\ \text{Mpc}/h).$$

- the redshift-space distortion factor is

$$R(k,\mu)=\left(1+\beta\mu^2\right)^2 \exp[-k^2\mu^2\Sigma_z^2],\quad\text{where}\quad\Sigma_z = \frac{c}{H(z)}\sigma_z.$$

- the anisotropic nonlinear damping is

$$\Sigma_{\rm nl}^2(\mu)=\Sigma_\parallel^2\mu^2+\Sigma_\perp^2(1-\mu^2),\quad \text{where}\quad \Sigma_\parallel=\Sigma_0 D(1+f)\quad\text{and}\quad\Sigma_\perp=\Sigma_0 D$$

The correlation coefficient between radial and transverse modes is, then, computed as

$$r=\frac{F_{DH}}{\sqrt{F_{DD}F_{HH}}},$$

and the angular BAO distance uncertainty is

$$\sigma_D=\frac{1}{\sqrt{F_{DD}(1-r^2)}}.$$

---

# Requirements

- numpy
- cosmoprimo
