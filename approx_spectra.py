"""Approximate a stellar spectrum from GALEX (NUV) and Gaia (G, BP, RP) photometry

Model: blackbody scaled by MW extinction curve with R_V = 3.1 (Gordon et al. 2023).
Search a grid in (T_eff, E(B-V)). The flux scale (R/d)**2 is solved at each grid point.
"""

import warnings
import numpy as np
import astropy.units as u
from astroquery.svo_fps import SvoFps # For filter curves
from astropy.io import fits
from astropy.table import QTable, Table
from dust_extinction.parameter_averages import G23
from synphot import ExtinctionCurve, Observation, SpectralElement, SourceSpectrum, units
from synphot.models import BlackBodyNorm1D, Empirical1D

BANDS = {
    'NUV':  ('GALEX/GALEX.NUV',  'AB'),
    'G':    ('GAIA/GAIA2r.G',    'Vega'),
    'BP':   ('GAIA/GAIA2r.Gbp',  'Vega'),
    'RP':   ('GAIA/GAIA2r.Grp',  'Vega'),
    'ps_g': ('PAN-STARRS/PS1.g', 'AB'),
    'ps_r': ('PAN-STARRS/PS1.r', 'AB'),
    'ps_i': ('PAN-STARRS/PS1.i', 'AB'),
    'ps_z': ('PAN-STARRS/PS1.z', 'AB'),
    'ps_y': ('PAN-STARRS/PS1.y', 'AB'),
}

# Columns for point_sources_from_spectra
REQ_COLS = ['ra', 'dec', 'ref', 'scale']

def load_bandpass(svo_id):
    table = SvoFps.get_transmission_data(svo_id)
    wave = u.Quantity(table['Wavelength'], u.AA)
    trans = np.asarray(table['Transmission'], dtype=float)
    return SpectralElement(Empirical1D, points=wave, lookup_table=trans)

def reddened_blackbody(teff, ebv, wave, ext_model):
    bb = SourceSpectrum(BlackBodyNorm1D, temperature=teff)
    trans = ext_model.extinguish(wave, Ebv=ebv)
    ext = ExtinctionCurve(Empirical1D, points=wave, lookup_table=trans)
    return bb * ext

def synthetic_mags(sp, bandpasses, vega):
    mags = []
    for name, (_, system) in BANDS.items():
        obs = Observation(sp, bandpasses[name])
        if system == 'AB':
            m = obs.effstim(flux_unit=u.ABmag)
        else:
            m = obs.effstim(flux_unit=units.VEGAMAG, vegaspec=vega)
        mags.append(m.value)
    return np.asarray(mags)

def mag_err(flux_snr, floor):
    return np.hypot(2.5 / np.log(10) / flux_snr, floor)

def build_mag_grid(teff_grid, ebv_grid, bandpasses, vega, wave, ext_model):
    mag_grid = np.empty((len(teff_grid), len(ebv_grid), len(BANDS)))
    for i, teff in enumerate(teff_grid):
        for j, ebv in enumerate(ebv_grid):
            spectrum = reddened_blackbody(teff, ebv, wave, ext_model)
            mag_grid[i, j] = synthetic_mags(spectrum, bandpasses, vega)
    return mag_grid

def fit_teff_ebv(obs_mag, obs_mag_err, mag_grid,
               ebv_grid=None, ebv_prior=None, ebv_prior_err=None):
    
    finite = np.isfinite(obs_mag) & np.isfinite(obs_mag_err)
    weight = np.where(finite, 1.0 / obs_mag_err**2, 0.0)
    residual = np.where(finite, obs_mag - mag_grid, 0.0)
    offset = np.sum(weight * residual, axis=-1) / np.sum(weight)
    chi2 = np.sum(weight * (residual - offset[..., None])**2, axis=-1)

    if ebv_prior is not None:
        chi2 = chi2 + ((ebv_grid - ebv_prior) / ebv_prior_err)**2

    i_best, j_best, = np.unravel_index(np.argmin(chi2), chi2.shape)

    return i_best, j_best, offset[i_best, j_best], chi2

def make_inputs(ra, dec,
                i_teff_fit, j_ebv_fit, flux_scale,
                teff_grid, ebv_grid, wave, ext_model):
    
    flux_scale = np.asarray(flux_scale, dtype=float)
    fitted = (i_teff_fit >= 0) & (j_ebv_fit >= 0) & np.isfinite(flux_scale)
    refs = np.array([f'BB_T{teff_grid[i]:.0f}_E{ebv_grid[j]:.3f}'
                     for i, j in zip(i_teff_fit[fitted], j_ebv_fit[fitted])])

    unique_grid_points = np.unique(np.column_stack([i_teff_fit[fitted], j_ebv_fit[fitted]]), axis=0)

    spectra = {f'BB_T{teff_grid[i]:.0f}_E{ebv_grid[j]:.3f}':
               reddened_blackbody(teff_grid[i], ebv_grid[j], wave, ext_model)
               for i, j in unique_grid_points}

    point_sources = QTable([ra[fitted].to(u.deg), dec[fitted].to(u.deg),
                            refs, flux_scale[fitted]],
                           names=['ra', 'dec', 'ref', 'scale'])
    return point_sources, spectra

def save_inputs(filename, point_sources, spectra, wave, overwrite=False):
    if point_sources.colnames != REQ_COLS:
        raise ValueError(f'Columns must be {REQ_COLS}, '
                         f'got {point_sources.colnames}')
    
    sources = QTable(point_sources, copy=True)
    sources.meta = {}
    n_rows = len(point_sources)
    for key, value in point_sources.meta.items():
        if key in REQ_COLS:
            raise ValueError(f"Metadata key '{key}' clashes with a required column name.")
        if np.ndim(value) > 0 and len(value) == n_rows:
            sources[key] = value
        elif np.isscalar(value):
            sources.meta[key] = value
        else:
            warnings.warn(f"Metadata ['{key}'] is neither a per-row array nor a scalar.")

    refs = list(spectra.keys())
    flux = np.array([spectra[r](wave, flux_unit=units.FLAM).value
                     for r in refs])
    spec_table = Table({'ref': refs, 'flux': flux})
    spec_table['flux'].unit = units.FLAM
    wave_table = Table({'wave': wave.to_value(u.AA)})
    wave_table['wave'].unit = u.AA

    hdus = fits.HDUList([fits.PrimaryHDU(),
                         fits.table_to_hdu(sources),
                         fits.table_to_hdu(spec_table),
                         fits.table_to_hdu(wave_table)])
    
    for hdu, name in zip(hdus[1:], ['SOURCES', 'SPECTRA', 'WAVE']):
        hdu.name = name

    hdus.writeto(filename, overwrite=overwrite)

def load_inputs(filename):
    '''
    EXAMPLE:
        save_survey_inputs('field4996_uvex_inputs.fits', point_sources, spectra, wave)
        point_sources_from_spectra, spectra = load_survey_inputs('field4496_survey_inputs.fits')
        print(point_sources_from_spectra.masked) # NOTE: Should be FALSE
    '''
    sources = QTable.read(filename, hdu='SOURCES')
    sources.meta.pop('EXTNAME', None)
    missing = [c for c in REQ_COLS if c not in sources.colnames]
    if missing:
        raise ValueError(f'{filename} is missing required column(s): {missing}')
    
    for name in [c for c in sources.colnames if c not in REQ_COLS]:
        col = sources[name]
        sources.meta[name] = col if isinstance(col, u.Quantity) else np.asarray(col)
        sources.remove_column(name)

    spec_table = Table.read(filename, hdu='SPECTRA')
    wave = Table.read(filename, hdu='WAVE')['wave'].quantity
    spectra = {str(r): SourceSpectrum(Empirical1D, points=wave,
                                      lookup_table=np.asarray(f) * units.FLAM)
               for r, f in zip(spec_table['ref'], spec_table['flux'])}
    return sources, spectra