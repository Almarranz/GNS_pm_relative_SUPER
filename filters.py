#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:13:35 2024

@author: amartinez
Filter astropy Table data based on provided criteria.

Returns:
- Filtered Astropy Table.
"""


# gaia_filters.py

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u 
from astropy.table import vstack

def filter_gaia_data(gaia_table, 
                     astrometric_params_solved=None, 
                     duplicated_source=None, 
                     parallax_over_error_min=None, 
                     astrometric_excess_noise_sig_max=None, 
                     phot_g_mean_mag_min=None, 
                     phot_g_mean_mag_max=None, 
                     pm_min=None, 
                     pmra_error_max=None, 
                     pmdec_error_max=None,
                     min_angular_separation_arcsec = None):
    
    mask = np.ones(len(gaia_table), dtype=bool)
    print('pmra_error_max',pmra_error_max)
    if astrometric_params_solved is not None:
        mask &= (gaia_table['astrometric_params_solved'] == astrometric_params_solved)
        
    if duplicated_source is not None:
        mask &= (gaia_table['duplicated_source'] == duplicated_source)
        
    if parallax_over_error_min is not None:
        mask &= (gaia_table['parallax_over_error'] >= parallax_over_error_min)
        
    if astrometric_excess_noise_sig_max is not None:
        mask &= (gaia_table['astrometric_excess_noise_sig'] <= astrometric_excess_noise_sig_max)
        
    if phot_g_mean_mag_min is not None:
        mask &= (gaia_table['phot_g_mean_mag'] < phot_g_mean_mag_min)
        
    if phot_g_mean_mag_max is not None:
        mask &= (gaia_table['phot_g_mean_mag'] > phot_g_mean_mag_max)
        
    if pm_min is not None:
        mask &= (gaia_table['pm'] > pm_min)
        
    if pmra_error_max is not None:
        mask &= (gaia_table['pmra_error'] < pmra_error_max)
        
    if pmdec_error_max is not None:
        mask &= (gaia_table['pmdec_error'] < pmdec_error_max)
    
    filtered_table = gaia_table[mask]
    
    # if min_angular_separation_arcsec is not None and len(filtered_table) > 1:
    #     coords = SkyCoord(ra=filtered_table['ra'],
    #                       dec=filtered_table['dec'])
        
    #     # Compute pairwise separations
    #     sep_matrix = coords[:, None].separation(coords[None, :]).to(u.arcsec)
        
    #     close_pairs = (sep_matrix < min_angular_separation_arcsec) & (sep_matrix > 0*u.arcsec)
       
    #     # Find indices involved in close pairs
    #     to_remove = np.unique(np.where(close_pairs)[0])

    #     # Remove stars involved in close pairs
    #     final_mask = np.ones(len(filtered_table), dtype=bool)
    #     final_mask[to_remove] = False
    #     filtered_table = filtered_table[final_mask]

    return filtered_table



    
def filter_hosek_data(hosek_table,
                      max_e_pos = None,
                      max_e_pm = None,
                      min_mag = None,
                      max_mag = None,
                      max_Pclust = None,
                      center = None):

    mask = np.ones(len(hosek_table), dtype=bool)
    
    if max_e_pos is not None:
        mask &= (hosek_table['e_dRA'] < max_e_pos) & (hosek_table['e_dDE'] < max_e_pos) 
        
    if max_e_pm is not None:
        mask &= (hosek_table['e_pmRA']< max_e_pm) & (hosek_table['e_pmDE']< max_e_pm)
        
    if min_mag is not None:
        mask &= (hosek_table['F127M'] < min_mag)
        
    if max_mag is not None:
        mask &= (hosek_table['F127M'] > max_mag)
        
    if max_Pclust is not None:
        mask &= (hosek_table['Pclust'] < max_Pclust)
        
    if center is not None:
        mask &= (hosek_table['F127M'] - hosek_table['F153M'] > 1.7)
        
    
    return hosek_table[mask]
    

def filter_gns_data(gns_table,
                      max_e_pos = None,
                      max_e_pm = None,
                      min_mag = None,
                      max_mag = None,
                      ):

    mask = np.ones(len(gns_table), dtype=bool)
    
    if max_e_pos is not None:
        mask &= (gns_table['sl'] < max_e_pos) & (gns_table['sb'] < max_e_pos) 
    
    if min_mag is not None:
        mask &= (gns_table['H'] < min_mag) 
    
    if max_mag is not None:
        mask &= (gns_table['H'] > max_mag) 
        
    if max_e_pm is not None:
        mask &= (gns_table['dpm_x'] < max_e_pm) & (gns_table['dpm_y'] < max_e_pm)

    return gns_table[mask]


def filter_gns_by_percentile(gns_table, mag_col='H', err_col='dH', sl_col='sl', sb_col='sb', bin_width=None, percentile_H = None, percentile_lb = None, mag_lim = None, pos_lim = None):
    """
    Filter stars in magnitude bins based on the 85th percentile of photometric and positional uncertainties.

    Parameters
    ----------
    gns_table : astropy.table.Table
        The input table with columns for magnitude, uncertainty, and position errors.
    mag_col : str
        Name of the magnitude column (e.g., 'H').
    err_col : str
        Name of the magnitude uncertainty column (e.g., 'dH').
    sl_col, sb_col : str
        Names of the positional uncertainty columns.
    bin_width : float
        Width of the magnitude bins.
    percentile : float
        Percentile cutoff (default is 85).

    Returns
    -------
    filtered_table : astropy.table.Table
        Table with rows that pass the percentile thresholding.
    """
    # Assuming gns1 is your input Table
    H = gns_table['H']
    pos_err = np.sqrt(gns_table['sl']**2 + gns_table['sb']**2)

    # Add temporary column for position uncertainty
    gns_table['pos_err'] = pos_err

    # Define bins
    H_min, H_max = np.nanmin(H), np.nanmax(H)
    bins = np.arange(H_min, H_max + bin_width, bin_width)

    # Container for filtered results
    filtered_tables = []

    # Iterate over H-magnitude bins
    for i in range(len(bins)-1):
        bin_mask = (H >= bins[i]) & (H < bins[i+1])
        bin_data = gns_table[bin_mask]

        if len(bin_data) == 0:
            continue

        # Compute 85th percentiles
        dH_thresh = np.percentile(bin_data['dH'], percentile_H)
        pos_thresh = np.percentile(bin_data['pos_err'], percentile_lb)
        l_thresh = np.percentile(bin_data['sl'], percentile_lb)
        b_thresh = np.percentile(bin_data['sb'], percentile_lb)

        # Apply selection criteria
        good_mask = (bin_data['dH'] <= dH_thresh) & (bin_data['sl'] <= l_thresh) & (bin_data['sb'] <= b_thresh)
        filtered_tables.append(bin_data[good_mask])
    
    
    filtered_table = vstack(filtered_tables)
    
    if mag_lim is not None:
        mH = filtered_table['H'] < mag_lim
        filtered_table = filtered_table[mH]
    if pos_lim is not None:
        
        unc_cut = np.where((filtered_table['sl']<pos_lim) & (filtered_table['sb']<pos_lim))
        filtered_table  = filtered_table[unc_cut]
        
    
    return filtered_table
    

def filter_vvv_data(vvv_table,
                    pmRA = None,
                    pmDE = None,
                    epm = None,
                    ok = None,
                    max_Ks = None,
                    min_Ks = None,
                    J = None,
                    center = None
                    ):
    mask = np.ones(len(vvv_table), dtype = bool)
        
    if center is not None:
        mask &= (vvv_table['J'] - vvv_table['Ks1'] > 2)
        
    if pmRA is not None:
        mask &= (vvv_table['pmRA'] < 900)
        
    if max_Ks is not None:
        mask &= (vvv_table['Ks'] > max_Ks)
        
    if min_Ks is not None:
        mask &= (vvv_table['Ks'] < min_Ks)
        
    if epm is not None:
        mask &= (vvv_table['epmRA'] <epm) & (vvv_table['epmDEC'] <epm)
        
    if ok is not None:
        mask &= (vvv_table['ok'] != 0)
        
        
    return vvv_table[mask]

    
    
    