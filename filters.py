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
    
    if min_angular_separation_arcsec is not None and len(filtered_table) > 1:
        coords = SkyCoord(ra=filtered_table['ra'],
                          dec=filtered_table['dec'])
        
        # Compute pairwise separations
        # sep_matrix = coords.separation(coords).to(u.arcsec)
        sep_matrix = coords[:, None].separation(coords[None, :]).to(u.arcsec)
        # Create a boolean mask to identify stars to keep
        # Initially, all are assumed valid
        # close_pairs = (sep_matrix < min_angular_separation_arcsec) & (sep_matrix > 0*u.arcsec)
        print(sep_matrix)
        # close_pairs = (sep_matrix < min_angular_separation_arcsec) & (sep_matrix > 0*u.arcsec)
        close_pairs = (sep_matrix < min_angular_separation_arcsec) & (sep_matrix > 0*u.arcsec)
        print(close_pairs)

        # Find indices involved in close pairs
        to_remove = np.unique(np.where(close_pairs)[0])

        # Remove stars involved in close pairs
        final_mask = np.ones(len(filtered_table), dtype=bool)
        final_mask[to_remove] = False
        filtered_table = filtered_table[final_mask]

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

    
    
    