# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Fri Nov  8 12:42:20 2024

@author: amartinez
"""

import numpy as np
import matplotlib.pyplot as plt
from compare_lists import compare_lists 
from astropy.table import Table
import skimage as ski
import sys
from astropy.stats import sigma_clip

from astropy.coordinates import SkyCoord
import Polywarp as pw

from astropy import units as u
from astropy.coordinates import search_around_sky
from collections import Counter
import pandas as pd
from matplotlib.colors import LogNorm
import os
import glob
from astropy.modeling.models import Polynomial2D
from astropy.modeling.fitting import LinearLSQFitter
from grid import grid_stars
from astropy.modeling.models import Polynomial2D
from astropy.modeling.fitting import LinearLSQFitter
from astropy.modeling import models, fitting
from alignator_relative import alg_rel
from astropy.time import Time
from astroquery.gaia import Gaia
import astroalign as aa
from filters import filter_gaia_data
from filters import filter_gns_data
from alignator_looping import alg_loop
from scipy import stats
# %% 
# %%plotting parametres
from matplotlib import rc
from matplotlib import rcParams
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'font.size': 20})
rcParams.update({'figure.figsize':(10,5)})
rcParams.update({
    "text.usetex": False,
    "font.family": "sans",
    "font.sans-serif": ["Palatino"]})
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
rc('font',**{'family':'serif','serif':['Palatino']})
plt.rcParams.update({'figure.max_open_warning': 0})# a warniing for matplot lib pop up because so many plots, this turining it of

# field_one, chip_one, field_two, chip_two,t1,t2,max_sig = np.loadtxt('/Users/amartinez/Desktop/PhD/HAWK/GNS_1relative_python/lists/fields_and_chips.txt', 
#                                                        unpack=True)
# field_one = field_one.astype(int)
# chip_one = chip_one.astype(int)
# field_two = field_two.astype(int)
# chip_two = chip_two.astype(int)

# sys.exit(75)

def sig_f(x, y,s):
    mx, lx, hx = sigma_clip(x , sigma = s, masked = True, return_bounds= True)
    my, ly, hy = sigma_clip(y , sigma = s, masked = True, return_bounds= True)
    m_xy = np.logical_and(np.logical_not(mx.mask),np.logical_not(my.mask))
    
    return m_xy, [lx,hx,ly,hy]

f1A = 'B1'
f1B = 'B6'
f2A = 20
e_pm_gns = 1
H_min = 22
destination  = 1
gaia_sep = 50*u.mas
pruebas1 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1relative_SUPER/pruebas/'
pruebas2 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2relative_SUPER/pruebas/'

# gnsA = Table.read(pruebas2 +f'gns2_pmSuper_F1_{f1A}_F2_{f2A}.ecsv', format = 'ascii.ecsv')
# gnsB = Table.read(pruebas2 +f'gns2_pmSuper_F1_{f1B}_F2_{f2A}.ecsv', format = 'ascii.ecsv')
gnsA = Table.read(pruebas1 +f'gns1_pmSuper_F1_{f1A}_F2_{f2A}.ecsv', format = 'ascii.ecsv')
# gnsB = Table.read(pruebas1 +f'gns1_pmSuper_F1_{f1B}_F2_{f2A}.ecsv', format = 'ascii.ecsv')

fig, ax = plt.subplots()
ax.scatter(gnsA['l'], gnsA['b'], label = f'F1_{f1A}_F2_{f2A}',s = 0.01)


fig, (axpm1, axpm2) = plt.subplots(1,2)
axpm1.hist2d(gnsA['H'], gnsA['dpm_x'], bins =100, norm = LogNorm())
axpm2.hist2d(gnsA['H'], gnsA['dpm_y'], bins =100, norm = LogNorm())
axpm1.set_ylabel('$\sigma \mu_{x}$[mas/yr]')
axpm2.set_ylabel('$\sigma \mu_{x}$[mas/yr]')
axpm1.set_xlabel('H')
axpm2.set_xlabel('H')
axpm1.axhline(e_pm_gns, color = 'red', ls = 'dashed', label = '$\sigma$pm < 1 mas/yr')
axpm1.legend()
fig.tight_layout()

gnsA = filter_gns_data(gnsA, max_e_pm = e_pm_gns, min_mag = H_min)

ax.scatter(gnsA['l'], gnsA['b'], label = 'Cut', s = 1)
ax.legend(markerscale =10)



# # %

# cA = SkyCoord(l = gnsA['l'], b = gnsA['b'], frame = 'galactic')

# fig, ax = plt.subplots(1,1)
# ax.sactter()


# %% Gaia Comparation#!!!
max_sep_ga = 50*u.mas# separation for comparison with gaia
e_pm = 0.3#!!! Maximun error in pm for Gaia stars
gaia_mags = [1,20]#!!! Gaia mag limtis for comparison with GNS
# %
# Before comparing witg Gaia we mask the best pms



extra_mag_cut = [12,20]

radius = abs(np.min(gnsA['l'])-np.max(gnsA['l']))*0.6*u.degree
center_g = SkyCoord(l = np.mean(gnsA['l']), b = np.mean(gnsA['b']), unit = 'degree', frame = 'galactic')

# try:
    
#     gaia = Table.read(pruebas1  + 'PM_gaia_f1%s_f2%s_r%.0f.ecsv'%(f1A,f2A,radius.to(u.arcsec).value))
#     print('Gaia from table')
# except:
#     print('Gaia from web')
#     center = SkyCoord(l = np.mean(gnsA['l']), b = np.mean(gnsA['b']), unit = 'degree', frame = 'galactic').icrs

#     Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source" # Select early Data Release 3
#     Gaia.ROW_LIMIT = -1  # it not especifty, Default rows are limited to 50. 
#     j = Gaia.cone_search_async(center, radius = abs(radius))
#     gaia = j.get_results()
#     gaia.write(pruebas1  + 'PM_gaia_f1%s_f2%s_r%.0f.ecsv'%(f1A,f2A,radius.to(u.arcsec).value))

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source" # Select early Data Release 3
Gaia.ROW_LIMIT = -1  # it not especifty, Default rows are limited to 50. 
j = Gaia.cone_search_async(center_g, radius = abs(radius))
gaia = j.get_results()


gaia = filter_gaia_data(
    gaia_table=gaia,
    astrometric_params_solved=31,
    duplicated_source= False,
    parallax_over_error_min=-10,
    astrometric_excess_noise_sig_max=2,
    phot_g_mean_mag_min= gaia_mags[1],
    phot_g_mean_mag_max=gaia_mags[0],
    pm_min=0,
    pmra_error_max=e_pm,
    pmdec_error_max=e_pm,
    min_angular_separation_arcsec = 0.1*u.arcsec
    )





gfig, (gax, gax2) = plt.subplots(1,2)
gax.scatter(gaia['phot_g_mean_mag'],gaia['pmra_error'],color = 'k')
gax.scatter(gaia['phot_g_mean_mag'],gaia['pmdec_error'], color = 'grey')
gax2.scatter(gaia['phot_g_mean_mag'],gaia['ra_error'],color = 'k')
gax2.scatter(gaia['phot_g_mean_mag'],gaia['dec_error'],color = 'grey')
gax.set_ylabel('$\sigma \mu$')
gax2.set_ylabel('$\sigma pos$')
gax.set_xlabel('[G]')
gax2.set_xlabel('[G]')
gfig.tight_layout()



# Step 1: Gaia proper motion as SkyCoord
c_gaia = SkyCoord(ra=gaia['ra'], dec=gaia['dec'],
                  pm_ra_cosdec=gaia['pmra'],
                  pm_dec=gaia['pmdec'],
                  obstime="J2016")

cg_gaia = SkyCoord(ra=gaia['ra'], dec=gaia['dec'],
                  pm_ra_cosdec=gaia['pmra'],
                  pm_dec=gaia['pmdec'],
                  obstime="J2016").galactic

gaia['pml'] = cg_gaia.pm_l_cosb
gaia['pmb'] = cg_gaia.pm_b

# Step 2: Convert to Cartesian offset using the tangent-plane projection
# Use a fixed reference point (e.g., center of your field)
# ref = SkyCoord(ra0*u.deg, dec0*u.deg)
# center = SkyCoord(ra = np.mean(gaia['ra']), dec = np.mean(gaia['dec']),unit = 'degree', frame='icrs')
center = SkyCoord(l = np.mean(gaia['l']), b = np.mean(gaia['b']),unit = 'degree', frame='galactic')
offset_frame = center.skyoffset_frame()

# c_proj = cg_gaia.transform_to(offset_frame)
c_proj = cg_gaia.transform_to(offset_frame)

# Step 3: Extract Gaia PM in tangent plane (same as your XY frame, in mas/yr)
pm_x_gaia = c_proj.pm_lon_coslat  # mas/yr
pm_y_gaia = c_proj.pm_lat        # mas/yr

gaia['pm_x'] = pm_x_gaia
gaia['pm_y'] = pm_y_gaia



mu_eq = np.hypot(gaia['pmra'], gaia['pmdec'])
mu_pr = np.hypot(gaia['pm_x'], gaia['pm_y'])

resid_mu = mu_eq - mu_pr
print('Differnces between Gaia pm and Gaia pm proyected')
print('Comoponete are differente, but modules of the vetors invariant under transformations')
print("σ(Δμ):", np.std(resid_mu))
print("mean(Δμ):", np.mean(resid_mu))
print("mean(μl - μl_p):", np.mean(gaia['pml'] - gaia['pm_x']))
print("mean(μb - μb_p):",np.mean(gaia['pmb'] - gaia['pm_y']))
print("σ(μl - μl_p):", np.std(gaia['pml'] - gaia['pm_x']))
print("σ(μb - μb_p):",np.std(gaia['pmb'] - gaia['pm_y']))







gnsc = SkyCoord(l = gnsA['l'], b = gnsA['b'], frame = 'galactic')
if destination == 1:
    
    # idx,d2d,d3d = gns1_gal.match_to_catalog_sky(gaia_c)
    # sep_constraint = d2d < max_sep_ga
    # gns_ga = gns1_mpm[sep_constraint]
    # ga_gns = gaia[idx[sep_constraint]]# ,nthneighbor=1 is for 1-to-1 match

    
    idx1, idx2, sep2d, _ = search_around_sky(gnsc, cg_gaia, gaia_sep)

    count1 = Counter(idx1)
    count2 = Counter(idx2)

    # Step 3: Create mask for one-to-one matches only
    mask_unique = np.array([
        count1[i1] == 1 and count2[i2] == 1
        for i1, i2 in zip(idx1, idx2)
    ])

    # Step 4: Apply the mask
    idx1_clean = idx1[mask_unique]
    idx2_clean = idx2[mask_unique]

    gns_m = gnsA[idx1_clean]
    gaia_m = gaia[idx2_clean]
    
ax.scatter(gaia_m['l'], gaia_m['b'])
# elif destination ==2:
#     # idx,d2d,d3d = gns2_gal.match_to_catalog_sky(gaia_c)
#     # sep_constraint = d2d < max_sep_ga
#     # gns_ga = gns2_mpm[sep_constraint]
#     # ga_gns = gaia[idx[sep_constraint]]# ,nthneighbor=1 is for 1-to-1 match

#     idx1, idx2, sep2d, _ = search_around_sky(gns2_gal, gaia_c, max_sep_ga)

#     count1 = Counter(idx1)
#     count2 = Counter(idx2)

#     # Step 3: Create mask for one-to-one matches only
#     mask_unique = np.array([
#         count1[i1] == 1 and count2[i2] == 1
#         for i1, i2 in zip(idx1, idx2)
#     ])

#     # Step 4: Apply the mask
#     idx1_clean = idx1[mask_unique]
#     idx2_clean = idx2[mask_unique]

#     gns_ga = gns2_mpm[idx1_clean]
#     ga_gns = gaia[idx2_clean]
    




d_pmx_ga = gaia_m['pm_x'] - gns_m['pm_x']
d_pmy_ga = gaia_m['pm_y'] - gns_m['pm_y']
# d_pmx_ga = gaia_m['pml'] - gns_m['pm_x']
# d_pmy_ga = gaia_m['pmb'] - gns_m['pm_y']

print(np.mean(d_pmx_ga), np.std(d_pmx_ga))
print(np.mean(d_pmy_ga), np.std(d_pmy_ga))



sig_ga = 3
m_pm, lxy = sig_f(d_pmx_ga, d_pmy_ga, sig_ga)


d_pmx_ga_m = d_pmx_ga[m_pm]
d_pmy_ga_m = d_pmy_ga[m_pm]


bins = 'auto'
fig, (ax,ax2) = plt.subplots(1,2)
ax2.set_title(f'Ref. epoch GNS{destination}. Matches {len(d_pmx_ga_m)}', fontsize = 15)
ax.hist(d_pmx_ga, bins = bins, color = 'k', alpha = 0.2)
ax2.hist(d_pmy_ga, bins = bins,color = 'k', alpha = 0.2)
ax.hist(d_pmx_ga_m, bins = bins, histtype = 'step', label = '$\overline{\mu}_{x}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(d_pmx_ga_m.value),np.std(d_pmx_ga_m.value)))
ax2.hist(d_pmy_ga_m, bins = bins, histtype = 'step',label = '$\overline{\mu}_{y}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(d_pmy_ga_m.value),np.std(d_pmy_ga_m.value)))
ax.set_xlabel('$\Delta \mu_{x}$ [mas/yr]')
ax2.set_xlabel('$\Delta\mu_{y}$ [mas/yr]')
ax.axvline(lxy[0], ls = 'dashed', color = 'r', label = f'{sig_ga}$\sigma$')
ax.axvline(lxy[1], ls = 'dashed', color = 'r')
ax2.axvline(lxy[2], ls = 'dashed', color = 'r')
ax2.axvline(lxy[3], ls = 'dashed', color = 'r')
ax.legend()
ax2.legend()

# %%

# =============================================================================
# Lets trye rotating µx and µy into µl and µb
# 
# Frist we define the rotation martrix between oour axis and ICRS 
# 
# R = np.array([[cosθ​, -sinθ​],
#               [sinθ​,  cosθ​]])
# Where θ​ is the rotation angles. The we calculate the proper motios µra* and µdec
# 
#  
# µICRS = R * np.array([[µl*],[µb]])
# 
# wher µICRS = np.array([[µra*],
#                             [µdec]])
# 
# Whe can obtain the theta from the WCS in the header:
#     
# wcs = WCS(header)  # or from a fits file
# theta = np.arctan2(wcs.pixel_scale_matrix[1,0], wcs.pixel_scale_matrix[0,0])  # radians
# 
# Or maybe using a similarity transformation to estinate the roation angle?
# 
# =============================================================================



















