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

pruebas1 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1relative_SUPER/pruebas/'
pruebas2 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2relative_SUPER/pruebas/'

# gnsA = Table.read(pruebas2 +f'gns2_pmSuper_F1_{f1A}_F2_{f2A}.ecsv', format = 'ascii.ecsv')
# gnsB = Table.read(pruebas2 +f'gns2_pmSuper_F1_{f1B}_F2_{f2A}.ecsv', format = 'ascii.ecsv')
gnsA = Table.read(pruebas1 +f'gns1_pmSuper_F1_{f1A}_F2_{f2A}.ecsv', format = 'ascii.ecsv')
gnsB = Table.read(pruebas1 +f'gns1_pmSuper_F1_{f1B}_F2_{f2A}.ecsv', format = 'ascii.ecsv')

e_pm_gns = 0.4
H_min = 22
gnsA = filter_gns_data(gnsA, max_e_pm = e_pm_gns, min_mag = H_min)
gnsB = filter_gns_data(gnsB, max_e_pm = e_pm_gns, min_mag = H_min)

fig, ax = plt.subplots()
ax.scatter(gnsA['l'], gnsA['b'], label = f'F1_{f1A}_F2_{f2A}')
ax.scatter(gnsB['l'], gnsB['b'], s = 1,label = f'F1_{f1B}_F2_{f2A}')
ax.legend()

# %

cA = SkyCoord(l = gnsA['l'], b = gnsA['b'], frame = 'galactic')
cB = SkyCoord(l = gnsB['l'], b = gnsB['b'], frame = 'galactic')

mat_dis = 50*u.mas
idx1, idx2, sep2d, _ = search_around_sky(cA, cB, mat_dis)

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

gnsA_m= gnsA[idx1_clean]
gnsB_m = gnsB[idx2_clean]


diff_H = gnsB_m['H']-gnsA_m['H']
sig_cl_H = 3
mask_H, l_lim,h_lim = sigma_clip(diff_H, sigma=sig_cl_H, masked = True, return_bounds= True, maxiters= 50)

gnsB_m = gnsB_m[np.logical_not(mask_H.mask)]
gnsA_m = gnsA_m[np.logical_not(mask_H.mask)]
# gns2_m = gnsB_m[np.logical_not(mask_H.mask)]
# gns1_m = gnsA_m[np.logical_not(mask_H.mask)]



fig,(ax,ax2) = plt.subplots(1,2)
ax.set_title(f'Matching starts {len(gnsA_m)}')
ax.hist(diff_H[np.logical_not(mask_H.mask)], bins = 'auto',histtype = 'step')
ax.hist(diff_H, bins = 'auto', color = 'grey', alpha = 0.3)
ax.axvline(np.mean(diff_H), color = 'k', ls = 'dashed', label = '$\overline{\Delta H}$= %.2f$\pm$%.2f'%(np.mean(diff_H),np.std(diff_H)))
ax.axvline(l_lim, ls = 'dashed', color ='r', label ='%s$\sigma$'%(sig_cl_H))

# ax2.hist2d(gns2_match['H'],diff_H[np.logical_not(mask_H.mask)], bins = 100, norm = LogNorm())
ax2.hist2d(gnsA_m['H'],diff_H[np.logical_not(mask_H.mask)], bins = 20, norm = LogNorm())
ax2.set_ylim(-0.3,0.3)
ax.axvline(h_lim, ls = 'dashed', color ='r')
ax.set_xlabel('$\Delta H$')
ax.legend() 

# %%


dmux = gnsA_m['pm_x'] - gnsB_m['pm_x'] 
dmuy = gnsA_m['pm_y'] - gnsB_m['pm_y'] 

sig_cl_pm = 3

m_pm, lims = sig_f(dmux, dmuy, sig_cl_pm)

dmux_m = dmux[m_pm]
dmuy_m = dmuy[m_pm]

# gns2_m = gnsB_m[np.logical_not(m_pm.mask)]
# gns1_m = gnsA_m[np.logical_not(m_pm.mask)]

fig,(ax,ax2) = plt.subplots(1,2)
ax2.set_title(f'mat.dist = {mat_dis}')
ax.set_title(f'Matching stars = {len(dmux_m)}')
ax.hist(dmux_m, bins = 'auto', histtype = 'step', label = '$\overline{\Delta \mu_{x}}$ = %.2f\n$\sigma$ = %.2f'%(np.mean(dmux_m),np.std(dmux_m)))
ax2.hist(dmuy_m, bins = 'auto',histtype = 'step', label = '$\overline{\Delta \mu_{x}}$ = %.2f\n$\sigma$ = %.2f'%(np.mean(dmuy_m),np.std(dmuy_m)))
ax.hist(dmux, bins = 'auto', color = 'k', alpha = 0.2)
ax2.hist(dmuy, bins = 'auto',color = 'k', alpha = 0.2)
ax.legend()
ax2.legend()

# %%















































