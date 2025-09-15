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
from filters import filter_gns_by_percentile
from alignator_looping import alg_loop
from scipy import stats
from matplotlib.ticker import FormatStrFormatter
import gns_cluster_finder
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
# field_one = 'B1'
# field_two = 20

field_one = 16 #GNS1
field_two = 7 #GNS2

chipA = 2 #GNS1
chipB = 1 #GNS2



if field_one == 7 or field_one == 12 or field_one == 10 or field_one == 16:
    t1 = Time(['2015-06-07T00:00:00'],scale='utc')
elif field_one == 'B6':
    t1 = Time(['2016-06-13T00:00:00'],scale='utc')
elif field_one ==  'B1':
    t1 = Time(['2016-05-20T00:00:00'],scale='utc')
else:
    print(f'NO time detected for this field_one = {field_one}')
    sys.exit()
if field_two == 7 or field_two == 5:
    t2 = Time(['2022-05-27T00:00:00'],scale='utc')
elif field_two == 4:
    t2 = Time(['2022-04-05T00:00:00'],scale='utc')
elif field_two == 20:
    t2 = Time(['2022-07-25T00:00:00'],scale='utc')
else:
    print(f'NO time detected for this field_two = {field_two}')
    sys.exit()

dt = t2 - t1

# sys.exit(87)
# for chip_one in range(1,2,1):


color = pd.read_csv('/Users/amartinez/Desktop/PhD/python/colors_html.csv')
morralla = '/Users/amartinez/Desktop/morralla/'
strin= color.values.tolist()
indices = np.arange(0,len(strin),1)

# =============================================================================
# VARIABLES
# =============================================================================
# center_only = 'yes'#TODO yes, eliminates foregroud, no, keep them
center_only = 'no'
# pix_scale = 0.1064*0.5
# pix_scale = 0.1064
# max_sig = 0.3#TODO


# =============================================================================
# QUALITY CUTS
# =============================================================================
gns_mags = [10, 18]#!!! GNS mag limtis
max_sig = 0.05# arcsec Max uncertainty position (l,b)
bin_width = 0.1
perc_H = 100
perc_lb = 100
# =============================================================================
# GRID PARAMS
# =============================================================================
grid_s = None
# grid_s = 100
grid_Hmin = 15
grid_Hmax = 18
isolation_radius = 1#arcsec isolation of the grid stars 

# =============================================================================
# ALIGNMENT PARAMS
# =============================================================================
max_loop = 3
mag_lim_alig = None# H Limits of the algnment stars
# mag_lim_alig = [12, 14]# H Limits of the algnment stars
max_sep = 30* u.mas# firts match gns1 to gns2 for astroaling
sig_cl = 3#!!!
max_deg =3
centered_in = 2
d_m = 30*u.mas#!!!max distance  for the fine alignment betwenn GNS1 and 2
destination = 2 #!!! GNS2 is reference
# destination = 1 #!!! GNS1 is reference
align_by = 'Polywarp'#!!!
# align_by = '2DPoly'#!!!
f_mode = 'W' # f_mode only useful for 2Dpoly
# f_mode = 'WnC'
# f_mode = 'NW'
# f_mode = 'NWnC'

# =============================================================================
# PMs PARAMS
# =============================================================================
d_m_pm = 0.15#!!! in arcs, max distance for the proper motions
e_pm_gns = 2#!!!error cut in proper motions


look_for_cluster = 'yes'


def sig_f(x, y,s):
    mx, lx, hx = sigma_clip(x , sigma = s, masked = True, return_bounds= True)
    my, ly, hy = sigma_clip(y , sigma = s, masked = True, return_bounds= True)
    m_xy = np.logical_and(np.logical_not(mx.mask),np.logical_not(my.mask))
    
    return m_xy, [lx,hx,ly,hy]
# %% 

pruebas1 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1relative_SUPER/pruebas/'
pruebas2 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2relative_SUPER/pruebas/'


# gns1 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/pruebas/F{field_one}_old/{field_one}_H_chips_opti.ecsv',  format = 'ascii.ecsv')


gns1 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/pruebas/F{field_one}/stars_chip{chipA}_opti.txt',  format = 'ascii')
gns1.rename_column('sm','dH')
gns1.rename_column('m','H')
gns1['sl'].unit = u.arcsec
gns1['sb'].unit = u.arcsec
gns1['l'].unit = u.deg
gns1['b'].unit = u.deg


gns1 = filter_gns_by_percentile(gns1, mag_col='H', err_col='dH', sl_col='sl', sb_col='sb', bin_width=bin_width, percentile_H=perc_H, percentile_lb=perc_lb, mag_lim = None, pos_lim = None)



# %%
# fig, (axa, axb)= plt.subplots(1,2, figsize =(21,7))
# # fig, (axa, axb)= plt.subplots(2,1, figsize =(7,10))
# hisa = axa.hist2d(gns1['H'],np.sqrt(gns1['sl']**2 + gns1['sb']**2) , bins = 100,norm = LogNorm())
# axa.set_ylabel('Position uncertainty (ℓ, b) [arcsec]')
# axa.set_xlabel('GNS I [H]')
# axa.set_ylim(0.001,0.4)
# fig.colorbar(hisa[3], ax =axa, fraction = 0.05, aspect = 30, label = '# stars')

# num_bins = 100
# statistic, x_edges, y_edges, binnumber = stats.binned_statistic_2d(gns1['l'], gns1['b'], np.sqrt(gns1['sl']**2 + gns1['sb']**2), statistic='median', bins=(num_bins,int(num_bins/2)))
# # Create a meshgrid for plotting
# X, Y = np.meshgrid(x_edges, y_edges)
# # Plot the result
# # c = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r', norm = LogNorm() ) 
# c = axb.pcolormesh(X, Y, statistic.T, cmap='Spectral_r') 
# # fig.colorbar(c, ax=ax, label='$\sqrt {\delta l^{2} + \delta b^{2}}$ [arcsec]', shrink = 1)
# cb = fig.colorbar(c, ax=axb,  fraction = 0.05, aspect = 30)
# cb.set_label('Position uncertainty (ℓ, b) [arcsec]', fontsize=20, labelpad = 20) 
# # ax.set_title(f'GNS1 Max $\delta$ posit = {max_sig}. Max mag = {gns_mags[1]}')
# axb.set_xlabel('l')
# axb.set_ylabel('b')
# # ax.axis('scaled')
# axb.axis('equal')
# fig.tight_layout()
# axb.scatter(ga_gns['l'], ga_gns['b'], marker = '*', edgecolor = 'k', label = 'Gaia Stars',s = 200)
# axb.legend()
# meta = {'Script': '/Users/amartinez/Desktop/PhD/HAWK/GNS_pm_scripts/GNS_pm_relative_SUPER/SUPER_alignment.py'}
# # plt.savefig('/Users/amartinez/Desktop/PhD/My_papers/SgrB1_cluster/images/dpos_lb_H.png', bbox_inches='tight', pad_inches=0, dpi = 300, edgecolor = 'white', transparent = True, metadata = meta)


# meta = {'Script': '/Users/amartinez/Desktop/PhD/HAWK/GNS_pm_scripts/GNS_pm_relative_SUPER/SUPER_alignment.py'}
# plt.savefig('/Users/amartinez/Desktop/PhD/My_papers/SgrB1_cluster/images/dpos_H_gns1.png', bbox_inches='tight', pad_inches=0, dpi = 300, edgecolor = 'white', transparent = True, metadata = meta)


# %%



# gns2 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/pruebas/F{field_two}_old/{field_two}_H_chips_opti.ecsv', format = 'ascii.ecsv')


gns2 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/pruebas/F{field_two}/stars_chip{chipB}_opti.txt', format = 'ascii')
gns2.rename_column('sm','dH')
gns2.rename_column('m','H')
gns2['sl'].unit = u.arcsec
gns2['sb'].unit = u.arcsec
gns2['l'].unit = u.deg
gns2['b'].unit = u.deg
buenos1 = (gns1['l']>min(gns2['l'])) & (gns1['l']<max(gns2['l'])) & (gns1['b']>min(gns2['b'])) & (gns1['b']<max(gns2['b']))

gns1 = gns1[buenos1]
gns1['ID'] = np.arange(len(gns1))
all_1 = len(gns1)

# %%
# fig, ax = plt.subplots()
# # ax.scatter(gns2_mpm['l'], gns2_mpm['b'], s= 1)
# # hst = ax.hist2d(gns1['l'], gns1['b'], bins = 100)
# hst = ax.hist2d(gns2['l'], gns2['b'], bins = 50)
# fig.colorbar(hst[3], ax = ax)
# ax.invert_xaxis()
# sys.exit(249)
# %%


fig, (ax, ax2) = plt.subplots(1,2, figsize = (20,10))
ax.hist2d(gns1['H'],gns1['sl'], bins = 100,norm = LogNorm())
his = ax2.hist2d(gns1['H'],gns1['sb'], bins = 100,norm = LogNorm())
fig.colorbar(his[3], ax =ax2)
ax.set_title(f'GNS1 ({len(gns1)})')
ax.set_ylabel('$\sigma l$ [arcsec]')
ax2.set_ylabel('$\sigma b$ [arcsec]')
ax.set_xlabel('[H]')
ax2.set_xlabel('[H]')
fig.tight_layout()
ax.axhline(max_sig,ls = 'dashed', color = 'r')
if gns_mags[1] is not None:
    ax.axvline(gns_mags[1],ls = 'dashed', color = 'r')
    
    
fig, ax = plt.subplots(1,1)
hits = ax.hist2d(gns1['H'],(gns1['sl'] +gns1['sl'])/2,  bins = 100,norm = LogNorm())
fig.colorbar(his[3], ax =ax)
ax.set_ylabel('($\sigma l + + \sigma b$)/2 [arcsec]')
ax.set_xlabel('[H]')
fig.tight_layout()
    
    

buenos2 = (gns2['l']>min(gns1['l'])) & (gns2['l']<max(gns1['l'])) & (gns2['b']>min(gns1['b'])) & (gns2['b']<max(gns1['b']))

gns2 = gns2[buenos2]
gns2['ID'] = np.arange(len(gns2))
gns2 = filter_gns_by_percentile(gns2, mag_col='H', err_col='dH', sl_col='sl', sb_col='sb', bin_width=bin_width, percentile_H=perc_H, percentile_lb=perc_lb, mag_lim = None, pos_lim = None)




fig2, (ax_2, ax2_2) = plt.subplots(1,2, figsize = (16,8)) 
ax_2.set_title(f'GNS2 ({len(gns2)})')
ax_2.hist2d(gns2['H'],gns2['sl'],cmap = 'inferno', bins = (200),norm = LogNorm())
his = ax2_2.hist2d(gns2['H'],gns2['sb'],cmap = 'inferno', bins = 100,norm = LogNorm())
fig2.colorbar(his[3], ax =ax2_2)
ax_2.set_ylabel('$\delta l$ [arcsec]')
ax2_2.set_ylabel('$\delta b$ [arcsec]')
ax2_2.set_xlabel('[H]')
ax_2.set_xlabel('[H]')
fig2.tight_layout()
ax_2.axhline(max_sig,ls = 'dashed', color = 'r')
if gns_mags[1] is not None:
    ax_2.axvline(gns_mags[1],ls = 'dashed', color = 'r')

ax_2.set_xlim(12,23)
ax_2.set_ylim(0.00,0.125)


all_2 = len(gns2)

gns1 = filter_gns_data(gns1, max_e_pos = max_sig, max_mag = gns_mags[0], min_mag = gns_mags[1] )

# %%
fig, ax = plt.subplots(1,1)
ax.scatter(gns2['l'],gns2['b'],label = 'gns2')
ax.scatter(gns1['l'],gns1['b'], label = 'gns1', color = 'r')
ax.axis('equal')

ax.legend()
# sys.exit(285)
# %%

# sys.exit('SYS_EXIT')
# %%

# Plot for article
# =============================================================================
# fig, (ax,ax2) = plt.subplots(1,2, figsize = (15,7.5))
# hist = ax.hist2d(gns1['H'],(gns1['sl'] +gns1['sl'])/2,  bins = 100,norm = LogNorm())
# fig.colorbar(hist[3], ax =ax)
# # ax.set_ylabel('($\sigma l + \sigma b$)/2 [arcsec]')
# ax.set_ylabel('$\overline{\sigma}_{(l,b)}$ [arcsec]')
# ax.set_xlabel('[H]')
# ax.set_ylim(0.001,0.07)
# hist2 = ax2.hist2d(gns2['H'],(gns2['sl'] +gns2['sl'])/2,  bins = 200,norm = LogNorm(), cmap = 'inferno')
# fig.colorbar(hist2[3], ax =ax2, label = 'stars/bin')
# # ax.set_ylabel('($\sigma l + \sigma b$)/2 [arcsec]')
# ax2.set_xlabel('[H]')
# ax2.set_ylim(0.001,0.07)
# ax2.set_xlim(12,23)
# ax.set_xlim(12,23)
# fig.tight_layout()
# 
# meta = {'Script': '/Users/amartinez/Desktop/PhD/HAWK/GNS_pm_scripts/GNS_pm_relative_SUPER/SUPER_alignment.py'}
# plt.savefig('/Users/amartinez/Desktop/PhD/My_papers/GNS_pm_catalog/images/GNS1-2_vs_sigmalb.png', bbox_inches='tight', dpi = 150, transparent = True, metadata = meta)
# 
# 
# sys.exit(279)
# =============================================================================
# %%


gns2 = filter_gns_data(gns2, max_e_pos = max_sig, max_mag = gns_mags[0], min_mag = gns_mags[1] )


ax2.set_title(f'Clipped {100 - 100*len(gns1)/all_1:.1f}%')
ax2_2.set_title(f'Clipped {100 - 100*len(gns2)/all_2:.1f}%')


if centered_in == 1:
    center = SkyCoord(l = np.mean(gns1['l']), b = np.mean(gns1['b']), unit = 'degree', frame = 'galactic')
elif centered_in == 2:
    center = SkyCoord(l = np.mean(gns2['l']), b = np.mean(gns2['b']), unit = 'degree', frame = 'galactic')
    

gns1_lb = SkyCoord(l = gns1['l'], b = gns1['b'], unit ='deg', frame = 'galactic')
gns2_lb = SkyCoord(l = gns2['l'], b = gns2['b'], unit ='deg', frame = 'galactic')

xg_1, yg_1 = center.spherical_offsets_to(gns1_lb)
xg_2, yg_2 = center.spherical_offsets_to(gns2_lb)

tag = center.skyoffset_frame()

gns1_t = gns1_lb.transform_to(tag)
gns2_t = gns2_lb.transform_to(tag)


gns1['xp'] = gns1_t.lon.to(u.arcsec)
gns1['yp'] = gns1_t.lat.to(u.arcsec)
gns2['xp'] = gns2_t.lon.to(u.arcsec)
gns2['yp'] = gns2_t.lat.to(u.arcsec)

# %%

fig, ax = plt.subplots(1,1)
ax.scatter(gns2['xp'], gns2['yp'])
ax.scatter(gns1['xp'], gns1['yp'])
ax.axis('equal')
# sys.exit(344)
# %%


num_bins = 100
statistic, x_edges, y_edges, binnumber = stats.binned_statistic_2d(gns1['l'], gns1['b'], np.sqrt(gns1['sl']**2 + gns1['sb']**2), statistic='median', bins=(num_bins,int(num_bins/2)))
# Create a meshgrid for plotting
X, Y = np.meshgrid(x_edges, y_edges)
# Plot the result
fig, ax = plt.subplots(figsize = (12,6))
# c = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r', norm = LogNorm() ) 
c = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r') 
# fig.colorbar(c, ax=ax, label='$\sqrt {\delta l^{2} + \delta b^{2}}$ [arcsec]', shrink = 1)
cb = fig.colorbar(c, ax=ax,  fraction = 0.05, aspect = 30)
cb.set_label('Position uncertainty (ℓ, b) [arcsec]', fontsize=20, labelpad = 20) 
# ax.set_title(f'GNS1 Max $\delta$ posit = {max_sig}. Max mag = {gns_mags[1]}')
ax.set_xlabel('l')
ax.set_ylabel('b')
# ax.axis('scaled')
ax.axis('equal')
# meta = {'Script': '/Users/amartinez/Desktop/PhD/HAWK/GNS_pm_scripts/GNS_pm_relative_SUPER/SUPER_alignment.py'}
# plt.savefig('/Users/amartinez/Desktop/PhD/My_papers/SgrB1_cluster/images/dpos_lb.png', bbox_inches='tight', pad_inches=0, dpi = 300, edgecolor = 'white', transparent = True, metadata = meta)


# sys.exit(275)
# %%


statistic, x_edges, y_edges, binnumber = stats.binned_statistic_2d(gns2['l'], gns2['b'], np.sqrt(gns2['sl']**2 + gns2['sb']**2), statistic='median', bins=(num_bins,int(num_bins/2)))
# Create a meshgrid for plotting
X, Y = np.meshgrid(x_edges, y_edges)
# Plot the result
fig, ax = plt.subplots()
# c = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r', norm = LogNorm(vmax = max_sig*2) ) 
c = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r',vmin = 0, vmax = max_sig*1.5) 
fig.colorbar(c, ax=ax, label='$\sqrt {\delta l^{2} + \delta b^{2}}$ [arcsec]', shrink = 1)
ax.set_title(f'GNS2 Max $\delta$ posit = {max_sig}. Max mag = {gns_mags[1]}')
ax.set_xlabel('l')
ax.set_ylabel('b')
ax.axis('equal')# %%
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#I cosider a math if the stars are less than 'max_sep' arcsec away 
# This is for cutting the the overlapping areas of both lists. (Makes astroaling work faster)
# sys.exit(237)

# idx,d2d,d3d = gns1_lb.match_to_catalog_sky(gns2_lb)# ,nthneighbor=1 is for 1-to-1 match
# sep_constraint = d2d < max_sep
# gns1_match = gns1[sep_constraint]
# gns2_match = gns2[idx[sep_constraint]]
# %%

idx1, idx2, sep2d, _ = search_around_sky(gns1_lb, gns2_lb, 50*u.mas)

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

gns1_match= gns1[idx1_clean]
gns2_match = gns2[idx2_clean]


diff_H = gns2_match['H']-gns1_match['H']
sig_cl_H = 3
mask_H, l_lim,h_lim = sigma_clip(diff_H, sigma=sig_cl_H, masked = True, return_bounds= True, maxiters= 50)

gns2_match = gns2_match[np.logical_not(mask_H.mask)]
gns1_match = gns1_match[np.logical_not(mask_H.mask)]



fig,(ax,ax2) = plt.subplots(1,2)
ax.set_title(f'Matching starts {len(gns2_match)}')
ax.hist(diff_H[np.logical_not(mask_H.mask)], bins = 'auto',histtype = 'step')
ax.hist(diff_H, bins = 'auto', color = 'grey', alpha = 0.3)
ax.axvline(np.mean(diff_H), color = 'k', ls = 'dashed', label = '$\overline{\Delta H}$= %.2f$\pm$%.2f'%(np.mean(diff_H),np.std(diff_H)))
ax.axvline(l_lim, ls = 'dashed', color ='r', label ='%s$\sigma$'%(sig_cl_H))

ax2.hist2d(gns2_match['H'],diff_H[np.logical_not(mask_H.mask)], bins = 100, norm = LogNorm())
ax2.set_ylim(-2,2)
ax.axvline(h_lim, ls = 'dashed', color ='r')
ax.set_xlabel('$\Delta H$')
ax.legend() 


# sys.exit(288)
# %%
g1_m = np.array([gns1_match['xp'],gns1_match['yp']]).T
g2_m = np.array([gns2_match['xp'],gns2_match['yp']]).T

sig_cl_H_aligment = sig_cl_H
if destination == 1:
    # Time lapse to move Gaia Stars.
    tg = Time(['2016-01-01T00:00:00'],scale='utc')
    dtg = t1 - tg
    
    # p,(_,_)= aa.find_transform(g2_m,g1_m,max_control_points=100)
    
    p = ski.transform.estimate_transform('similarity',
                                    g2_m, 
                                    g1_m)
    # 
    print("Translation: (x, y) = (%.2f, %.2f)"%(p.translation[0],p.translation[1]))
    print("Rotation: %.2f deg"%(p.rotation * 180.0/np.pi)) 
    print("Rotation: %.0f arcmin"%(p.rotation * 180.0/np.pi*60)) 
    print("Rotation: %.0f arcsec"%(p.rotation * 180.0/np.pi*3600)) 
    
    
    
    
    loop = 0
    comom_ls = []
    dic_xy = {}
    dic_Kx ={}
    dic_xy_final = {}
    
        
    gns2_xy = np.array((gns2['xp'],gns2['yp'])).T
    gns2_xyt = p(gns2_xy)
    
    s_ls = compare_lists(gns2_xyt, np.array([gns1['xp'],gns1['yp']]).T, d_m.to(u.arcsec).value)
    print(f'Common stars after astroaling similaryty:{len(s_ls)}')
    
    gns2['xp'] = gns2_xyt[:,0]
    gns2['yp'] = gns2_xyt[:,1]
    
    
    # gns2 = alg_rel(gns2, gns1,'xp', 'yp', align_by,use_grid,max_deg = max_deg, d_m = d_m,f_mode = f_mode,grid_s = grid_s )
    # def alg_loop(gns_A, gns_B,col1, col2, align_by, max_deg, d_m,                        max_loop,  use_grid,grid_s= None, f_mode = None  ) :
    gns1 = alg_loop(gns2, gns1, 'xp', 'yp', align_by, max_deg, d_m.to(u.arcsec).value, max_loop,sig_cl_H = sig_cl_H_aligment, 
                     grid_s = grid_s, grid_Hmin = grid_Hmin, grid_Hmax = grid_Hmax ,isolation_radius = isolation_radius,  f_mode = f_mode, mag_lim_alig=mag_lim_alig)

if destination == 2:
    # Time lapse to move Gaia Stars.
    tg = Time(['2016-01-01T00:00:00'],scale='utc')
    dtg = t2 - tg
    
    # p,(_,_)= aa.find_transform(g1_m,g2_m,max_control_points=100)
    
    p = ski.transform.estimate_transform('similarity',
                                    g1_m, 
                                    g2_m)
    
    print("Translation: (x, y) = (%.2f, %.2f)"%(p.translation[0],p.translation[1]))
    print("Rotation: %.2f deg"%(p.rotation * 180.0/np.pi)) 
    print("Rotation: %.0f arcmin"%(p.rotation * 180.0/np.pi*60)) 
    print("Rotation: %.0f arcsec"%(p.rotation * 180.0/np.pi*3600)) 
    
    loop = 0
    comom_ls = []
    dic_xy = {}
    dic_Kx ={}
    dic_xy_final = {}
    
        
    gns1_xy = np.array((gns1['xp'],gns1['yp'])).T
    gns1_xyt = p(gns1_xy)
    
    s_ls = compare_lists(gns1_xyt, np.array([gns2['xp'],gns2['yp']]).T, d_m.to(u.arcsec).value)
    print(f'Common stars after astroaling similaryty:{len(s_ls)}')
    
    gns1['xp'] = gns1_xyt[:,0]
    gns1['yp'] = gns1_xyt[:,1]
    
    
    # gns1 = alg_rel(gns1, gns2,'xp', 'yp', align_by,use_grid,max_deg = max_deg, d_m = d_m,f_mode = f_mode,grid_s= grid_s )
    gns1 = alg_loop(gns1, gns2, 'xp', 'yp', align_by, max_deg, d_m.to(u.arcsec).value, max_loop,sig_cl_H = sig_cl_H_aligment, 
                     grid_s = grid_s, grid_Hmin = grid_Hmin, grid_Hmax = grid_Hmax ,isolation_radius = isolation_radius,  f_mode = f_mode, mag_lim_alig=mag_lim_alig)

# sys.exit(202) 
# %%
l1_xy = np.array([gns1['xp'],gns1['yp']]).T
l2_xy = np.array([gns2['xp'],gns2['yp']]).T
l_12 = compare_lists(l1_xy,l2_xy,d_m_pm)


print(30*'*'+'\nComon stars to be use for pm calculation :%s\n'%(len(l_12))+30*'*')
gns1_mi = gns1[l_12['ind_1']]
gns2_mi = gns2[l_12['ind_2']]



dx = (gns2_mi['xp'].value- gns1_mi['xp'].value)*1000
dy = (gns2_mi['yp'].value- gns1_mi['yp'].value)*1000

pm_x = (dx*u.mas)/dt.to(u.year)
pm_y = (dy*u.mas)/dt.to(u.year)

dpm_x = np.sqrt((gns2_mi['sl'].to(u.mas))**2 + (gns1_mi['sl'].to(u.mas))**2)/dt.to(u.year)
dpm_y = np.sqrt((gns2_mi['sb'].to(u.mas))**2 + (gns1_mi['sb'].to(u.mas))**2)/dt.to(u.year)

       

# pm_l = (dl*mean_b)/dt.to(u.year)
# pm_b = (db)/dt.to(u.year)

sig_cl_pm = 3

m_pm, lims = sig_f(pm_x, pm_x, sig_cl_pm)


pm_xm = pm_x[m_pm]
pm_ym = pm_y[m_pm]
dpm_xm = dpm_x[m_pm]
dpm_ym = dpm_y[m_pm]

gns1_mi = gns1_mi[m_pm]
gns2_mi = gns2_mi[m_pm]

gns1_mi['pm_x']  = pm_xm
gns1_mi['pm_y']  = pm_ym

gns2_mi['pm_x']  = pm_xm
gns2_mi['pm_y']  = pm_ym

gns1_mi['dpm_x']  = dpm_xm
gns1_mi['dpm_y']  = dpm_ym

gns2_mi['dpm_x']  = dpm_xm
gns2_mi['dpm_y']  = dpm_ym


gns1_mi.meta['max_loop'] = max_loop
gns1_mi.meta['gns_mags'] = gns_mags
gns1_mi.meta['max_sig'] = max_sig
gns1_mi.meta['e_pm_gns'] = e_pm_gns
gns1_mi.meta['grid_s'] = grid_s
gns1_mi.meta['max_sep'] = max_sep
gns2_mi.meta['sig_cl_pm'] = sig_cl_pm
gns1_mi.meta['sig_cl_H'] = sig_cl_H
gns1_mi.meta['sig_cl_H_alignment'] = sig_cl_H_aligment
gns1_mi.meta['max_deg'] = max_deg
gns1_mi.meta['d_m'] = d_m
gns1_mi.meta['d_m_pm'] = d_m_pm
gns1_mi.meta['destination'] = destination
gns1_mi.meta['align_by'] = align_by
gns1_mi.meta['f_mode'] = f_mode

# gns1_mi.write(pruebas1 + f'gns1_pmSuper_F1_{field_one}_F2_{field_two}.ecsv', format = 'ascii.ecsv', overwrite = True)

gns2_mi.meta['max_loop'] = max_loop
gns2_mi.meta['gns_mags'] = gns_mags
gns2_mi.meta['max_sig'] = max_sig
gns2_mi.meta['e_pm_gns'] = e_pm_gns
gns2_mi.meta['grid_s'] = grid_s
gns2_mi.meta['max_sep'] = max_sep
gns2_mi.meta['sig_cl_pm'] = sig_cl_pm
gns2_mi.meta['sig_cl_H'] = sig_cl_H
gns2_mi.meta['sig_cl_H_alignment'] = sig_cl_H_aligment
gns2_mi.meta['max_deg'] = max_deg
gns2_mi.meta['d_m'] = d_m
gns2_mi.meta['d_m_pm'] = d_m_pm
gns2_mi.meta['destination'] = destination
gns2_mi.meta['align_by'] = align_by
gns2_mi.meta['f_mode'] = f_mode

# gns2_mi.write(pruebas2 + f'gns2_pmSuper_F1_{field_one}_F2_{field_two}.ecsv', format = 'ascii.ecsv', overwrite = True)



# %%
# e_pm_gns = 2##
gns1_m = filter_gns_data(gns1_mi, max_e_pm = e_pm_gns)
gns2_m = filter_gns_data(gns2_mi, max_e_pm = e_pm_gns)


# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable
dh = gns2_m['H'] - gns1_m['H']



fig, (ax,ax2,ax3) = plt.subplots(1,3, figsize = (21,7))
ax.hist(dh, bins = 'auto', label = '$\overline{\Delta H}$ =%.2f\n$\sigma$ = %.2f'%(np.mean(dh) ,np.std(dh) ))
ax2.axhline(l_lim, color = 'red', ls = 'dashed')
ax2.axhline(h_lim, color = 'red', ls = 'dashed')
ax.legend()
ax.set_title(f'gns1 -> f{field_one}c{chipA}. GNS2 -> F{field_two}C{chipB}')
ax.set_xlabel('$\Delta$[H]')
# ax2.scatter(gns1['H1'][l_12['ind_1']], dh)
his = ax2.hist2d(gns1_m['H'], dh, bins = 100, norm = LogNorm())
ax2.set_ylabel('$\Delta$[H]')
ax2.set_xlabel('GNS1[H]')
ax2.grid()
his3 = ax3.hist2d(gns2_m['H'], dh, bins = 100, norm = LogNorm())
ax3.set_xlabel('GNS2[H]')

ax3.grid()
ax.set_xlim(-2,2)
ax2.set_xlim(11,22)
ax3.set_xlim(11,22)
ax2.set_ylim(-2,2)
ax3.set_ylim(-2,2)
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(his3[3], cax=cax)

fig.tight_layout()

mask_H, l_lim,h_lim = sigma_clip(dh, sigma=sig_cl_H, masked = True, return_bounds= True, maxiters= 50)
gns1_m = gns1_m[np.logical_not(mask_H.mask)]
gns2_m = gns2_m[np.logical_not(mask_H.mask)]

# %%
# fig, (ax,ax2) = plt.subplots(1,2, figsize = (16,8))
# his = ax.hist2d(gns1_mi['H'],(gns1_mi['pm_x'])/2 , bins = 100,norm = LogNorm())
# fig.colorbar(his[3], ax =ax, label = '#stars/bin',fraction = 0.049)
# his2 = ax2.hist2d(gns1_mi['H'],(gns1_mi['pm_y']) , bins = 100,norm = LogNorm())
# fig.colorbar(his2[3], ax =ax2, label = '#stars/bin',fraction = 0.049)
# # ax.set_title('GNS1')
# ax.set_ylabel('$\mu_x$ [mas/yr]')
# ax2.set_ylabel('$\mu_y$ [mas/yr]')
# ax.set_xlabel('[H]')
# ax2.set_xlabel('[H]')
# # ax.axis('scaled')
# ax.set_xlim(np.min(gns1_mi['H'])- 0.01,np.max(gns1_mi['H'])+ 0.01)

# fig.tight_layout()

# # pm_lim = (abs(gns1_m['pm_x'])<4)
# # gns1_m = gns1_m[pm_lim]
# %%
bins = 'auto'
fig, (ax,ax2) = plt.subplots(1,2)
ax.set_title(f'Ref. = GNS{destination}. Degree = {max_deg-1}', fontsize= 15)
ax2.set_title(f'E_pm < {e_pm_gns}mas/yr. H = [{gns_mags[0], gns_mags[1]}]', fontsize= 15)

ax.hist(gns1_m['pm_x'], bins = bins, histtype = 'step', label = '$\overline{\mu}_{x}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(gns1_m['pm_x'].value),np.std(gns1_m['pm_x'].value)),)
ax2.hist(gns1_m['pm_y'], bins = bins, histtype = 'step',label = '$\overline{\mu}_{y}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(gns1_m['pm_y'].value),np.std(gns1_m['pm_y'].value)))
ax.set_xlabel('$\Delta \mu_{x}$ [mas/yr]')
ax2.set_xlabel('$\Delta\mu_{y}$ [mas/yr]')
# ax.axvline(lims[0].value , ls = 'dashed', color = 'r', label = f'{sig_dis}$\sigma$')
# ax.axvline(lims[1].value , ls = 'dashed', color = 'r')
# ax2.axvline(lims[2].value , ls = 'dashed', color = 'r')
# ax2.axvline(lims[3].value , ls = 'dashed', color = 'r')
ax.set_xlim(-20,20)
ax2.set_xlim(-20,20)
ax.invert_xaxis()

# ax.invert_xaxis()
ax.legend( fontsize = 15)
ax2.legend(fontsize = 15)


# sys.exit(712)
# %% Gaia Comparation#!!!
max_sep_ga = 50*u.mas# separation for comparison with gaia
e_pm = 0.3#!!! Maximun error in pm for Gaia stars
gaia_mags = [12,19]#!!! Gaia mag limtis for comparison with GNS
# %
# Before comparing witg Gaia we mask the best pms



extra_mag_cut = [0,100]
gns1_mpm = filter_gns_data(gns1_m, max_e_pm = e_pm_gns, min_mag = extra_mag_cut[1], max_mag = extra_mag_cut[0])
gns2_mpm = filter_gns_data(gns2_m, max_e_pm = e_pm_gns,  min_mag = extra_mag_cut[1], max_mag = extra_mag_cut[0])



radius = abs(np.min(gns1['l'])-np.max(gns1['l']))*0.5*u.degree
# radius = abs(np.min(gns1['l'])-np.max(gns1['l']))*2.6*u.degree
center_g = SkyCoord(l = np.mean(gns1['l']), b = np.mean(gns1['b']), unit = 'degree', frame = 'galactic')

try:
    
    gaia = Table.read(pruebas1  + 'gaia_f1%s_f2%s_r%.0f.ecsv'%(field_one,field_two,radius.to(u.arcsec).value))
    print('Gaia from table')
except:
    print('Gaia from web')
    center = SkyCoord(l = np.mean(gns1['l']), b = np.mean(gns1['b']), unit = 'degree', frame = 'galactic').icrs

    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source" # Select early Data Release 3
    Gaia.ROW_LIMIT = -1  # it not especifty, Default rows are limited to 50. 
    j = Gaia.cone_search_async(center, radius = abs(radius))
    gaia = j.get_results()
    gaia.write(pruebas1  + 'gaia_f1%s_f2%s_r%.0f.ecsv'%(field_one,field_two,radius.to(u.arcsec).value))



gfig, (gax, gax2) = plt.subplots(1,2)
gax.scatter(gaia['phot_g_mean_mag'],gaia['pmra_error'],color = 'k')
gax.scatter(gaia['phot_g_mean_mag'],gaia['pmdec_error'], color = 'grey')
# gax2.scatter(gaia['phot_g_mean_mag'],gaia['ra_error'],color = 'k')
# gax2.scatter(gaia['phot_g_mean_mag'],gaia['dec_error'],color = 'grey')
gax.set_ylabel('$\sigma \mu$')
gax2.set_ylabel('$\sigma \mu (gns)$')
gax.set_xlabel('[G]')
gax2.set_xlabel('[H]')
gfig.tight_layout()



# Step 1: Gaia proper motion as SkyCoord
c_gaia = SkyCoord(ra=gaia['ra'], dec=gaia['dec'],
                  pm_ra_cosdec=gaia['pmra'],
                  pm_dec=gaia['pmdec'],
                  obstime="J2016")

# c_gaia = SkyCoord(ra=gaia['ra'], dec=gaia['dec'],
#                   pm_ra_cosdec=gaia['pmra_error'],
#                   pm_dec=gaia['pmdec_error'],
#                   obstime="J2016").galactic


# Step 2: Convert to Cartesian offset using the tangent-plane projection
# Use a fixed reference point (e.g., center of your field)
# ref = SkyCoord(ra0*u.deg, dec0*u.deg)
# center = SkyCoord(ra=265.75405671 * u.deg, dec=-28.66946795 * u.deg, frame='icrs')
offset_frame = center.skyoffset_frame()

c_proj = c_gaia.transform_to(offset_frame)

# Step 3: Extract Gaia PM in tangent plane (same as your XY frame, in mas/yr)
pm_x_gaia = c_proj.pm_lon_coslat  # mas/yr
pm_y_gaia = c_proj.pm_lat        # mas/yr

gaia['pm_x'] = pm_x_gaia
gaia['pm_y'] = pm_y_gaia

# gaia['pm_x'] = gaia['pmra']
# gaia['pm_y'] = gaia['pmdec']

mu_eq = np.hypot(gaia['pmra'], gaia['pmdec'])
mu_pr = np.hypot(gaia['pm_x'], gaia['pm_y'])

resid_mu = mu_eq - mu_pr
print('Differnces between Gaia pm and Gaia pm proyected')
print('Comoponete are differente, but modules of the vetors invariant under transformations')
print("σ(Δμ):", np.std(resid_mu))
print("mean(Δμ):", np.mean(resid_mu))
print("σ(Gaia_μra - Gaia_μx):", np.std(gaia['pmra'] - gaia['pm_x']))
print("σ(Gaia_μdec  - Gaia_μy):",np.std(gaia['pmdec'] - gaia['pm_y']))


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





# g_gpm = SkyCoord(ra = gaia['ra'], dec = gaia['dec'], pm_ra_cosdec = gaia['pmra'].value*u.mas/u.yr, pm_dec = ['pmdec'].value*u.mas/u.yr, obstime = 'J2016', equinox = 'J2000', frame = 'fk5')
ga_c = SkyCoord(ra = gaia['ra'], dec = gaia['dec'], pm_ra_cosdec = gaia['pmra'],
                 pm_dec = gaia['pmdec'], obstime = 'J2016', 
                 equinox = 'J2000', frame = 'icrs').galactic


# l_off,b_off = center_g.spherical_offsets_to(ga_gpm.frame)
# l_offt = l_off + (ga_gpm.pm_l_cosb)*dtg.to(u.yr)
# b_offt = b_off + (ga_gpm.pm_b)*dtg.to(u.yr)
# ga_gtc = center_g.spherical_offsets_by(l_offt, b_offt)


gaia['l'] = ga_c.l + (ga_c.pm_l_cosb)*dtg.to(u.yr)
gaia['b'] = ga_c.b + (ga_c.pm_b)*dtg.to(u.yr)


# %
# gaia_c = SkyCoord(ra = gaia['ra'], dec = gaia['dec'], 
#                   pm_ra_cosdec = gaia['pmra'], pm_dec = gaia['pmdec'],
#                   frame = 'icrs', obstime = 'J2016.0').galactic



gaia['pm_l'] = ga_c.pm_l_cosb
gaia['pm_b'] = ga_c.pm_b

print("σ(Gaia_μl - Gaia_μx):", np.std(gaia['pm_l'] - gaia['pm_x']))
print("σ(Gaia_μb  - Gaia_μy):",np.std(gaia['pm_b'] - gaia['pm_y']))


# %

ga_l = ga_c.l.wrap_at('360.02d')
ga_b = ga_c.b.wrap_at('180d')



gns2_gal = SkyCoord(l = gns2_mpm['l'], b = gns2_mpm['b'], 
                    unit = 'degree', frame = 'galactic')
gns1_gal = SkyCoord(l = gns1_mpm['l'], b = gns1_mpm['b'], 
                    unit = 'degree', frame = 'galactic')

gaia_c = SkyCoord(l = gaia['l'], b = gaia['b'], frame = 'galactic')


# %


if destination == 1:
    
    idx,d2d,d3d = gns1_gal.match_to_catalog_sky(gaia_c)
    sep_constraint = d2d < max_sep_ga
    gns_ga = gns1_mpm[sep_constraint]
    ga_gns = gaia[idx[sep_constraint]]# ,nthneighbor=1 is for 1-to-1 match

    
    # idx1, idx2, sep2d, _ = search_around_sky(gns1_gal, gaia_c, max_sep_ga)

    # count1 = Counter(idx1)
    # count2 = Counter(idx2)

    # # Step 3: Create mask for one-to-one matches only
    # mask_unique = np.array([
    #     count1[i1] == 1 and count2[i2] == 1
    #     for i1, i2 in zip(idx1, idx2)
    # ])

    # # Step 4: Apply the mask
    # idx1_clean = idx1[mask_unique]
    # idx2_clean = idx2[mask_unique]

    # gns_ga = gns1_mpm[idx1_clean]
    # ga_gns = gaia[idx2_clean]
    
    
elif destination ==2:
    # idx,d2d,d3d = gns2_gal.match_to_catalog_sky(gaia_c)
    # sep_constraint = d2d < max_sep_ga
    # gns_ga = gns2_mpm[sep_constraint]
    # ga_gns = gaia[idx[sep_constraint]]# ,nthneighbor=1 is for 1-to-1 match

    idx1, idx2, sep2d, _ = search_around_sky(gns2_gal, gaia_c, max_sep_ga)

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

    gns_ga = gns2_mpm[idx1_clean]
    ga_gns = gaia[idx2_clean]
    


l2 = gns2_gal.l.wrap_at('360d')
l1 = gns1_gal.l.wrap_at('360d')

# %


# %


# =============================================================================
# ⚠️⚠️⚠️⚠️WARNING!!!⚠️⚠️⚠️
# The pm errors, in both Gaia and GNS have to been proyected to the tangetial plane as well!!!
# =============================================================================
# %



# d_pmx_ga = gns_ga['pm_x']- ga_gns['pm_x']
# d_pmy_ga = gns_ga['pm_y']- ga_gns['pm_y']
d_pmx_ga = gns_ga['pm_x']- ga_gns['pm_l']
d_pmy_ga = gns_ga['pm_y']- ga_gns['pm_b']
# 

e_dpmx = np.sqrt(gns_ga['dpm_x']**2  + ga_gns['pmra_error']**2)
e_dpmy = np.sqrt(gns_ga['dpm_y']**2  + ga_gns['pmdec_error']**2)

sig_ga = 3
m_pm, lxy = sig_f(d_pmx_ga, d_pmy_ga, sig_ga)

gns_ga_m = gns_ga[m_pm]
ga_gns_m = ga_gns[m_pm]

d_pmx_ga_m = d_pmx_ga[m_pm]
d_pmy_ga_m = d_pmy_ga[m_pm]

e_dpmx_m = e_dpmx[m_pm]
e_dpmy_m = e_dpmy[m_pm]

gax.scatter(ga_gns['phot_g_mean_mag'],ga_gns['pmra_error'])
gax.scatter(ga_gns['phot_g_mean_mag'],ga_gns['pmdec_error'])
gax2.scatter(gns_ga['H'],gns_ga['dpm_x'])
gax2.scatter(gns_ga['H'],gns_ga['dpm_y'])
gax2.set_ylim(-0.1,2.6)

# gax2.scatter(ga_gns['phot_g_mean_mag'],ga_gns['ra_error'])
# gax2.scatter(ga_gns['phot_g_mean_mag'],ga_gns['dec_error'])

gax.scatter(ga_gns[np.logical_not(m_pm)]['phot_g_mean_mag'],ga_gns[np.logical_not(m_pm)]['pmra_error'], marker ='x', color = 'red',s = 1, lw = 5)
gax.scatter(ga_gns[np.logical_not(m_pm)]['phot_g_mean_mag'],ga_gns[np.logical_not(m_pm)]['pmdec_error'], marker ='x', color = 'red',s = 1, lw = 5)
# gax2.scatter(ga_gns[np.logical_not(m_pm)]['phot_g_mean_mag'],ga_gns[np.logical_not(m_pm)]['ra_error'], marker ='x', color = 'red',s = 1, lw = 5)
# gax2.scatter(ga_gns[np.logical_not(m_pm)]['phot_g_mean_mag'],ga_gns[np.logical_not(m_pm)]['dec_error'], marker ='x', color = 'red',s = 1, lw = 5)


# =============================================================================
# CLipping 3sigma residuals of position with Gaia
# =============================================================================
dl = (gns_ga_m['l'] - ga_gns_m['l']).to(u.mas)
db = (gns_ga_m['b'] - ga_gns_m['b']).to(u.mas)

sig_ga = 3
m_lb, l_ = sig_f(dl, db, sig_ga)


fig, (ax,ax2)= plt.subplots(1,2)
ax.set_title(f'Ref. ep. GNS{destination}')
ax2.set_title(f'Matching {len(dl)}')
ax.hist(dl, color = '#ff7f0e',histtype = 'step', label = '$\overline{\Delta l}$ = %.2f\n$\sigma_{l} = %.2f$'%(np.mean(dl.value),np.std(dl.value)))
ax2.hist(db, color = '#ff7f0e', histtype = 'step', label = '$\overline{\Delta b}$ = %.2f\n$\sigma_{b} = %.2f$'%(np.mean(db.value),np.std(db.value)))

ax.axvline(l_[0].value, ls = 'dashed', color = 'r', label = f'{sig_ga}$\sigma$')
ax.axvline(l_[1].value, ls = 'dashed', color = 'r')
ax2.axvline(l_[2].value, ls = 'dashed', color = 'r')
ax2.axvline(l_[3].value, ls = 'dashed', color = 'r')


ax.legend()
ax2.legend()
ax.set_xlabel('$\Delta l$[mas}')
ax2.set_xlabel('$\Delta b$[mas}')

gns_ga_m = gns_ga_m[m_lb]
ga_gns_m = ga_gns_m[m_lb]

d_pmx_ga_m = d_pmx_ga_m[m_lb]
d_pmy_ga_m = d_pmy_ga_m[m_lb]

e_dpmx_m = e_dpmx_m[m_lb]
e_dpmy_m = e_dpmy_m[m_lb]


gns2_mpm['pm_x'] -= np.mean(d_pmx_ga_m.value)
gns2_mpm['pm_y'] -= np.mean(d_pmy_ga_m.value)
 
# %


# %
fig, ax_g = plt.subplots(1,1,figsize =(10,10))
ax_g.set_title(f'Macthing distance = {max_sep_ga}. GNS1 {field_one} GNS2 {field_two}')
ax_g.scatter(l1[::1], gns1_mpm['b'][::1],s=1, marker = 'x',label = 'GNS_1 Fied %s, chip %s'%(field_one,chipA))

# ax.scatter(l2[::10],  gns2_m['b'][::10], label = 'GNS_2 Fied %s, chip %s'%(field_two,chip_two))
clb = ax_g.scatter(ga_gns['l'],ga_gns['b'], c = np.sqrt(ga_gns['pmra_error']**2 + ga_gns['pmdec_error']**2 ),s =200,label = f'Gaia comp pm = {len(ga_gns)}', cmap = 'Spectral_r', edgecolor = 'k')
# clb = ax_g.scatter(ga_gns['l'],ga_gns['b'], c = np.mean(d_pmx_ga) -d_pmx_ga,s =200,label = f'Gaia comp pm = {len(ga_gns)}', cmap = 'Spectral_r', edgecolor = 'k')
ax_g.invert_xaxis()
# ax_g.legend()
ax_g.set_xlabel('l[ deg]', fontsize = 10)
ax_g.set_ylabel('b [deg]', fontsize = 10)
# ax.axis('scaled')
ax_g.set_ylim(min(gaia['b']),max(gaia['b']))
fig.colorbar(clb, ax = ax_g,fraction = 0.0275, aspect = 30, label = 'Gaia pm uncer [mas/yr] \n$\sqrt{\sigma_{pmra}^2 + \sigma_{pmde}^2  }$')
ax_g.scatter(ga_gns[np.logical_not(m_pm)]['l'], ga_gns[np.logical_not(m_pm)]['b'], marker = 'x', color = 'orange', s = 200, lw =5, label = f'{sig_ga}$\sigma$ Clipped Gaia')
ax_g.scatter(gaia['l'], gaia['b'], marker = 'x', color = 'k', alpha = 0.5)
lgnd = ax_g.legend()

for handle in lgnd.legend_handles:
    handle.set_sizes([200])
# %
fig, (ax,ax2) = plt.subplots(1,2)
ax.set_title(f'Proyected Gaia pm. Degre {max_deg-1}', fontsize= 15)
ax2.set_title(f'Ref. epoch GNS{destination}. Matches {len(d_pmx_ga_m)}', fontsize = 15)
ax.hist(d_pmx_ga, bins = bins, color = 'k', alpha = 0.2)
ax2.hist(d_pmy_ga, bins = bins,color = 'k', alpha = 0.2)
ax.hist(d_pmx_ga_m, bins = bins,lw =2, histtype = 'step', label = '$\overline{\mu}_{x}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(d_pmx_ga_m.value),np.std(d_pmx_ga_m.value)))
ax2.hist(d_pmy_ga_m, bins = bins, histtype = 'step',label = '$\overline{\mu}_{y}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(d_pmy_ga_m.value),np.std(d_pmy_ga_m.value)))
ax.set_xlabel('$\Delta \mu_{l}$ [mas/yr]')
ax2.set_xlabel('$\Delta\mu_{b}$ [mas/yr]')
# ax.set_xlabel('$\Delta \mu_{x}$ [mas/yr]')
# ax2.set_xlabel('$\Delta\mu_{y}$ [mas/yr]')
ax.axvline(lxy[0], ls = 'dashed', color = 'r', label = f'{sig_ga}$\sigma$')
ax.axvline(lxy[1], ls = 'dashed', color = 'r')
ax2.axvline(lxy[2], ls = 'dashed', color = 'r')
ax2.axvline(lxy[3], ls = 'dashed', color = 'r')
ax.legend()
ax2.legend()
# %%
if look_for_cluster == 'yes':
    
   
    # modes = ['pm_xy_color']
    modes = ['pm_xy']
    knn = 15
    gen_sim = 'kernnel'
    sim_lim ='minimun'
    # sim_lim ='mean'
    if destination == 2:
        clus_dic = gns_cluster_finder.finder(gns2_mpm['pm_x'], gns2_mpm['pm_y'],
                                     gns2_mpm['xp'], gns2_mpm['yp'], 
                                     gns2_mpm['l'].value, gns2_mpm['b'].value,
                                     modes[0],
                                     gns2_mpm['H'],gns2_mpm['H'],
                                     knn,gen_sim,sim_lim, save_reg = pruebas2)
    elif destination == 1:
        clus_dic = gns_cluster_finder.finder(gns1_mpm['pm_x'], gns1_mpm['pm_y'],
                                     gns1_mpm['xp'], gns1_mpm['yp'], 
                                     gns1_mpm['l'].value, gns1_mpm['b'].value,
                                     modes[0],
                                     gns1_mpm['H'],gns1_mpm['H'],
                                     knn,gen_sim,sim_lim, save_reg_folder = pruebas1)

sys.exit(1098)
