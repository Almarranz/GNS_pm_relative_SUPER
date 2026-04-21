# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Fri Nov  8 12:42:20 2024

@author: amartinez
"""
import sys
sys.path.append("/Users/amartinez/Desktop/pythons_imports/")

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
import cluster_finder
from kneed import KneeLocator
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from sys import exit as stop
from astropy.coordinates import Longitude
from ds9_region import region
from pyplots import l_function
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


# band1 = 'Ks'
band1 = 'H'

rebfacI = 2
rebfacII = 1

# field_one = 'D12'
# field_one = 'D13'
# field_two = 1



# field_one = 2
# field_two = 22

# field_one = 'B1'
# field_two = 20

# field_one = 'D19'
# field_two = 16

field_one = 16
field_two = 7

# field_one = 24  
# field_two = 9

# field_one = 2
# field_two = 9

# field_one = 2
# field_two = 9

# field_one = 2
# field_two = 22

# field_one = 2
# field_two = 9

chip_one = 0
chip_two = 0

GNS_1='/Users/amartinez/Desktop/PhD/HAWK/GNS_1/lists/%s/chip%s/'%(field_one, chip_one)
GNS_2='/Users/amartinez/Desktop/PhD/HAWK/GNS_2/lists/%s/chip%s/'%(field_two, chip_two)

pruebas1 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1relative_SUPER/pruebas/'
pruebas2 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2relative_SUPER/pruebas/'
bs1 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1relative_SUPER/bootstrapping/'
bs2 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2relative_SUPER/bootstrapping/'


dates1 = Table.read('/Volumes/teabag-data/alvaro/GNS_HB_red/GNS1/date_of_field.csv', format = 'ascii')
dates2 = Table.read('/Volumes/teabag-data/alvaro/GNS_HB_red/GNS2/date_of_field.csv', format = 'ascii')
# dates1 = Table.read('/Users/amartinez/Desktop/Projects/GNS_gd/superlists/GNS1/date_of_field.csv', format = 'ascii')
# dates2 = Table.read('/Users/amartinez/Desktop/Projects/GNS_gd/superlists/GNS2/date_of_field.csv', format = 'ascii')

# t1 =  Time(dates1[dates1['field'] == str(field_one)]['H'].value,scale='utc')
# t2 =  Time(dates2[dates2['field'] == field_two]['H'].value,scale='utc')

# print(f'GNS1 obstime: {t1}')
# print(f'GNS2 obstime: {t2}')
# print(t2)

# stop(144)

if field_one == 7 or field_one == 12 or field_one == 10 or field_one == 16  or field_one == 19 or field_one == 2 or field_one == 24  :
    t1 = Time(['2015-06-07T00:00:00'],scale='utc')
elif field_one == 'B6':
    t1 = Time(['2016-06-13T00:00:00'],scale='utc')
elif field_one ==  'B1' or field_one == 1:
    t1 = Time(['2016-05-20T00:00:00'],scale='utc')
elif field_one ==  'D12':
    t1 = Time(['2017-06-03T00:00:00'],scale='utc')
elif field_one ==  'D13':
    t1 = Time(['2017-06-24T00:00:00'],scale='utc')
elif field_one ==  'D19':
    t1 = Time(['2018-05-21T00:00:00'],scale='utc')
else:
    print(f'NO time detected for this field_one = {field_one}')
    sys.exit()
if field_two == 7 or field_two == 5:
    t2 = Time(['2022-05-27T00:00:00'],scale='utc')
elif field_two == 4:
    t2 = Time(['2022-04-05T00:00:00'],scale='utc')
elif field_two == 20:
    t2 = Time(['2022-07-25T00:00:00'],scale='utc')
elif field_two == 1:
    t2 = Time(['2021-09-17T00:00:00'],scale='utc')
elif field_two == 16 or field_two == 22:
    t2 = Time(['2022-08-14T00:00:00'],scale='utc')
elif field_two == 9:
    t2 = Time(['2021-08-14T00:00:00'],scale='utc')
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
gns_mags = [10, 22]#!!! GNS mag limtis
# gns_mags = [None, None]#!!! GNS mag limtis
max_sig = 0.05# arcsec Max uncertainty position (l,b)
# max_sig = 1000# arcsec Max uncertainty position (l,b)
# max_sig = 0.02# arcsec Max uncertainty position (l,b)
bin_width = 0.1
perc_H = 100
perc_lb = 100
# =============================================================================
# GRID PARAMS
# =============================================================================
grid_s = None
# grid_s = 2# si1ce of the grid cell in arcsec 
grid_Hmin = 12
grid_Hmax = 18
isolation_radius = 0.7#arcsec isolation of the grid stars 

# =============================================================================
# ALIGNMENT PARAMS
# =============================================================================
max_loop = 5
mag_lim_alig = None# H Limits of the algnment stars
mag_lim_alig = [12, 18]# H Limits of the algnment stars
max_sep = 50*u.mas#, firts match gns1 to gns2 for astroaling
# init_alig = 'polynomial1'# Options are: 'astroalign', 'smilirarity' or 'polynomial<degre of the alignment>'
init_alig = 'astroalign'# Options are: 'astroalign', 'similirarity' or 'polynomial<degre of the alignment>'
# init_alig = 'similarity'# Options are: 'astroalign', 'similirarity' or 'polynomial<degre of the alignment>'
sig_cl = 3#!!!
max_deg = 3# Maximun degree of the alignment minus one()
centered_in = 1
# centered_in = 2
d_m = 50*u.mas#!!! max  distance  for the fine alignment betwenn GNS1 and 2
# destination = 2#!!!1 = GNS2 is reference, 2 = GNS1 in reference
destination = 1 #!!! GNS1 is reference
align_by = 'Polywarp'#!!!
# align_by = '2DPoly'#!!!
f_mode = 'W' # f_mode only useful for 2Dpoly
# f_mode = 'WnC'
# f_mode = 'NW'
# f_mode = 'NWnC'

# add_aligm_error = 'yes'
add_aligm_error = 'no'
# =============================================================================
# PMs PARAMS
# =============================================================================
d_m_pm = 0.150#!!! in arcs, max distance for the proper motions
# e_pm_gns = .9 #!!!error cut in proper motio
e_pm_gns = None#!!!error cut in proper motio
if band1 == 'H':
    sig_cl_H = 3 # Eliminates bad matches before the proper motions computations
else:
    sig_cl_H = 1e3 # Eliminates bad matches before the proper motions computations
sig_cl_pm = 3 # Clipping outlayer from the pm distributions. 
# sig_cl_pm = None # Clipping outlayer from the pm distributions. 
# =============================================================================
# GAIA PARAMS
# =============================================================================

max_sep_ga = 150*u.mas# separation for comparison with gaia
e_pm_gaia = 1#!!! Maximun error in pm for Gaia stars
e_pos_gaia = 1
gaia_mags = [12,18.5]#!!! Gaia mag limtis for comparison with GNS
sig_ga = 3   # CLipping 3sigma residuals of position with Gaia

# =============================================================================
# Cluster
# =============================================================================
# look_for_cluster = 'no'
look_for_cluster = 'yes'
# 
def sig_f(x, y,s):
    mx, lx, hx = sigma_clip(x , sigma = s, masked = True, return_bounds= True)
    my, ly, hy = sigma_clip(y , sigma = s, masked = True, return_bounds= True)
    m_xy = np.logical_and(np.logical_not(mx.mask),np.logical_not(my.mask))
    
    return m_xy, [lx,hx,ly,hy]
# %% 

# 
if chip_one == 0:
    # gns1 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/pruebas/F{field_one}/{field_one}_H_chips_opti.ecsv',  format = 'ascii.ecsv')
    # gns1 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/superlists/GNS1/F{field_one}/{field_one}_H_chips_opti.ecsv',  format = 'ascii.ecsv')
    # gns1 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/superlists/GNS1/F{field_one}/{field_one}_H_chips_opti_rebfac{rebfacI}.ecsv',  format = 'ascii.ecsv')
    # gns1 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/superlists/GNS1/F{field_one}/{field_one}_H_chips_opti_noDup_rebfac{rebfacI}.ecsv',  format = 'ascii.ecsv')
    # gns1 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/superlists/GNS1/F{field_one}/{field_one}_{band1}_chips_opti_noDup_rebfac{rebfacI}.ecsv',  format = 'ascii.ecsv')
    gns1 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/superlists/GNS1/F{field_one}/{field_one}_{band1}_chips_opti_rebfac{rebfacI}.ecsv',  format = 'ascii.ecsv')
    # gns1 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/superlists/GNS1/F{field_one}/{field_one}_H_chips_opti_OLD.ecsv',  format = 'ascii.ecsv')
    # gns1 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/superlists/GNS1/F{field_one}/{field_one}_H_chips_opti_rebfac{rebfacI}_VVV.ecsv',  format = 'ascii.ecsv')
    
    # gns1 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/superlists/GNS1/F{field_one}/{field_one}_H_chips_opti_noDup_rebfac{rebfacI}.ecsv',  format = 'ascii.ecsv')

    
    # gns1 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/pruebas/F{field_one}_old/{field_one}_H_chips_opti.ecsv',  format = 'ascii.ecsv')
    # gns1 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/pruebas/F{field_one}/{field_one}_H_chips_opti_rebf1.ecsv',  format = 'ascii.ecsv')
    
    # gns1 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/pruebas/F{field_one}/{field_one}_and_D13_comb.ecsv',  format = 'ascii.ecsv')
    
# =============================================================================
#     gns1 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/superlists/GNS1/tiles/tile_gns1_Ori_f8.txt',  format = 'ascii')
#     gns1['l'] = gns1['l']*u.degree
#     gns1['b'] = gns1['b']*u.degree
#     gns1['sl'] = gns1['sl']*u.arcsec
#     gns1['sb'] = gns1['sb']*u.arcsec
#     gns1['H'] = gns1['m'] 
# =============================================================================
    
    gns1['ID'] = np.arange(len(gns1))
else:
    gns1 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/pruebas/F{field_one}/{field_one}_H_chips_opti.ecsv',  format = 'ascii.ecsv')
    gns1['ID'] = np.arange(len(gns1))

# gns1 = filter_gns_by_percentile(gns1, mag_col='H', err_col='dH', sl_col='sl', sb_col='sb', bin_width=bin_width, percentile_H=perc_H, percentile_lb=perc_lb, mag_lim = None, pos_lim = None)



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
# # axb.scatter(ga_gns['l'], ga_gns['b'], marker = '*', edgecolor = 'k', label = 'Gaia Stars',s = 200)
# axb.legend()
# meta = {'Script': '/Users/amartinez/Desktop/PhD/HAWK/GNS_pm_scripts/GNS_pm_relative_SUPER/SUPER_alignment.py'}
# # plt.savefig('/Users/amartinez/Desktop/PhD/My_papers/SgrB1_cluster/images/dpos_lb_H.png', bbox_inches='tight', pad_inches=0, dpi = 300, edgecolor = 'white', transparent = True, metadata = meta)


# meta = {'Script': '/Users/amartinez/Desktop/PhD/HAWK/GNS_pm_scripts/GNS_pm_relative_SUPER/SUPER_alignment.py'}
# # plt.savefig('/Users/amartinez/Desktop/PhD/My_papers/SgrB1_cluster/images/dpos_H_gns1.png', bbox_inches='tight', pad_inches=0, dpi = 300, edgecolor = 'white', transparent = True, metadata = meta)


# %%



gns2 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/pruebas/F{field_two}/{field_two}_H_chips_opti.ecsv', format = 'ascii.ecsv')
# gns2 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/superlists/GNS2/F{field_two}/some_lists/{field_two}_H_chips_opti_rebfac2.ecsv' )
# gns2 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/superlists/GNS2/F{field_two}/{field_two}_H_chips_opti.ecsv', format = 'ascii.ecsv')
# gns2 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/superlists/GNS2/F{field_two}/{field_two}_H.ecsv', format = 'ascii.ecsv')
# gns2 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/superlists/GNS2/F{field_two}/{field_two}_H_chips_opti_RS.ecsv', format = 'ascii.ecsv')
# gns2 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/superlists/GNS2/F{field_two}/{field_two}_H_chips_opti_rebfac{rebfacII}.ecsv', format = 'ascii.ecsv')
# gns2 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/superlists/GNS2/F{field_two}/{field_two}_H_chips_opti_rebfac{rebfacII}_VVV.ecsv', format = 'ascii.ecsv')
# gns2 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/superlists/GNS2/F{field_two}/{field_two}_H_chips_opti_rebfac{rebfacII}_old.ecsv', format = 'ascii.ecsv')
# gns2 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/superlists/GNS2/F{field_two}/{field_two}_H_chips_opti_noDup_rebfac{rebfacII}.ecsv', format = 'ascii.ecsv')

# =============================================================================
# gns2 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/superlists/GNS2/tiles/tile_gns2_Ori_f8.txt',  format = 'ascii')
# gns2['l'] = gns2['l']*u.degree
# gns2['b'] = gns2['b']*u.degree
# gns2['H'] = gns2['m'] 
# gns2['sl'] = gns2['sl']*u.arcsec
# gns2['sb'] = gns2['sb']*u.arcsec
# =============================================================================

# gns2 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/pruebas/F{field_two}_old/{field_two}_H_chips_opti.ecsv', format = 'ascii.ecsv')
# gns2 = Table.read(f'/Users/amartinez/Desktop/Projects/GNS_gd/pruebas/F{field_two}/{field_two}_H_chips_opti_rebf1.ecsv', format = 'ascii.ecsv')
gns2['ID'] = np.arange(len(gns2))

gns1_wr = Longitude(gns1['l']).wrap_at('180d')
gns2_wr = Longitude(gns2['l']).wrap_at('180d')

gns1['l'] = Longitude(gns1['l']).wrap_at('180d')
gns2['l'] = Longitude(gns2['l']).wrap_at('180d')


buenos1 = (gns1_wr > min(gns2_wr)) & (gns1_wr < max(gns2_wr)) & (gns1['b']>min(gns2['b'])) & (gns1['b']<max(gns2['b']))

gns1 = gns1[buenos1]
gns1_wr = gns1_wr[buenos1]

# %%

# extra_cut = -0.08
# #
# gns1 = gns1[gns1['l'] > extra_cut]
# gns1_wr = gns1_wr[gns1_wr > extra_cut*u.deg]


# 

# %%
# fig, ax = plt.subplots()
# br = gns1['H']<13
# ax.set_title('Density plot')
# # ax.scatter(gns2_mpm['l'], gns2_mpm['b'], s= 1)
# # hst = ax.hist2d(gns1['l'], gns1['b'], bins = 100)
# hst = ax.hist2d(gns1['l'][br], gns1['b'][br], bins = 20)
# fig.colorbar(hst[3], ax = ax)
# ax.invert_xaxis()
# sys.exit(249)
# %%


# gns1['ID'] = np.arange(len(gns1))
all_1 = len(gns1)

fig, (ax, ax2) = plt.subplots(1,2, figsize = (20,10))
ax.hist2d(gns1[band1],gns1['sl'], bins = 100,norm = LogNorm())
his = ax2.hist2d(gns1[band1],gns1['sb'], bins = 100,norm = LogNorm())
fig.colorbar(his[3], ax =ax2)
ax.set_title('GNS1')
ax.set_ylabel('$\sigma l$ [arcsec]')
ax2.set_ylabel('$\sigma b$ [arcsec]')
ax.set_xlabel(band1)
ax2.set_xlabel(f'[{band1}]')
fig.tight_layout()
ax.axhline(max_sig,ls = 'dashed', color = 'r')
if gns_mags[1] is not None:
    ax.axvline(gns_mags[1],ls = 'dashed', color = 'r')

ax.set_xticks(np.arange(min(np.floor(gns1[band1])), max(gns1[band1]),1))
ax2.set_xticks(np.arange(min(np.floor(gns1[band1])), max(gns1[band1]),1))

ax.grid()
ax2.grid()

fig, ax = plt.subplots(1,1)
hits = ax.hist2d(gns1[band1],(gns1['sl'] +gns1['sl'])/2,  bins = 100,norm = LogNorm())
fig.colorbar(his[3], ax =ax)
ax.set_ylabel('($\sigma l + + \sigma b$)/2 [arcsec]')
ax.set_xlabel(f'[{band1}]')
fig.tight_layout()
    
# %%


buenos2 = (gns2_wr > min(gns1_wr)) & (gns2_wr < max(gns1_wr)) & (gns2['b']>min(gns1['b'])) & (gns2['b']<max(gns1['b']))

gns2 = gns2[buenos2]

gns2_wr = gns2_wr[buenos2]
# gns2['ID'] = np.arange(len(gns2))
# gns2 = filter_gns_by_percentile(gns2, mag_col='H', err_col='dH', sl_col='sl', sb_col='sb', bin_width=bin_width, percentile_H=perc_H, percentile_lb=perc_lb, mag_lim = None, pos_lim = None)

fig, ax = plt.subplots(1,1)
ax.scatter(gns2_wr, gns2['b'] )
ax.scatter(gns1_wr, gns1['b'] )
ax.axis('equal')



# %%
fig2, (ax_2, ax2_2) = plt.subplots(1,2, figsize = (16,8)) 
ax_2.set_title('GNS2')
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

# ax_2.set_xlim(12,23)
# ax_2.set_ylim(0.00,0.125)
ax_2.set_xticks(np.arange(min(np.floor(gns2['H'])), max(gns2['H']),1))
ax2_2.set_xticks(np.arange(min(np.floor(gns2['H'])), max(gns2['H']),1))
ax_2.grid()
ax2_2.grid()


all_2 = len(gns2)

gns1 = filter_gns_data(gns1, max_e_pos = max_sig, max_mag = gns_mags[0], min_mag = gns_mags[1], band = band1 )
gns2 = filter_gns_data(gns2, max_e_pos = max_sig, max_mag = gns_mags[0], min_mag = gns_mags[1] )

# %%

# Plot for article
# =============================================================================
# fig, (ax,ax2) = plt.subplots(1,2, figsize = (15,7.5))
# hist = ax.hist2d(gns1['H'],1000*(gns1['sl'] +gns1['sl'])/2,  bins = 100,norm = LogNorm())
# fig.colorbar(hist[3], ax =ax, aspect = 30)
# # ax.set_ylabel('($\sigma l + \sigma b$)/2 [arcsec]')
# # ax.set_ylabel('$\overline{\sigma}_{(l,b)}$ [arcsec]')
# ax.set_ylabel('$\overline{\sigma}_{(l,b)}$ [mas]')
# ax.set_xlabel('[H]')
# 
# hist2 = ax2.hist2d(gns2['H'],1000*(gns2['sl'] +gns2['sl'])/2,  bins = 200,norm = LogNorm(), cmap = 'inferno')
# fig.colorbar(hist2[3], ax =ax2, label = 'stars/bin', aspect = 30)
# # ax.set_ylabel('($\sigma l + \sigma b$)/2 [arcsec]')
# ax2.set_xlabel('[H]')
# # ax2.set_ylim(0.001*1000,0.1*1000)
# ax2.set_xlim(10,23)
# ax.set_xlim(10,23)
# # a1x.set_xticks(np.arange(int(min(gns1['H'])),max(gns1['H']),1 ))
# ax.set_xticks(np.arange(10,23,2 ))
# ax.set_yticks(np.arange(0,450,50 ))
# ax2.set_yticks(np.arange(0,450,50 ))
# ax2.set_xticks(np.arange(10,23,2 ))
# ax.set_ylim(0,400)
# ax2.set_ylim(0,400)
# ax.axhline(50, color = 'red', ls ='dashed')
# ax2.axhline(50, color = 'red', ls ='dashed', lw = 2)
# fig.tight_layout()
# 
# meta = {'Script': '/Users/amartinez/Desktop/PhD/HAWK/GNS_pm_scripts/GNS_pm_relative_SUPER/SUPER_alignment.py'}
# =============================================================================
# plt.savefig('/Users/amartinez/Desktop/PhD/My_papers/GNS_pm_catalog/images/GNS1-2_vs_sigmalb.png', bbox_inches='tight', dpi = 150, transparent = True, metadata = meta)
# sys.exit(378)
# %%
# br = gns2['H']<14
# fig,ax = plt.subplots(1,1)
# ax.set_title('Density plot')
# # ax.scatter(gns2_mpm['l'], gns2_mpm['b'], s= 1)
# # hst = ax.hist2d(gns1['l'], gns1['b'], bins = 100)
# hst = ax.hist2d(gns2['l'][br], gns2['b'][br], bins = 20, cmap = 'inferno')
# fig.colorbar(hst[3], ax = ax)
# ax.invert_xaxis()
# sys.exit(249)
# %%



# =============================================================================
# # %
# def perX(arr):
#     return np.percentile(arr, 10)
# mask_kn = (gns1['H'] > 12) & (gns1['H'] < 20)
# gns1_kn = gns1['H'][mask_kn]
# sig_m1 = (gns1['sl'][mask_kn]*1000 +gns1['sb'][mask_kn]*1000)/2
# sig_b,b_edg, bin_n = binned_statistic(gns1_kn,sig_m1,statistic = perX, bins = 150)
# # sig_b,b_edg, bin_n = binned_statistic(gns1_kn,sig_m1,statistic = 'median', bins = 300)
# 
# x_c = (b_edg[1:] + b_edg[:-1])/2   
# ax.scatter(x_c, sig_b, color = 'r', s = 1)
# 
# kl = KneeLocator(x_c, sig_b, curve="convex", direction="increasing")
# print("Knee at:", kl.knee)
# 
# 
# mask_kn2 = (gns2['H'] > 12) & (gns2['H'] < 22)
# gns2_kn = gns2['H'][mask_kn2]
# sig_m2 = (gns2['sl'][mask_kn2]*1000 +gns2['sb'][mask_kn2]*1000)/2
# sig_b,b_edg, bin_n = binned_statistic(gns2_kn,sig_m2,statistic = perX, bins = 150)
# # sig_b,b_edg, bin_n = binned_statistic(gns1['H'],sig_m1,statistic = 'std', bins = 200)
# 
# x_c = (b_edg[1:] + b_edg[:-1])/2   
# ax2.scatter(x_c, sig_b, color = 'cyan', s = 10)
# 
# kl = KneeLocator(x_c, sig_b, curve="convex", direction="increasing")
# print("Knee at:", kl.knee)
# 
# # sys.exit(361)
# # x_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
# =============================================================================
# %%






ax2.set_title(f'Clipped {100 - 100*len(gns1)/all_1:.1f}%')
ax2_2.set_title(f'Clipped {100 - 100*len(gns2)/all_2:.1f}%')



if centered_in == 1:
    center = SkyCoord(l = np.mean(gns1_wr), b = np.mean(gns1['b']), unit = 'degree', frame = 'galactic')
    # center = SkyCoord(l = 359.9443, b = -0.0462, unit = 'degree', frame = 'galactic')
elif centered_in == 2:
    center = SkyCoord(l = np.mean(gns2_wr), b = np.mean(gns2['b']), unit = 'degree', frame = 'galactic')
    

gns1_lb = SkyCoord(l = gns1['l'], b = gns1['b'], unit ='deg', frame = 'galactic')
gns2_lb = SkyCoord(l = gns2['l'], b = gns2['b'], unit ='deg', frame = 'galactic')
# =============================================================================
# # =============================================================================
# # 
# # =============================================================================
# # 1. Select bright stars
# bright_mask = gns1['H'] < 13
# bright_coords = gns1_lb[bright_mask]
# 
# # 2. Search for all pairs of bright stars within a distance
# threshold = 0.5 * u.arcsec
# 
# idx1, idx2, sep2d, _ = search_around_sky(
#         bright_coords,
#         bright_coords,
#         threshold
# )
# 
# check = np.c_[idx1,idx2,sep2d.to(u.mas).value]
# 
# # 3. Remove self-matches (each star matches itself at sep=0)
# self_matches = idx1 != idx2
# idx1 = idx1[self_matches]
# idx2 = idx2[self_matches]
# 
# # 4. Any star involved in a "close pair" is considered a duplicate
# duplicate_bright = np.zeros(len(bright_coords), dtype=bool)
# duplicate_bright[np.unique(idx1)] = True
# duplicate_bright[np.unique(idx2)] = True
# 
# # 5. We keep only the isolated bright stars
# keep_bright = np.logical_not(duplicate_bright)
# 
# # 6. Build final mask for the full catalog
# final_mask = np.ones(len(gns1), dtype=bool)
# final_mask[bright_mask] = keep_bright
# 
# # 7. Apply filtering
# gns1 = gns1[final_mask]
# gns1_lb = gns1_lb[final_mask]
# # =============================================================================
# # 
# # =============================================================================
# 
# # =============================================================================
# # 
# # =============================================================================
# # 1. Select bright stars
# bright_mask = gns2['H'] < 13
# bright_coords = gns2_lb[bright_mask]
# 
# # 2. Search for all pairs of bright stars within a distance
# threshold = 0.5 * u.arcsec
# 
# idx1, idx2, sep2d, _ = search_around_sky(
#         bright_coords,
#         bright_coords,
#         threshold
# )
# 
# check = np.c_[idx1,idx2,sep2d.to(u.mas).value]
# 
# # 3. Remove self-matches (each star matches itself at sep=0)
# self_matches = idx1 != idx2
# idx1 = idx1[self_matches]
# idx2 = idx2[self_matches]
# 
# # 4. Any star involved in a "close pair" is considered a duplicate
# duplicate_bright = np.zeros(len(bright_coords), dtype=bool)
# duplicate_bright[np.unique(idx1)] = True
# duplicate_bright[np.unique(idx2)] = True
# 
# # 5. We keep only the isolated bright stars
# keep_bright = np.logical_not(duplicate_bright)
# 
# # 6. Build final mask for the full catalog
# final_mask = np.ones(len(gns2), dtype=bool)
# final_mask[bright_mask] = keep_bright
# 
# # 7. Apply filtering
# gns2 = gns2[final_mask]
# gns2_lb = gns2_lb[final_mask]
# # =============================================================================
# # 
# # =============================================================================
# 
# =============================================================================

xg_1, yg_1 = center.spherical_offsets_to(gns1_lb)
xg_2, yg_2 = center.spherical_offsets_to(gns2_lb)

tag = center.skyoffset_frame()

gns1_t = gns1_lb.transform_to(tag)
gns2_t = gns2_lb.transform_to(tag)


gns1['xp'] = gns1_t.lon.to(u.arcsec)
gns1['yp'] = gns1_t.lat.to(u.arcsec)
gns2['xp'] = gns2_t.lon.to(u.arcsec)
gns2['yp'] = gns2_t.lat.to(u.arcsec)




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
ax.invert_xaxis()
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
ax.invert_xaxis()
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

idx1, idx2, sep2d, _ = search_around_sky(gns1_lb, gns2_lb, max_sep)

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



if band1 == 'H':
    diff_H = gns2_match['H']-gns1_match[band1]
    
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


# stop(595)
# %%
g1_m = np.array([gns1_match['xp'],gns1_match['yp']]).T
g2_m = np.array([gns2_match['xp'],gns2_match['yp']]).T

sig_cl_H_aligment = sig_cl_H
if destination == 1:
    # Time lapse to move Gaia Stars.
    tg = Time(['2016-01-01T00:00:00'],scale='utc')
    dtg = t1 - tg
    
    p,(_,_)= aa.find_transform(g2_m,g1_m,max_control_points=100)
    
    # p = ski.transform.estimate_transform('similarity',
    #                                 g2_m, 
    #                                 g1_m)
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
    gns2 = alg_loop(gns2, gns1, 'xp', 'yp', align_by, max_deg, d_m.to(u.arcsec).value, max_loop,sig_cl_H = sig_cl_H_aligment, 
                    grid_s = grid_s, grid_Hmin = grid_Hmin, grid_Hmax = grid_Hmax ,isolation_radius = isolation_radius,  f_mode = f_mode, mag_lim_alig=mag_lim_alig)

if destination == 2:
    # Time lapse to move Gaia Stars.
    tg = Time(['2016-01-01T00:00:00'],scale='utc')
    dtg = t2 - tg
    
    if init_alig == 'astroalign':
        p,(_,_)= aa.find_transform(g1_m,g2_m,max_control_points=100)
    
        print("Translation: (x, y) = (%.2f, %.2f)"%(p.translation[0],p.translation[1]))
        print("Rotation: %.2f deg"%(p.rotation * 180.0/np.pi)) 
        print("Rotation: %.0f arcmin"%(p.rotation * 180.0/np.pi*60)) 
        print("Rotation: %.0f arcsec"%(p.rotation * 180.0/np.pi*3600))  
    elif init_alig == 'similarity':
        p = ski.transform.estimate_transform('similarity',
                                        g1_m, 
                                        g2_m)
        print("Translation: (x, y) = (%.2f, %.2f)"%(p.translation[0],p.translation[1]))
        print("Rotation: %.2f deg"%(p.rotation * 180.0/np.pi)) 
        print("Rotation: %.0f arcmin"%(p.rotation * 180.0/np.pi*60)) 
        print("Rotation: %.0f arcsec"%(p.rotation * 180.0/np.pi*3600)) 
    else:
        order = int(init_alig[-1])
        p = ski.transform.estimate_transform('polynomial',
                                        g1_m, 
                                        g2_m, order = 2)
        
    
    
    
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
    
    
    # def alg_loop(gns_A, gns_B,col1, col2, align_by, max_deg, d_m,                    max_loop,sig_cl_H, grid_s = None, grid_Hmin = None, grid_Hmax = None,dm_plots = None, f_mode = None, mag_lim_alig = None  ) :

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



dx = (gns2_mi['xp'].value - gns1_mi['xp'].value)*1000
dy = (gns2_mi['yp'].value - gns1_mi['yp'].value)*1000


pm_x = (dx*u.mas)/dt.to(u.year)
pm_y = (dy*u.mas)/dt.to(u.year)

if add_aligm_error == 'yes':
    if destination ==2:
        filepath = os.path.join(bs1, f'uncer_alig_F1_{field_one}_F2_{field_two}.txt')
    if destination ==1:
        filepath = os.path.join(bs2, f'uncer_alig_F1_{field_one}_F2_{field_two}.txt')
    
    if os.path.exists(filepath):
        print("✅ File exists:", filepath)
        gns1_mi['sl_xalign'] = gns1_mi['sl']
        gns1_mi['sb_yalign'] = gns1_mi['sb']
        bs_gns = Table.read(filepath, format = 'ascii')
        x_std_map = {row['ID']: row['x_std'] for row in bs_gns}
        y_std_map = {row['ID']: row['y_std'] for row in bs_gns}
        
    # Loop through gns1_mi and update when ID is in bs_gns1
        for i, star_id in enumerate(gns1_mi['ID']):
            if star_id in x_std_map:
                gns1_mi['sl_xalign'][i] = np.sqrt(gns1_mi['sl'][i]**2 +  x_std_map[star_id]**2)
            if star_id in y_std_map:
                gns1_mi['sb_yalign'][i] =  np.sqrt(gns1_mi['sb'][i]**2 +  y_std_map[star_id]**2)
                
        bs_gns = Table.read(filepath, format = 'ascii')
        ebs_mean, bs_bins, _= stats.binned_statistic(bs_gns['H'],(bs_gns['x_std'] + bs_gns['y_std'])/2, bins = 10)
        e_mean, all_bins,_  = stats.binned_statistic(gns1_mi['H'],(gns1_mi['sl'] + gns1_mi['sb'])/2 , bins = 10)
        e_mean2, all_bins2,_ =  stats.binned_statistic(gns1_mi['H'],(gns1_mi['sl_xalign'] + gns1_mi['sb_yalign'])/2 , bins = 10)
        # ebs_mean, bs_bins, _= stats.binned_statistic(bs_gns['H'],(bs_gns['x_std']**2 + bs_gns['y_std']**2)**0.5, bins = 10)
        # e_mean, all_bins,_  = stats.binned_statistic(gns1_mi['H'],(gns1_mi['sl']**2 + gns1_mi['sb']**2)**0.5 , bins = 10)
        bin_c = 0.5*(bs_bins[1:] + bs_bins[:-1])
        bin_ca = 0.5*(all_bins[1:] + all_bins[:-1])
        bin_ca2 = 0.5*(all_bins2[1:] + all_bins2[:-1])


        fig, ax = plt.subplots(1,1)

        ax.plot(bin_c,ebs_mean*1000, label = '$\sigma_{DR1}$')
        ax.plot(bin_ca,e_mean*1000,label =  '$\sigma_{align}$')
        ax.plot(bin_ca2,e_mean2*1000,label =  '$\sigma$')
        ax.semilogy()
        ax.legend()
        # ax.set_yticks([0.9,1,2,3,4,5,6])
        # ax.set_yticklabels([0.9,1,2,3,4,5,6])
        ax.set_xlabel('[H]')
        ax.set_ylabel('$\overline{\sigma}_{pos}$ [mas]')
        # ax.set_ylim(-0,0.002*1000)
    
    else:
        print("❌ File not found:", filepath)
        add_aligm_error = 'no'
        
    


if add_aligm_error == 'yes':
    dpm_x = np.sqrt((gns2_mi['sl'].to(u.mas))**2 + (gns1_mi['sl_xalign'].to(u.mas))**2)/dt.to(u.year)
    dpm_y = np.sqrt((gns2_mi['sb'].to(u.mas))**2 + (gns1_mi['sb_yalign'].to(u.mas))**2)/dt.to(u.year)
if add_aligm_error == 'no':
    dpm_x = np.sqrt((gns2_mi['sl'].to(u.mas))**2 + (gns1_mi['sl'].to(u.mas))**2)/dt.to(u.year)
    dpm_y = np.sqrt((gns2_mi['sb'].to(u.mas))**2 + (gns1_mi['sb'].to(u.mas))**2)/dt.to(u.year)

       

# pm_l = (dl*mean_b)/dt.to(u.year)
# pm_b = (db)/dt.to(u.year)

if sig_cl_pm is not None:
    m_pm, lims = sig_f(pm_x, pm_x, sig_cl_pm)
    
    
    pm_x = pm_x[m_pm]
    pm_y = pm_y[m_pm]
    dpm_x = dpm_x[m_pm]
    dpm_y = dpm_y[m_pm]
    
    gns1_mi = gns1_mi[m_pm]
    gns2_mi = gns2_mi[m_pm]

gns1_mi['pm_x']  = pm_x
gns1_mi['pm_y']  = pm_y

gns2_mi['pm_x']  = pm_x
gns2_mi['pm_y']  = pm_y

gns1_mi['dpm_x']  = dpm_x
gns1_mi['dpm_y']  = dpm_y

gns2_mi['dpm_x']  = dpm_x
gns2_mi['dpm_y']  = dpm_y


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

gns1_mi.write(pruebas1 + f'gns1_pmSuper_F1_{field_one}_F2_{field_two}.ecsv', format = 'ascii.ecsv', overwrite = True)

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

gns2_mi.write(pruebas2 + f'gns2_pmSuper_F1_{field_one}_F2_{field_two}.ecsv', format = 'ascii.ecsv', overwrite = True)


# =============================================================================
# Hyper-velocity star
# =============================================================================

# hv = 25# We cosider hv if pm > 25 mas/yr (1000 km/s)

# hv_mask = np.sqrt(gns1_mi['pm_x']**2 + gns1_mi['pm_y']**2) > hv

# gns1_hv = gns1_mi[hv_mask]
# gns2_hv = gns2_mi[hv_mask]

# gns1_hv['H2'] = gns2_hv['H']
# gns1_hv.write(pruebas1 +  f'HV_gns1_pmSuper_F1_{field_one}_F2_{field_two}.ecsv', format = 'ascii.ecsv', overwrite = True)
# print(gns1_hv['pm_x', 'pm_y','H', 'H2'])
# stop()

# %%
fig, ax = plt.subplots(figsize = (10,10))
# mean_elb = (gns2_mi['dpm_x'] + gns2_mi['dpm_y'])/2
mean_elb = (gns2_mi['dpm_x']**2 + gns2_mi['dpm_y']**2)**0.5
# hpm = ax.hist2d(gns2_mi['H'], mean_elb, norm = LogNorm(), bins = 200)
hpm = ax.hexbin(gns2_mi['H'], mean_elb, norm = LogNorm(), gridsize = 60)

# fig.colorbar(hpm[3], ax = ax, label = 'stars/bin', aspect = 30)
fig.colorbar(hpm, ax = ax, label = 'stars/bin', aspect = 30)
ax.set_xlabel('[H]')
ax.set_ylabel('$\overline{\sigma}_{\mu}$ [mas/yr]')
ax.set_ylim(0,)
ax.set_yticks(np.arange(1,11,1))
# ax.set_xticks(np.arange(11,23,1))
# ax.axhline(e_pm_gns, ls = 'dashed', color = 'red')
meta = {'Script': '/Users/amartinez/Desktop/PhD/HAWK/GNS_pm_scripts/GNS_pm_relative_SUPER/SUPER_alignment.py'}
# plt.savefig(f'/Users/amartinez/Desktop/PhD/My_papers/GNS_pm_catalog/images/pmError_vs_maxH{gns_mags[1]}_hexa.png', dpi = 150, transparent = True, metadata = meta)
# ax.grid()

# stop(800)
# %%
# Pm uncertaintis plots
gns1_mm = filter_gns_data(gns1_mi, max_e_pm = e_pm_gns)
gns2_mm = filter_gns_data(gns2_mi, max_e_pm = e_pm_gns)
lim_br =16
lim_rc = [16,18]
lm_s = [0.5,1]
gns2_br = gns2_mm[gns2_mm['H']<lim_br]
gns2_rc = gns2_mm[(gns2_mm['H']<lim_rc[1]) & (gns2_mm['H']>lim_rc[0])]

fig, ax = plt.subplots(1,1)
# ax.set_title(f'RELATIVE GNS1= f{field_one}, GNS2 = F{field_two}')
normal = False
lw = 3
# mean_br = (gns2_br['dpm_x'] + gns2_br['dpm_y'])/2
# mean_br = gns2_br['dpm_y'] 
mean_br = np.sqrt(gns2_br['dpm_x']**2 + gns2_br['dpm_y']**2)
mean_rc = (gns2_rc['dpm_x'] + gns2_rc['dpm_y'])/2
# mean_rc = gns2_rc['dpm_x']

# mean_sl = (gns2_sl['dpm_x'] + gns2_sl['dpm_y'])/2
# mean_sl = (gns2_sl['dpm_x']**2 + gns2_sl['dpm_y']**2)**0.5

# mean_sl2 = (gns2_sl2['dpm_x'] + gns2_sl2['dpm_y'])/2
ax.hist(mean_br , histtype = 'step', bins = 'auto',lw = lw, density = normal, label = f'H < {lim_br}; $P_{{50}}$ = {np.percentile(mean_br,50):.2f}',zorder = 5)
ax.hist(mean_rc, histtype = 'step', bins = 'auto',lw = lw, density = normal, label = f'{lim_rc[0]} < H < {lim_rc[1]}; $P_{{50}}$  = {np.percentile(mean_rc,50):.2f}')
# ax.hist(mean_sl, histtype = 'step', bins = 'auto',lw = lw, density = normal, label = f'pm < {lm_s[0]}; $P_{{50}}$  = {np.percentile(mean_sl,50):.2f}')
# ax.hist(mean_sl2, histtype = 'step', bins = 'auto',lw = lw, density = normal, label = f'pm < {lm_s[1]}; $P_{{50}}$  = {np.percentile(mean_sl,50):.2f}')
# ax.hist(gns2_br['dpm_y'], histtype = 'step', bins = 'auto')
# ax.semilogy()
ax.set_xlim(0,2)
ax.set_xticks(np.arange(0,2,0.5))
ax.grid()
ax.set_xlabel('$\overline{\sigma}_{\mu}$ [mas/yr]')
ax.set_ylabel('N')
ax.legend()

meta = {'Script': '/Users/amartinez/Desktop/PhD/HAWK/GNS_pm_scripts/GNS_pm_relative_SUPER/SUPER_alignment.py'}
# plt.savefig(f'/Users/amartinez/Desktop/PhD/My_papers/GNS_pm_catalog/images/gns_epm_hist_f1_{field_one}_F2_{field_two}.png', dpi = 150, transparent = True, metadata = meta)
# sys.exit(840)
# %%
# e_pm_gns = 0.75##
gns1_m = filter_gns_data(gns1_mi, max_e_pm = e_pm_gns)
gns2_m = filter_gns_data(gns2_mi, max_e_pm = e_pm_gns)

bA, cA = l_function(gns2_mi, 'H', 0.3, m_l = 10, m_h = 22)
bB, cB = l_function(gns2_m, 'H', 0.3, m_l = 10, m_h = 22)

plt.figure()

plt.plot(bA, cA, marker = '.', label = 'All pm')
plt.plot(bB, cB, marker = '.', label = f'$\sigma \mu$ < {e_pm_gns} ')
plt.legend()
plt.gca().set_yscale('log')
plt.show()
# stop(1045)
# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable
if band1 == 'H':
    dh = gns2_m['H'] - gns1_m['H']
    
    
    
    fig, (ax,ax2,ax3) = plt.subplots(1,3, figsize = (21,7))
    ax.hist(dh, bins = 'auto', label = '$\overline{\Delta H}$ =%.2f\n$\sigma$ = %.2f'%(np.mean(dh) ,np.std(dh) ))
    ax2.axhline(l_lim, color = 'red', ls = 'dashed')
    ax2.axhline(h_lim, color = 'red', ls = 'dashed')
    ax.legend()
    ax.set_title(f'gns1 -> f{field_one}c{chip_one}. GNS2 -> F{field_two}C{chip_two}')
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

# pm_lim = (abs(gns1_m['pm_x'])<4)
# gns1_m = gns1_m[pm_lim]
# %%
bins = 'auto'
fig, (ax,ax2) = plt.subplots(1,2)
if mag_lim_alig is not None:
    ax.set_title(f'Ref. GNS{destination}. Deg {max_deg-1}, Mag_lim_alig {mag_lim_alig[0], mag_lim_alig[1]}', fontsize= 15)
else:
    ax.set_title(f'Ref. GNS{destination}. Deg {max_deg-1}', fontsize= 15)
ax2.set_title(f'E_pm < {e_pm_gns}mas/yr. H = [{gns_mags[0], gns_mags[1]}]', fontsize= 15)

ax.hist(gns1_m['pm_x'], bins = bins, histtype = 'step', label = '$\overline{\mu}_{x}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(gns1_m['pm_x'].value),np.std(gns1_m['pm_x'].value)),)
ax.hist(gns1_mi['pm_x'], bins = 'auto',color = 'k', alpha = 0.1)
ax2.hist(gns1_m['pm_y'], bins = bins, histtype = 'step',label = '$\overline{\mu}_{y}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(gns1_m['pm_y'].value),np.std(gns1_m['pm_y'].value)))
ax.set_xlabel('$\Delta \mu_{x}$ [mas/yr]')
ax2.set_xlabel('$\Delta\mu_{y}$ [mas/yr]')
# ax.axvline(lims[0].value , ls = 'dashed', color = 'r', label = f'{sig_dis}$\sigma$')
# ax.axvline(lims[1].value , ls = 'dashed', color = 'r')
# ax2.axvline(lims[2].value , ls = 'dashed', color = 'r')
# ax2.axvline(lims[3].value , ls = 'dashed', color = 'r')
# ax.set_xlim(-20,20)
# ax2.set_xlim(-20,20)
ax.invert_xaxis()

# ax.invert_xaxis()
ax.legend( fontsize = 15)
ax2.legend(fontsize = 15)

# stop(984)

# %%
fig, ax = plt.subplots(figsize = (10,10))
mean_elb = (gns2_mi['dpm_x'] + gns2_mi['dpm_y'])/2
hpm = ax.hist2d(gns2_mi['H'], mean_elb, norm = LogNorm(), bins = (80))

fig.colorbar(hpm[3], ax = ax, label = 'stars/bin', aspect = 40)
ax.set_xlabel('[H]')
ax.set_ylabel('$\overline{\mu}_{(l,b)}$ [mas/yr]')

# sys.exit(712)
# %%
# =============================================================================
# # %% Gaia Comparation#!!!
# =============================================================================
# max_sep_ga = 100*u.mas# separation for comparison with gaia
# e_pm_gaia = 1#!!! Maximun error in pm for Gaia stars
# gaia_mags = [0,19]#!!! Gaia mag limtis for comparison with GNS
# %
# Before comparing witg Gaia we mask the best pms



extra_mag_cut = [12,18]
extra_epm = e_pm_gns
# extra_epm = 0.5
gns1_mpm = filter_gns_data(gns1_m, max_e_pm = extra_epm, min_mag = extra_mag_cut[1], max_mag = extra_mag_cut[0], band = band1)
gns2_mpm = filter_gns_data(gns2_m, max_e_pm = extra_epm,  min_mag = extra_mag_cut[1], max_mag = extra_mag_cut[0])


# radius = abs(np.min(gns1_wr)-np.max(gns1_wr))*r_search_gaia.value 
radius = 300*u.arcsec
# radius = abs(np.min(gns1['l'])-np.max(gns1['l']))*2.6*u.degree
# center_g = SkyCoord(l = np.mean(gns1['l']), b = np.mean(gns1['b']), unit = 'degree', frame = 'galactic')

try:
    gaia = Table.read(pruebas1  + '------gaia_f1%s_f2%s_r%.0f.ecsv'%(field_one,field_two,radius.to(u.arcsec).value))
    print('Gaia from table')
except:
    print('Gaia from web')
    gaia = Table.read('/Users/amartinez/Desktop/PhD/Catalogs/Gaia/gaia_over_GNS.txt', format = 'ascii.ecsv')    
    gaia['id'] = np.arange(len(gaia))
    gaia['l'] = Longitude(gaia['l']).wrap_at('180d')
    gaia_coord = SkyCoord(ra = gaia['ra'], dec = gaia['dec'],
                       frame = 'icrs', obstime = 'J2016').galactic
    
    
    sep = gaia_coord.separation(center)
    mask_gaia = sep < radius
    gaia = gaia[mask_gaia]
    gaia.write(pruebas1  + 'gaia_f1%s_f2%s_r%.0f.ecsv'%(field_one,field_two,radius.to(u.arcsec).value), overwrite = True)

gaia['l'] = Longitude(gaia['l']).wrap_at('180d')


# sys.exit(1028)
gaia.sort('phot_g_mean_mag')
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

x = np.array(gaia['phot_g_mean_mag'])
y = np.array(gaia['pmra_error'])

# sort both arrays by x
sort_idx = np.argsort(x)
x_sorted = x[sort_idx]
y_sorted = y[sort_idx]

num_bins = 500

# =============================================================================
# # This find Gaia kneew
# =============================================================================
# =============================================================================
# for i,column in enumerate((gaia['pmra_error'],gaia['pmdec_error'])):
#     
#     x = np.array(gaia['phot_g_mean_mag'])
#     y = np.array(column)
#     zero_mask = (y >0) & (x>12) 
#     x = x[zero_mask]
#     y = y[zero_mask]
#     
#     # y_min, bin_edges, _ = binned_statistic(gaia['phot_g_mean_mag'], column, statistic='median', bins=num_bins)
#     
#     # x_bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
#    
#     def perc10(arr):
#         return np.percentile(arr, 10)
#     y_env, bin_edges, _ = binned_statistic(x,y, statistic=perc10, bins=num_bins)
#     x_env = 0.5 * (bin_edges[1:] + bin_edges[:-1])
# 
#     # Remove NaNs
#     mask = ~np.isnan(y_env)
#     x_env, y_env = x_env[mask], y_env[mask]
# 
# 
#     mask = (y_env >0) & (x_env > 12)
#     x_env = x_env[mask]
#     y_env = y_env[mask]
#     
#     def power(x, a, b, c):
#         return a * (x**b) + c
# 
#     # --- Fit power law ---
#     popt_pow, _ = curve_fit(power, x_env, y_env, maxfev=10000)
#     # popt_pow, _ = curve_fit(power, x_env, y_env, p0=(1e-5, 2, 1e-5), maxfev=10000)
# 
#     xx = np.linspace(min(x_env), max(x_env), num_bins)
#     
#     kl = KneeLocator(xx, power(xx, *popt_pow), curve="convex", direction="increasing")
#     print("Knee at:", kl.knee)
#     y_kn = power(xx, *popt_pow)[xx == kl.knee]
#     # gax.axvline(kl.knee,ls = 'dashed')
#     # gax.axhline(y_kn,ls = 'dashed')
# # gax.scatter(gaia['phot_g_mean_mag'],gaia['pmra_error'])
#     if i == 0:
#         gax.axvline(kl.knee,ls = 'dashed', label = f'$l_k$  {kl.knee:.1f}, {y_kn[0]:.1f}')
#         gax.axhline(y_kn,ls = 'dashed')
#         gax.plot(xx, power(xx, *popt_pow))
#         # gax.scatter(x, y, s=5, alpha=0.2)
#         # gax.scatter(x_, y_min, color="red") 
#     
#     if i == 1:
#         gax.axvline(kl.knee,ls = 'dashed', label = f'$l_b$  {kl.knee:.1f}, {y_kn[0]:.1f}')
#         gax.axhline(y_kn,ls = 'dashed')
#         gax.plot(xx, power(xx, *popt_pow))
#         # gax.scatter(x, y, s=5, alpha=0.2)
#         # gax.scatter(x_bin_centers, y_min, color="red",marker ='x') 
#     gax.set_xticks(np.arange(min(np.floor(x-2)), max(x+1)))
#     gax.legend()
# # %
# =============================================================================

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

gaia['xp'] = c_proj.lon.to(u.arcsec)
gaia['yp'] = c_proj.lat.to(u.arcsec)
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
    pm_min=None,
    pmra_error_max=e_pm_gaia,
    pmdec_error_max=e_pm_gaia,
    ra_error_max = e_pos_gaia,
    dec_error_max = e_pos_gaia,
    # min_angular_separation_arcsec = 0.1*u.arcsec
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

gaia['l'] = Longitude(gaia['l']).wrap_at('180d')
# %
# gaia_c = SkyCoord(ra = gaia['ra'], dec = gaia['dec'], 
#                   pm_ra_cosdec = gaia['pmra'], pm_dec = gaia['pmdec'],
#                   frame = 'icrs', obstime = 'J2016.0').galactic



gaia['pm_l'] = ga_c.pm_l_cosb
gaia['pm_b'] = ga_c.pm_b

print("σ(Gaia_μl - Gaia_μx):", np.std(gaia['pm_l'] - gaia['pm_x']))
print("σ(Gaia_μb  - Gaia_μy):",np.std(gaia['pm_b'] - gaia['pm_y']))


# %

ga_l = ga_c.l.wrap_at('180d')
# ga_b = ga_c.b.wrap_at('180d')



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
    


l2 = gns2_gal.l.wrap_at('180d')
l1 = gns1_gal.l.wrap_at('180d')

# %


# %


# =============================================================================
# ⚠️⚠️⚠️⚠️WARNING!!!⚠️⚠️⚠️
# The pm errors, in both Gaia and GNS have to been proyected to the tangetial plane as well!!!
# =============================================================================
# %%
# 



# d_pmx_ga = gns_ga['pm_x']- ga_gns['pm_x']
# d_pmy_ga = gns_ga['pm_y']- ga_gns['pm_y']
d_pmx_ga = gns_ga['pm_x'] - ga_gns['pm_l']
d_pmy_ga = gns_ga['pm_y'] - ga_gns['pm_b']
# 

e_dpmx = np.sqrt(gns_ga['dpm_x']**2  + ga_gns['pmra_error']**2)
e_dpmy = np.sqrt(gns_ga['dpm_y']**2  + ga_gns['pmdec_error']**2)

# sig_ga = 3
m_pm, lxy = sig_f(d_pmx_ga, d_pmy_ga, sig_ga)

gns_ga_m = gns_ga[m_pm]
ga_gns_m = ga_gns[m_pm]

d_pmx_ga_m = d_pmx_ga[m_pm]
d_pmy_ga_m = d_pmy_ga[m_pm]

e_dpmx_m = e_dpmx[m_pm]
e_dpmy_m = e_dpmy[m_pm]

gax.scatter(ga_gns['phot_g_mean_mag'],ga_gns['pmra_error'])
gax.scatter(ga_gns['phot_g_mean_mag'],ga_gns['pmdec_error'])
gax2.scatter(gns_ga[band1],gns_ga['dpm_x'])
gax2.scatter(gns_ga[band1],gns_ga['dpm_y'])
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
gns1_mpm['pm_x'] -= np.mean(d_pmx_ga_m.value)
gns1_mpm['pm_y'] -= np.mean(d_pmy_ga_m.value) 



# %
fig, ax_g = plt.subplots(1,1,figsize =(10,10))
ax_g.set_title(f'Matching distance = {max_sep_ga}. GNS1 F{field_one} GNS2 F{field_two}')
ax_g.scatter(l1[::1], gns1_mpm['b'][::1],s=1, marker = 'x',label = 'GNS_1 Fied %s, chip %s'%(field_one,chip_one))

gaia_lb_ = SkyCoord(l = ga_gns['l'], b = ga_gns['b'], frame = 'galactic')

gaia_lplot = gaia_lb_.l.wrap_at('180d')

# ax.scatter(l2[::10],  gns2_m['b'][::10], label = 'GNS_2 Fied %s, chip %s'%(field_two,chip_two))
clb = ax_g.scatter(ga_gns['l'],ga_gns['b'], c = np.sqrt(ga_gns['pmra_error']**2 + ga_gns['pmdec_error']**2 ),s =200,label = f'Gaia comp pm = {len(ga_gns)}', cmap = 'Spectral_r', edgecolor = 'k')
# clb = ax_g.scatter(ga_gns['l'],ga_gns['b'], c = np.mean(d_pmx_ga) -d_pmx_ga,s =200,label = f'Gaia comp pm = {len(ga_gns)}', cmap = 'Spectral_r', edgecolor = 'k')
# ax_g.set_xlim(360,359.85)
# ax_g.invert_xaxis()
# ax_g.legend()
# stop(1405)
ax_g.set_xlabel('l[deg]', fontsize = 25)
ax_g.set_ylabel('b [deg]', fontsize = 25)
# ax.axis('scaled')
ax_g.set_ylim(min(gaia['b']),max(gaia['b']))
fig.colorbar(clb, ax = ax_g,fraction = 0.0275, aspect = 30, label = 'Gaia pm uncer [mas/yr] \n$\sqrt{\sigma_{pmra}^2 + \sigma_{pmde}^2  }$')
ax_g.scatter(ga_gns[np.logical_not(m_pm)]['l'], ga_gns[np.logical_not(m_pm)]['b'], marker = 'x', color = 'orange', s = 200, lw =5, label = f'{sig_ga}$\sigma$ Clipped Gaia')

gaia_lb_1 = SkyCoord(l = gaia['l'], b = gaia['b'], frame = 'galactic')

gaia_lplot_1 = gaia_lb_1.l.wrap_at('180d')
ax_g.scatter(gaia_lplot_1, gaia['b'], marker = 'x', color = 'k', alpha = 1)
lgnd = ax_g.legend()
ax_g.axis('equal')
ax_g.set_aspect('equal', adjustable='box')
for handle in lgnd.legend_handles:
    handle.set_sizes([200])


#
# max_sep_ga = 80*u.mas# separation for comparison with gaia
# e_pm = 0.5#!!! Maximun error in pm for Gaia stars
# gaia_mags = [13,19]#!!! Gaia mag limtis for comparison with GNS
# # %
# # Before comparing witg Gaia we mask the best pms

# %

# extra_mag_cut = [12,19]
# extra_epm = 0.5
fig, (ax,ax2) = plt.subplots(1,2)
fig.suptitle(f'Gaia-GNS: e_m ={e_pm_gaia,extra_epm}, Mags =[{gaia_mags},{extra_mag_cut}]',fontsize = 15)
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
fig, ax = plt.subplots()
ax.scatter(gns2_mpm['xp'], gns2_mpm['yp'], s= 1)
ax.scatter(gaia['xp'], gaia['yp'], s= 1)
ax.scatter(center.l, center.b, marker = '*', color = 'k', s = 200)
ax.invert_xaxis()
ax.invert_xaxis()
ax.axis('scaled')
sys.exit(1683)
# %%
# region(gns2_mpm, 'l', 'b',name = '88_super', save_in = pruebas2, wcs= 'galactic', color = 'blue', marker = 'x')
region(gns2_mpm, 'l', 'b',name = '88_super', save_in = '/Users/amartinez/Desktop/Projects/GNS_gd/pruebas', wcs= 'galactic', color = 'blue', marker = 'x')


if look_for_cluster == 'yes':
    
   
    # modes = ['pm_xy_color']
    modes = ['pm_xy']
    knn = 20
    gen_sim = 'kernnel'
    # sim_lim ='minimun'
    sim_lim ='mean'
    if destination == 2:
        clus_dic = cluster_finder.finder(gns2_mpm, 'pm_x', 'pm_y',
                                     'xp', 'yp', 
                                     'l', 'b',
                                     modes[0],
                                     'H','H',
                                     knn,gen_sim,sim_lim, save_reg = pruebas2)
    elif destination == 1:
        clus_dic = cluster_finder.finder(gns1_mpm, 'pm_x', 'pm_y',
                                     'xp', 'yp', 
                                     'l', 'b',
                                     modes[0],
                                     'H','H',
                                     knn,gen_sim,sim_lim, save_reg = pruebas1)
    
    
    if len(clus_dic) > 0:
        for i in range(len(clus_dic)):
            region(clus_dic[f'clus_{i}'],  'l', 'b',
                   name = f'clus_{i}_f1_{field_one}_f2_{field_two}',
                   wcs = 'galactic',
                   color = 'red',
                   save_in = pruebas1)
        
        
    # Extract IDs of cluster members
# =============================================================================
#     clus_ids = set(clus_dic['clus_0']['ID'])
# 
#     # Create a membership flag (1 = in cluster, 0 = not in cluster)
#     gns2_mpm['flag'] = np.array([1 if id_ in clus_ids else 0 
#                                  for id_ in gns2_mpm['ID']], dtype=int)
# 
#     
#     fig, ax = plt.subplots(1,1)
#     clusf = gns2_mpm['flag'] == 1
#     ax.scatter(gns2_mpm['l'], gns2_mpm['b'])
#     ax.scatter(gns2_mpm['l'][clusf], gns2_mpm['b'][clusf])
#     ax.axis('equal')
# 
# =============================================================================


    # gns2_mpm.write('/Users/amartinez/Desktop/for_people/for_Carmen/Arches_field.txt', format = 'ascii.ecsv', overwrite=True)

else:
    print('99')
 
stop(1691)
# %%


# %%
# catalogs = '/Users/amartinez/Desktop/PhD/Catalogs/Candela/'
# candela = Table.read( catalogs + 'candela.txt', format = 'ascii')
# region(candela, 'RA', 'Dec',
#        name = 'candela',
#        save_in = '/Users/amartinez/Desktop/PhD/regions/Candela/',
#        wcs = 'fk5',
#        color = 'cyan',
#        marker = 'circle')

# candela_y = Table.read(catalogs + 'candela_young.txt', format = 'ascii')
# region(candela_y, 'RA', 'Dec',
#        name = 'candela_young',
#        save_in = '/Users/amartinez/Desktop/PhD/regions/Candela/',
#        wcs = 'fk5',
#        color = 'blue',
#        marker = 'diamond')

# stop(1650)


# %%
rcParams.update({
    "figure.figsize": (10, 5),
    "font.size": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})


fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax.hist(d_pmx_ga, bins=bins, color='k', alpha=0.2)
ax2.hist(d_pmy_ga, bins=bins, color='k', alpha=0.2)

ax.hist(d_pmx_ga_m, lw=2, bins='auto', histtype='step',
        label='$\overline{\Delta\mu}_{\parallel}$ = %.2f'
              '\n$\sigma_{\parallel}$ = %.2f' % 
              (np.mean(d_pmx_ga_m.value), np.std(d_pmx_ga_m.value)))

ax2.hist(d_pmy_ga_m, lw=2, bins='auto', histtype='step',
         label='$\overline{\Delta\mu}_{\perp}$ = %.2f'
               '\n$\sigma_{\perp}$ = %.2f' % 
               (np.mean(d_pmy_ga_m.value), np.std(d_pmy_ga_m.value)))

ax.set_xlabel(r'$\Delta \mu_{\parallel}$ [mas/yr]')
ax2.set_xlabel(r'$\Delta \mu_{\perp}$ [mas/yr]')
ax.set_ylabel('# stars')
ax.set_xlim(2.5,8.5)
ax2.set_xlim(-2.5,3.5)
ax2.set_xticks(np.arange(-2,3.5))
ax.axvline(lxy[0], ls='dashed', color='r')
ax.axvline(lxy[1], ls='dashed', color='r')
ax2.axvline(lxy[2], ls='dashed', color='r')
ax2.axvline(lxy[3], ls='dashed', color='r')

ax.legend(loc=1)
ax2.legend(loc=1)

fig.tight_layout()

meta = {'Script': '/Users/amartinez/Desktop/PhD/HAWK/GNS_pm_scripts/GNS_pm_relative_SUPER/SUPER_alignment.py'}

# plt.savefig(f'/Users/amartinez/Desktop/PhD/My_papers/GNS_pm_catalog/images/REL_F1_{field_one}_gaia_resi_pm.png', dpi = 150, transparent = True, metadata = meta)

sys.exit(1381)
# %%
fig, ax = plt.subplots(1,1, figsize = (8,8))
his = ax.hist2d(gns1_mi['H'],(gns1_mi['dpm_x'] + gns1_mi['dpm_y'])/2 , bins = 100,norm = LogNorm())
fig.colorbar(his[3], ax =ax, label = '#stars/bin',fraction = 0.049)
# ax.set_title('GNS1')
ax.set_ylabel('$\overline{\sigma}_{\mu}$ [mas/yr]')
ax.set_xlabel('[H]')
# ax.axis('scaled')
ax.set_xlim(np.min(gns1_mi['H'])- 0.01,np.max(gns1_mi['H'])+ 0.01)

fig.tight_layout()

# meta = {'Script': '/Users/amartinez/Desktop/PhD/HAWK/GNS_pm_scripts/GNS_pm_relative_SUPER/SUPER_alignment.py'}
# plt.savefig('/Users/amartinez/Desktop/PhD/My_papers/GNS_pm_catalog/images/pmError_vs_H.png', dpi = 150, transparent = True, metadata = meta)



# ax.axhline(np.median(gns1_m['dpm_x']),ls = 'dashed', color = 'r', label = 'Median %.2f' %(np.median(gns1_m['dpm_x'])))
# ax.axhline(e_pm_gns, color = 'k', label = f'Max Epm = {e_pm_gns}' )
# ax.legend()


# %%

    

#
# Create the figure and gridspec layout
fig_g = plt.figure(figsize=(4, 4))
gs = fig_g.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0, wspace=0)

# Main scatter plot
axg = fig_g.add_subplot(gs[1, 0])
axg.scatter(d_pmx_ga_m, d_pmy_ga_m, s=200, edgecolor='k', zorder=3)
axg.set_xlabel(r'$\Delta \mu_{l}$ [mas/yr]', fontsize=16)
axg.set_ylabel(r'$\Delta \mu_{b}$ [mas/yr]', fontsize=16)
axg.set_title(f'Gaia matches {len(d_pmy_ga_m)} Max sep = {max_sep_ga}', fontsize=12)
axg.grid()

props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
axg.text(0.05, 0.95,
         '$\sigma_x$ = %.2f mas/yr\n$\sigma_y$ = %.2f mas/yr' % (np.std(d_pmx_ga_m), np.std(d_pmy_ga_m)),
         transform=axg.transAxes, fontsize=12, verticalalignment='top', bbox=props)
axg.tick_params(axis='both', labelsize=12)

# Histogram on the top
ax_histx = fig_g.add_subplot(gs[0, 0], sharex=axg)
ax_histx.hist(d_pmx_ga_m, bins='auto', histtype='step', linewidth=1.5, color='k')
ax_histx.tick_params(axis='x', labelbottom=False)
ax_histx.set_yticks([])
ax_histx.axis('off')

# Histogram on the right
ax_histy = fig_g.add_subplot(gs[1, 1], sharey=axg)
ax_histy.hist(d_pmy_ga_m, bins='auto', orientation='horizontal', histtype='step', linewidth=1.5, color='k')
ax_histy.tick_params(axis='y', labelleft=False)
ax_histy.set_xticks([])
ax_histy.axis('off')

plt.show()

# meta = {'Title': 'v-p.png','script': '/Users/amartinez/Desktop/PhD/HAWK/GNS_pm_scripts/GNS_pm_absolute_SUPER/SUPER_alignment.py'}
# plt.savefig('/Users/amartinez/Desktop/for_people/for_Rainer/v-p_rel.png', transparent=True, bbox_inches = 'tight', metadata = meta)
# %%

pm_gns_p = np.hypot(gns_ga_m['pm_x'], gns_ga_m['pm_y'])
pm_ga_eq = np.hypot(ga_gns_m['pmra'], ga_gns_m['pmdec'])
# pm_ga_eq = np.hypot(ga_gns_m['pm_x'], ga_gns_m['pm_y'])

resid = pm_gns_p - pm_ga_eq
print('')
print('Comoponete are differente, but modules of the vetors invariant under transformations')
print("σ(Δ|μ|):", np.std(resid))
print("mean(Δ|μ|):", np.mean(resid_mu))
print("σ(GNS_μx - Gaia_μra):", np.std(gns_ga_m['pm_x'] - ga_gns_m['pmra']))
print("σ(GNS_μy - Gaia_μdec):",np.std(gns_ga_m['pm_y'] - ga_gns_m['pmdec']))
print("σ(GNS_μx - Gaia_μx):", np.std(gns_ga_m['pm_x'] - ga_gns_m['pm_x']))
print("σ(GNS_μy - Gaia_μy):",np.std(gns_ga_m['pm_y'] - ga_gns_m['pm_y']))
print("σ(GNS_μx - Gaia_μl):", np.std(gns_ga_m['pm_x'] - ga_gns_m['pm_l']))
print("σ(GNS_μy - Gaia_μb):",np.std(gns_ga_m['pm_y'] - ga_gns_m['pm_b']))

# fig, ax = plt.subplots(1,1)
# ax.hist(resid)
# %%

# Create the figure and gridspec layout
fig, ax = plt.subplots(figsize=(6, 6))


# Main scatter plot
cax = ax.scatter(d_pmx_ga_m, d_pmy_ga_m, s=200, edgecolor='k', zorder=1, c = np.sqrt(e_dpmx_m**2 +e_dpmy_m**2), cmap = 'Spectral_r')
cb = fig.colorbar(cax, ax = ax,fraction = 0.0275, aspect = 30)
ax.set_xlabel(r'$\Delta \mu_{x}$ [mas/yr]', fontsize=16)
ax.set_ylabel(r'$\Delta \mu_{y}$ [mas/yr]', fontsize=16)

# ax.set_title(f'Gaia matches {len(d_pmy_ga_m)} Max sep = {max_sep_ga}', fontsize=12)
ax.grid()
ax.axis('scaled')

cb.set_label(' (ℓ, b) []', fontsize=20, labelpad = 20) 

props = dict(boxstyle='round', facecolor='white', alpha=1)
ax.text(0.3, 1.65,
         '$\sigma_x$ = %.2f mas/yr\n$\sigma_y$ = %.2f mas/yr' % (np.std(d_pmx_ga_m), np.std(d_pmy_ga_m)),
         transform=axg.transAxes, fontsize=20, verticalalignment='top', bbox=props)

# sys.exit(832)
# %%


# %%  
num_bins = 8
statistic, x_edges, y_edges, binnumber = stats.binned_statistic_2d(d_pmx_ga_m, d_pmy_ga_m, np.sqrt(e_dpmx_m**2 +e_dpmy_m**2), statistic='median', bins=(num_bins))
# Create a meshgrid for plotting
X, Y = np.meshgrid(x_edges, y_edges)
fig, ax = plt.subplots(figsize=(6, 6))
# c = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r', norm = LogNorm() ) 
cax = ax.pcolormesh(X, Y, statistic.T, cmap='Spectral_r') 
fig.colorbar(cax, ax = ax)
ax.set_xlabel(r'$\Delta \mu_{l}$ [mas/yr]', fontsize=16)
ax.set_ylabel(r'$\Delta \mu_{b}$ [mas/yr]', fontsize=16)
# ax.set_title(f'Gaia matches {len(d_pmy_ga_m)} Max sep = {max_sep_ga}', fontsize=12)
ax.grid()
ax.axis('scaled')
# %%
m_for_g = (gns1_m['H']>gns_mags[0]) & (gns1_m['H']<gns_mags[1])
gns1_m = gns1_m[m_for_g]



radius = abs(np.min(gns1['l'])-np.max(gns1['l']))*0.3*u.degree
# # center_g = SkyCoord(l = np.mean(gns1['l']), b = np.mean(gns1['b']), unit = 'degree', frame = 'galactic')
# center_g = SkyCoord(l = np.mean(gns2['l']), b = np.mean(gns2['b']), unit = 'degree', frame = 'galactic')

try:
    
    gaia = Table.read(pruebas1  + 'gaia_f1%s_f2%s_r%.0f.ecsv'%(field_one,field_two,300))
    # gaia = Table.read(pruebas1  + 'NOgaia_f1%s_f2%s_r%.0f.ecsv'%(field_one,field_two,radius.to(u.arcsec).value))
    print('Gaia from table')
except:
    print('Gaia from web')
    center = SkyCoord(l = np.mean(gns1['l']), b = np.mean(gns1['b']), unit = 'degree', frame = 'galactic').icrs

    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source" # Select early Data Release 3
    Gaia.ROW_LIMIT = -1  # it not especifty, Default rows are limited to 50. 
    # j = Gaia.cone_search_async(center, radius = abs(radius))
    j = Gaia.cone_search_async(center, radius = abs(300*u.arcsec))
    gaia = j.get_results()
    # gaia.write(pruebas1  + 'gaia_f1%s_f2%s_r%.0f.ecsv'%(field_one,field_two,radius.to(u.arcsec).value), overwrite = True)
    gaia.write(pruebas1  + 'gaia_f1%s_f2%s_r%.0f.ecsv'%(field_one,field_two,300), overwrite = True)

# %%


# gaia = filter_gaia_data(
#     gaia_table=gaia,
#     astrometric_params_solved=31,
#     duplicated_source= False,
#     parallax_over_error_min=-10,
#     astrometric_excess_noise_sig_max=2,
#     phot_g_mean_mag_min= gaia_mags[1],
#     phot_g_mean_mag_max=gaia_mags[0],
#     pm_min=0,
#     pmra_error_max=e_pm,
#     pmdec_error_max=e_pm,
#     min_angular_separation_arcsec = 0.5
#     )


gaia = filter_gaia_data(
    gaia_table=gaia,
    astrometric_params_solved=31,
    duplicated_source= False,
    parallax_over_error_min=-10,
    astrometric_excess_noise_sig_max=2,
    phot_g_mean_mag_min= gaia_mags[1],
    phot_g_mean_mag_max=gaia_mags[0],
    pm_min=None,
    pmra_error_max=e_pm_gaia,
    pmdec_error_max=e_pm_gaia,
    ra_error_max = e_pos_gaia,
    dec_error_max = e_pos_gaia,
    min_angular_separation_arcsec = 0.1*u.arcsec
    )

ga_c = SkyCoord(ra = gaia['ra'], dec = gaia['dec'], pm_ra_cosdec = gaia['pmra'],
                 pm_dec = gaia['pmdec'], obstime = 'J2016', 
                 equinox = 'J2000', frame = 'icrs').galactic



gaia['pm_l'] = ga_c.pm_l_cosb 
gaia['pm_b'] = ga_c.pm_b



# Comaparing Gaia with itself

dga_ra = gaia['pmra'] - gaia['pm_l']
dga_dec = gaia['pmdec'] - gaia['pm_b']
print(np.mean(dga_ra),np.std(dga_ra))
print(np.mean(dga_dec),np.std(dga_dec))
m_ga, lga = sig_f(dga_ra, dga_dec ,3)

dga_ram = dga_ra[m_ga]
dga_decm = dga_dec[m_ga]

fig, (ax,ax2) = plt.subplots(1,2)

fig.suptitle(f'Gaia Ecu vs Gaia Gal. Stars =  {len(gaia)} ')
ax.hist(dga_ra, bins = bins, color = 'k', alpha = 0.2)
ax2.hist(dga_dec, bins = bins,color = 'k', alpha = 0.2)
ax.hist(dga_ram, bins = bins, histtype = 'step', label = '$\overline{\mu}_{Ra-l}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(dga_ram.value),np.std(dga_ram.value)))
ax2.hist(dga_decm, bins = bins, histtype = 'step',label = '$\overline{\mu}_{Dec-b}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(dga_decm.value),np.std(dga_decm.value)))
ax.set_xlabel('$\Delta \mu_{RA-l}$ [mas/yr]')
ax2.set_xlabel('$\Delta\mu_{Dec-b}$ [mas/yr]')
ax.axvline(lga[0], ls = 'dashed', color = 'r')
ax.axvline(lga[1], ls = 'dashed', color = 'r')
ax2.axvline(lga[2], ls = 'dashed', color = 'r')
ax2.axvline(lga[3], ls = 'dashed', color = 'r')
ax.legend()
ax2.legend()




# %%
ga_c = SkyCoord(ra=gaia['ra'], dec=gaia['dec'],
                pm_ra_cosdec=gaia['pmra'],
                pm_dec=gaia['pmdec'],
                frame='icrs', obstime='J2016')
ga_gal = ga_c.galactic

ga_icrs_back = ga_gal.transform_to('icrs')
# Residuals
dga_ra = (ga_icrs_back.pm_ra_cosdec - ga_c.pm_ra_cosdec)
dga_dec = (ga_icrs_back.pm_dec - ga_c.pm_dec)

m_ga, lga = sig_f(dga_ra, dga_dec ,3)

dga_ram = dga_ra[m_ga]  
dga_decm = dga_dec[m_ga]

fig, (ax,ax2) = plt.subplots(1,2)

fig.suptitle(f'Gaia Ecu vs Gaia Gal to ICRS. Stars =  {len(gaia)} ')
ax.hist(dga_ra, bins = bins, color = 'k', alpha = 0.2)
ax2.hist(dga_dec, bins = bins,color = 'k', alpha = 0.2)
ax.hist(dga_ram, bins = bins, histtype = 'step', label = '$\overline{\mu}_{Ra}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(dga_ram.value),np.std(dga_ram.value)))
ax2.hist(dga_decm, bins = bins, histtype = 'step',label = '$\overline{\mu}_{Dec}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(dga_decm.value),np.std(dga_decm.value)))
ax.set_xlabel('$\Delta \mu_{RA}$ [mas/yr]')
ax2.set_xlabel('$\Delta\mu_{Dec}$ [mas/yr]')
ax.axvline(lga[0].value, ls = 'dashed', color = 'r')
ax.axvline(lga[1].value, ls = 'dashed', color = 'r')
ax2.axvline(lga[2].value, ls = 'dashed', color = 'r')
ax2.axvline(lga[3].value, ls = 'dashed', color = 'r')
ax.legend()
ax2.legend()
# %%














