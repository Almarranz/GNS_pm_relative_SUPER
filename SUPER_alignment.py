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

import pandas as pd

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
from alignator_looping import alg_loop

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
field_one = 60
chip_one = 0
field_two = 20
chip_two = 0


if field_one == 7 or field_one == 12 or field_one == 10 or field_one == 16:
    t1 = Time(['2015-06-07T00:00:00'],scale='utc')
elif field_one == 60:
    t1 = Time(['2016-06-13T00:00:00'],scale='utc')
elif field_one ==  100:
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
pix_scale = 0.1064*0.5
# pix_scale = 0.1064
# max_sig = 0.3#TODO
e_pm = 0.3#!!! Maximun error in pm for Gaia stars
gaia_mags = [0, 100]#!!! Gaia mag limtis for comparison with GNS
max_sep_ga = 50*u.mas# separation for comparison with gaia

max_loop = 10
gns_mags = [0,20]#!!! GNS mag limtis for comparison with Gaia
max_sig = 0.005
# max_sig = 2
# use_grid = 'yes'
use_grid = 'no'
grid_s = 700
bad_lim = -0.7#!!! weird offset in Delat H between GNS 1 and 2
# use_grid = 'no'
max_sep = 50* u.mas# firts match gns1 to gns2 for astroaling
sig_cl = 3#!!!
max_deg =5
d_m = 80*u.mas#!!!in arcse, max distance for the fine alignment betwenn GNS1 and 2
d_m_pm = 0.300#!!! in arcs, max distance for the proper motions
destination = 1 #!!! GNS1 is reference
# destination = 2 #!!! GNS2 is reference
align_by = 'Polywarp'#!!!
# align_by = '2DPoly'#!!!
f_mode = 'W' # f_mode only useful for 2Dpoly
# f_mode = 'WnC'
# f_mode = 'NW'
# f_mode = 'NWnC'

def sig_f(x, y,s):
    mx, lx, hx = sigma_clip(x , sigma = s, masked = True, return_bounds= True)
    my, ly, hy = sigma_clip(y , sigma = s, masked = True, return_bounds= True)
    m_xy = np.logical_and(np.logical_not(mx.mask),np.logical_not(my.mask))
    
    return m_xy, [lx,hx,ly,hy]
# %% 
GNS_1='/Users/amartinez/Desktop/PhD/HAWK/GNS_1/lists/%s/chip%s/'%(field_one, chip_one)
GNS_2='/Users/amartinez/Desktop/PhD/HAWK/GNS_2/lists/%s/chip%s/'%(field_two, chip_two)

pruebas1 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1absolute_SUPER/pruebas/'
pruebas2 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2absolute_SUPER/pruebas/'


gns1 = Table.read(GNS_1 + 'stars_calibrated_H_chip%s.ecsv'%(chip_one),  format = 'ascii.ecsv')
gns2 = Table.read(GNS_2 + 'stars_calibrated_H_chip%s.ecsv'%(chip_two), format = 'ascii.ecsv')

# m_mask = (gns1['H']>14) & (gns1['H']<17)
# gns1 = gns1[m_mask]

unc_cut1 = (gns1['sl']<max_sig) & (gns1['sb']<max_sig)
gns1 = gns1[unc_cut1]

unc_cut2 = (gns2['sl']<max_sig) & (gns2['sb']<max_sig)
gns2 = gns2[unc_cut2]


buenos1 = (gns1['l']>min(gns2['l'])) & (gns1['l']<max(gns2['l'])) & (gns1['b']>min(gns2['b'])) & (gns1['b']<max(gns2['b']))

gns1 = gns1[buenos1]
gns1['ID'] = np.arange(len(gns1))

buenos2 = (gns2['l']>min(gns1['l'])) & (gns2['l']<max(gns1['l'])) & (gns2['b']>min(gns1['b'])) & (gns2['b']<max(gns1['b']))

gns2 = gns2[buenos2]
gns2['ID'] = np.arange(len(gns2))

# center = SkyCoord(l = np.mean(gns1['l']), b = np.mean(gns1['b']), unit = 'degree', frame = 'galactic')
center_1 = SkyCoord(l = np.mean(gns1['l']), b = np.mean(gns1['b']), unit = 'degree', frame = 'galactic')
center_2 = SkyCoord(l = np.mean(gns2['l']), b = np.mean(gns2['b']), unit = 'degree', frame = 'galactic')

gns1_lb = SkyCoord(l = gns1['l'], b = gns1['b'], unit ='deg', frame = 'galactic')
gns2_lb = SkyCoord(l = gns2['l'], b = gns2['b'], unit ='deg', frame = 'galactic')

xg_1, yg_1 = center_1.spherical_offsets_to(gns1_lb)
xg_2, yg_2 = center_2.spherical_offsets_to(gns2_lb)

# %%
#
gns1['xp'] = xg_1.to(u.arcsec)
gns1['yp'] = yg_1.to(u.arcsec)
gns2['xp'] = xg_2.to(u.arcsec)
gns2['yp'] = yg_2.to(u.arcsec)

#I cosider a math if the stars are less than 'max_sep' arcsec away 
# This is for cutting the the overlapping areas of both lists. (Makes astroaling work faster)

idx,d2d,d3d = gns1_lb.match_to_catalog_sky(gns2_lb)# ,nthneighbor=1 is for 1-to-1 match
sep_constraint = d2d < max_sep
gns1_match = gns1[sep_constraint]
gns2_match = gns2[idx[sep_constraint]]

diff_H = gns2_match['H']-gns1_match['H']
# off_s = np.mean(diff_H)
diff_H = gns2_match['H'] - gns1_match['H']

diff_H = gns2_match['H']-gns1_match['H']
sig_cl = 3
mask_H, l_lim,h_lim = sigma_clip(diff_H, sigma=sig_cl, masked = True, return_bounds= True, maxiters= 50)

# gns2_match = gns2_match[mask_H.mask]
# gns1_match = gns1_match[mask_H.mask]


bad_H = diff_H <bad_lim
fig,ax = plt.subplots(1,1)
ax.hist(diff_H, bins = 'auto',histtype = 'step')
ax.axvline(bad_lim, ls = 'dashed', color = 'k', label = 'Malas?')
ax.axvline(np.mean(diff_H), color = 'k', ls = 'dashed', label = '$\overline{\Delta H}$= %.2f$\pm$%.2f'%(np.mean(diff_H),np.std(diff_H)))
ax.axvline(l_lim, ls = 'dashed', color ='r', label ='%s$\sigma$'%(sig_cl))

ax.axvline(h_lim, ls = 'dashed', color ='r')
ax.set_xlabel('$\Delta H$')
ax.legend() 



fig,ax = plt.subplots()
ax.scatter(gns1_match['l'],gns1_match['b'])
ax.scatter(gns1_match['l'][bad_H],gns1_match['b'][bad_H], label = 'Malas')
ax.legend()
sys.exit(212)
# %%
g1_m = np.array([gns1_match['xp'],gns1_match['yp']]).T
g2_m = np.array([gns2_match['xp'],gns2_match['yp']]).T

if destination == 1:
    # Time lapse to move Gaia Stars.
    tg = Time(['2016-01-01T00:00:00'],scale='utc')
    dtg = t1 - tg
    
    p,(_,_)= aa.find_transform(g2_m,g1_m,max_control_points=100)
    
    print("Translation: (x, y) = (%.2f, %.2f)"%(p.translation[0],p.translation[1]))
    print("Rotation: %.2f deg"%(p.rotation * 180.0/np.pi)) 
    print("Rotation: %.0f arcmin"%(p.rotation * 180.0/np.pi*60)) 
    print("Rotation: %.0f arcsec"%(p.rotation * 180.0/np.pi*3600)) 
    
    
    # %%
    
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
    gns2 = alg_loop(gns2, gns1, 'xp', 'yp', align_by, max_deg, d_m.to(u.arcsec).value, max_loop, use_grid ='no')

if destination == 2:
    # Time lapse to move Gaia Stars.
    tg = Time(['2016-01-01T00:00:00'],scale='utc')
    dtg = t2 - tg
    
    p,(_,_)= aa.find_transform(g1_m,g2_m,max_control_points=100)
    
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
    gns1 = alg_loop(gns1, gns2, 'xp', 'yp', align_by, max_deg, d_m.to(u.arcsec).value, max_loop, use_grid ='no')

# sys.exit(202) 
# %%
l1_xy = np.array([gns1['xp'],gns1['yp']]).T
l2_xy = np.array([gns2['xp'],gns2['yp']]).T
l_12 = compare_lists(l1_xy,l2_xy,d_m_pm)


print(30*'*'+'\nComon stars to be use for pm calculation :%s\n'%(len(l_12))+30*'*')
gns1_m = gns1[l_12['ind_1']]
gns2_m = gns2[l_12['ind_2']]



dx = (gns2_m['xp'].value- gns1_m['xp'].value)*1000
dy = (gns2_m['yp'].value- gns1_m['yp'].value)*1000

pm_x = (dx*u.mas)/dt.to(u.year)
pm_y = (dy*u.mas)/dt.to(u.year)
# pm_l = (dl*mean_b)/dt.to(u.year)
# pm_b = (db)/dt.to(u.year)

sig_dis = 3

m_pm, lims = sig_f(pm_x, pm_x, sig_dis)


pm_xm = pm_x[m_pm]
pm_ym = pm_y[m_pm]

gns1_m = gns1_m[m_pm]
gns2_m = gns2_m[m_pm]

gns1_m['pm_x']  = pm_xm
gns1_m['pm_y']  = pm_ym

gns2_m['pm_x']  = pm_xm
gns2_m['pm_y']  = pm_ym


# %%
bins = 'auto'
fig, (ax,ax2) = plt.subplots(1,2)
ax.set_title(f'Ref. = GNS{destination}. Degree = {max_deg-1}', fontsize= 15)

ax.hist(pm_x, bins = bins, color = 'k', alpha = 0.2)
ax2.hist(pm_y, bins = bins,color = 'k', alpha = 0.2)
ax.hist(pm_xm, bins = bins, histtype = 'step', label = '$\overline{\mu}_{x}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(pm_xm.value),np.std(pm_xm.value)),)
ax2.hist(pm_ym, bins = bins, histtype = 'step',label = '$\overline{\mu}_{y}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(pm_ym.value),np.std(pm_ym.value)))
ax.set_xlabel('$\Delta \mu_{x}$ [mas/yr]')
ax2.set_xlabel('$\Delta\mu_{y}$ [mas/yr]')
ax.axvline(lims[0].value , ls = 'dashed', color = 'r', label = f'{sig_dis}$\sigma$')
ax.axvline(lims[1].value , ls = 'dashed', color = 'r')
ax2.axvline(lims[2].value , ls = 'dashed', color = 'r')
ax2.axvline(lims[3].value , ls = 'dashed', color = 'r')
ax.set_xlim(-20,20)
ax2.set_xlim(-20,20)

# ax.invert_xaxis()
ax.legend( fontsize = 15)
ax2.legend(fontsize = 15)

# %% Gaia Comparation



# %
# Before comparing witg Gaia we mask the best pms

m_for_g = (gns1_m['H']>gns_mags[0]) & (gns1_m['H']<gns_mags[1])
gns1_m = gns1_m[m_for_g]



radius = abs(np.min(gns1['l'])-np.max(gns1['l']))*0.6*u.degree
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
    pmdec_error_max=e_pm
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
# %

ga_l = ga_c.l.wrap_at('360.02d')
ga_b = ga_c.b.wrap_at('180d')

gns2_gal = SkyCoord(l = gns2_m['l'], b = gns2_m['b'], 
                    unit = 'degree', frame = 'galactic')
gns1_gal = SkyCoord(l = gns1_m['l'], b = gns1_m['b'], 
                    unit = 'degree', frame = 'galactic')

gaia_c = SkyCoord(l = gaia['l'], b = gaia['b'], frame = 'galactic')

if destination == 1:
    idx,d2d,d3d = gns1_gal.match_to_catalog_sky(gaia_c)
    sep_constraint = d2d < max_sep_ga
    gns_ga = gns1_m[sep_constraint]
    ga_gns = gaia[idx[sep_constraint]]# ,nthneighbor=1 is for 1-to-1 match
elif destination ==2:
    idx,d2d,d3d = gns2_gal.match_to_catalog_sky(gaia_c)
    sep_constraint = d2d < max_sep_ga
    gns_ga = gns2_m[sep_constraint]
    ga_gns = gaia[idx[sep_constraint]]# ,nthneighbor=1 is for 1-to-1 match



l2 = gns2_gal.l.wrap_at('360d')
l1 = gns1_gal.l.wrap_at('360d')


fig, ax = plt.subplots(1,1,figsize =(10,10))
ax.scatter(l1[::1], gns1_m['b'][::1],label = 'GNS_1 Fied %s, chip %s'%(field_one,chip_one))
# ax.scatter(l2[::10],  gns2_m['b'][::10], label = 'GNS_2 Fied %s, chip %s'%(field_two,chip_two))
ax.scatter(ga_l,gaia['b'], color = 'k',label = f'Gaia stars = {len(gaia)}')
ax.scatter(ga_gns['l'],ga_gns['b'], color = 'r',s =100,label = f'Gaia comp pm = {len(ga_gns)}')
ax.invert_xaxis()
ax.legend()
ax.set_xlabel('l[ deg]', fontsize = 10)
ax.set_ylabel('b [deg]', fontsize = 10)
# ax.axis('scaled')
ax.set_ylim(min(gaia['b']),max(gaia['b']))



d_pmx_ga = gns_ga['pm_x'] - ga_gns['pm_l']

d_pmy_ga = gns_ga['pm_y'] - ga_gns['pm_b']


m_pm, lxy = sig_f(d_pmx_ga, d_pmy_ga, 3)


d_pmx_ga_m = d_pmx_ga[m_pm]
d_pmy_ga_m = d_pmy_ga[m_pm]



fig, (ax,ax2) = plt.subplots(1,2)
ax.set_title(f'Gaia stars = {len(d_pmx_ga_m)}, Degree = {max_deg-1}', fontsize= 15)
ax2.set_title(f'Ref. epoch GNS{destination}', fontsize = 15)
ax.hist(d_pmx_ga, bins = bins, color = 'k', alpha = 0.2)
ax2.hist(d_pmy_ga, bins = bins,color = 'k', alpha = 0.2)
ax.hist(d_pmx_ga_m, bins = bins, histtype = 'step', label = '$\overline{\mu}_{x}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(d_pmx_ga_m.value),np.std(d_pmx_ga_m.value)))
ax2.hist(d_pmy_ga_m, bins = bins, histtype = 'step',label = '$\overline{\mu}_{y}$ = %.2f\n$\sigma$ =%.2f'%(np.mean(d_pmy_ga_m.value),np.std(d_pmy_ga_m.value)))
ax.set_xlabel('$\Delta \mu_{x}$ [mas/yr]')
ax2.set_xlabel('$\Delta\mu_{y}$ [mas/yr]')
ax.axvline(lxy[0], ls = 'dashed', color = 'r')
ax.axvline(lxy[1], ls = 'dashed', color = 'r')
ax2.axvline(lxy[2], ls = 'dashed', color = 'r')
ax2.axvline(lxy[3], ls = 'dashed', color = 'r')
ax.legend()
ax2.legend()


# %%

zone =  'F20_f01_f06_H'

scamp_f = '/Users/amartinez/Desktop/Projects/GNS_gd/scamp/GNS0/%s/'%(zone)

try:
    cat = Table.read(scamp_f +f'merged_{zone}_1.ocat', format = 'ascii') 
except:
    cat = Table.read(scamp_f +f'merged_{zone}.ocat', format = 'ascii') 
vel_max = 50
pm_mask = (abs(cat['PMALPHA_J2000']) <vel_max) &  (abs(cat['PMDELTA_J2000']) <vel_max) & (abs(cat['PMALPHA_J2000']) >1e-5) &  (abs(cat['PMDELTA_J2000']) >1e-5)
cat = cat[pm_mask]

cat_c = SkyCoord(ra = cat['ALPHA_J2000'], dec = cat['DELTA_J2000'], frame = 'fk5').galactic

if destination == 1:
    
    gns_c = SkyCoord(l = gns1_m['l'], b = gns1_m['b'], frame = 'galactic')

    max_sep = 20*u.mas
    idx, d2d, _ = cat_c.match_to_catalog_sky(gns_c, nthneighbor=1)
    match_mask = d2d < max_sep
    cat_m = cat[match_mask]
    gns_mm = gns1_m[idx[match_mask]]

if destination == 2:
    
    gns_c = SkyCoord(l = gns2_m['l'], b = gns2_m['b'], frame = 'galactic')

    max_sep = 50*u.mas
    idx, d2d, _ = cat_c.match_to_catalog_sky(gns_c, nthneighbor=1)
    match_mask = d2d < max_sep
    cat_m = cat[match_mask]
    gns_mm = gns2_m[idx[match_mask]]




# %
fig, ax = plt.subplots(1,1)
ax.scatter(cat_c.l,cat_c.b)
ax.scatter(cat_c.l[match_mask],cat_c.b[match_mask])


dpmx = (gns_mm['pm_x'] - cat_m['PMALPHA_J2000'])
dpmy = (gns_mm['pm_y'] - cat_m['PMDELTA_J2000'])

m_pm, lim = sig_f(dpmx, dpmy, 3) 

dpmx_m = dpmx[m_pm]
dpmy_m = dpmy[m_pm]


fig, ax = plt.subplots(1,1)

ax.scatter(dpmx_m,dpmy_m, color = 'k', alpha = 0.3)
ax.axvline(lims[0].value, color = 'r', ls = 'dashed', label = f'{3}$\sigma$')
ax.scatter(dpmx_m,dpmy_m,edgecolor = 'k', label = '$\overline{\Delta \mu_{x}}$ = %.2f, $\sigma$ = %.2f\n''$\overline{\Delta \mu_{y}}$ = %.2f, $\sigma$ = %.2f'%(np.mean(dpmx_m),np.std(dpmx_m),np.mean(dpmy_m),np.std(dpmy_m)))
ax.axvline(lims[1].value, color = 'r', ls = 'dashed')
ax.axhline(lims[2].value, color = 'r', ls = 'dashed')
ax.axhline(lims[3].value, color = 'r', ls = 'dashed')
ax.legend()









