#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:44:21 2024

@author: amartinez
"""
import numpy as np
from compare_lists import compare_lists
import Polywarp as pw
import matplotlib.pyplot as plt
import sys
from astropy.stats import sigma_clip
from astropy.modeling.models import Polynomial2D
from astropy.modeling.fitting import LinearLSQFitter
from astropy.modeling import models, fitting
from grid import grid_stars
"""
 Iteratively aling gns_A (source) to gns_B (destination)
 Parameters:
 -----------
 table : astropy.table.Table
     Input table containing star data.
 gns_A : table 
     Astropy Table, or similar 
 gns_B : table 
     Astropy Table, or similar 
 align_bu : str
     'Polywarp' or '2DPolt'
 
 Returns:
 --------
 astropy.table.Table
     aligned gns_A table
 """

def alg_rel(gns_A, gns_B, align_by,use_grid,max_deg,d_m,grid_s = None, f_mode = None  ) :
    loop = 0
    deg = 1
    loop = 0
    sig_cl = 3
    comom_ls = []
    dic_xy = {}
    dic_Kx ={}
    dic_xy_final = {}
    if use_grid == 'yes':
        gns_g = gns_B
        # def grid_stars(table, x_col, y_col, mag_col, mag_min, mag_max, grid_size=50, isolation_radius=0.5):
        gns_g = grid_stars(gns_g,'x','y','H',12,18,grid_size=100,isolation_radius=0.7)
        fig, ax = plt.subplots(1,1)
        ax.scatter(gns_A['x'], gns_A['y'])
        ax.scatter(gns_g['x'], gns_g['y'],s =1)
        l2_xy = np.array([gns_g['x'],gns_g['y']]).T
    else:
        l2_xy = np.array([gns_B['x'],gns_B['y']]).T

    while deg < max_deg:
        loop += 1 
        l1_xy = np.array([gns_A['x'],gns_A['y']]).T
        
        comp = compare_lists(l1_xy,l2_xy,d_m)
        if len(comom_ls) >1:
            # if comom_ls[-1] <= comom_ls[-2]:
            if comom_ls[-1] <= comom_ls[-2]:
                try:
                    gns_A['x'] = dic_xy[f'trans_{loop-2}'][:,0]
                    gns_A['y'] = dic_xy[f'trans_{loop-2}'][:,1]
                    dic_xy_final['xy_deg%s'%(deg)] = np.array([dic_xy[f'trans_{loop-2}'][:,0],dic_xy[f'trans_{loop-2}'][:,1]]).T            
                    comom_ls =[]
                    dic_xy = {}
                    dic_Kx = {}
                    deg += 1
                    print(f'Number of common star in loop {loop-1} lower tha in loop {loop-2}.\nJupping to degree {deg} ')
                    loop = -1
                    continue
                except:
                    gns_A['x'] = dic_xy_final[f'xy_deg{deg-1}'][:,0]
                    gns_A['y'] = dic_xy_final[f'xy_deg{deg-1}'][:,1]
                    print(f'Number of common star with polynomial degere {deg} decreases after a single iteration.\nUsing the last iteration of degree {deg -1} ')
                    deg = deg
                    break
                
        comom_ls.append(len(comp))
        print(f'Common in loop {loop}, degree {deg} = %s'%(len(comp['ind_1'])))
        # if loop == 1:
        #     with open(pruebas + 'sig_and_com.txt', 'a') as file:
        #         file.write('%.1f %.0f\n'%(sig_cl, len(comp['ind_1'])))

        l1_com = gns_A[comp['ind_1']]
        
        if use_grid == 'yes':
            l2_com = gns_g[comp['ind_2']]
        else: 
            l2_com = gns_B[comp['ind_2']]
        
        diff_mag = l1_com['H'] - l2_com['H'] 
        # diff_mag1 = l1_com['IB230_diff'] - l2_com['IB230_diff'] 
        diff_x =  l2_com['x'] - l1_com['x'] 
        diff_y =  l2_com['y'] - l1_com['y'] 
        diff_xy = (diff_x**2 + diff_y**2)**0.5
        mask_m, l_lim,h_lim = sigma_clip(diff_mag, sigma=sig_cl, masked = True, return_bounds= True)
        
        
        
        # l2_clip = l2_com
        # l1_clip = l1_com
        
        l1_clip = l1_com[np.logical_not(mask_m.mask)]
        l2_clip = l2_com[np.logical_not(mask_m.mask)]
        
    # =============================================================================
    #         fig, (ax,ax1) = plt.subplots(1,2)
    #         fig.suptitle(f'Degree = {deg}. Loop = {loop}')
    #         ax.set_xlabel('$\Delta$ H')
    #         ax.hist(diff_mag, label = 'matches = %s\ndist = %.2f arcsec'%(len(comp['ind_1']), d_m*pix_scale))
    #         ax.axvline(l_lim, color = 'red', ls = 'dashed', label = '$\pm$%s$\sigma$'%(sig_cl))
    #         ax.axvline(h_lim, color = 'red', ls = 'dashed')
    #         ax.legend()
    #         
    #         ax1.hist(diff_x, label = '$\overline{\Delta x} = %.2f\pm%.2f$'%(np.mean(diff_x),np.std(diff_x)), histtype = 'step')
    #         ax1.hist(diff_y, label = '$\overline{\Delta y} = %.2f\pm%.2f$'%(np.mean(diff_y),np.std(diff_y)), histtype = 'step')
    #     
    #         ax1.set_xlabel('$\Delta$ pixel')
    #         ax1.legend()
    # =============================================================================
        
        
        
        
        xy_1c = np.array([l1_clip['x'],l1_clip['y']]).T
        xy_2c = np.array([l2_clip['x'],l2_clip['y']]).T
        
        if align_by == 'Polywarp':
            Kx,Ky=pw.polywarp(xy_2c[:,0],xy_2c[:,1],xy_1c[:,0],xy_1c[:,1],degree=deg)
            
            xi=np.zeros(len(gns_A))
            yi=np.zeros(len(gns_A))
            
            for k in range(deg+1):
                        for m in range(deg+1):
                            xi=xi+Kx[k,m]*gns_A['x']**k*gns_A['y']**m
                            yi=yi+Ky[k,m]*gns_A['x']**k*gns_A['y']**m
        elif align_by == '2DPoly':
            model_x = Polynomial2D(degree=deg)
            model_y = Polynomial2D(degree=deg)
            
            # Linear least-squares fitter
            fitter = LinearLSQFitter()
            # fitter = fitting.LMLSQFitter()
            
            if f_mode == 'W':
                #==============
                # Fit weighted
                #==============
                fit_xw = fitter(model_x, xy_1c[:,0],xy_1c[:,1], xy_2c[:,0], weights= 1/np.sqrt(l1_clip['dx1']**2 + l1_clip['dy1']**2))  # Fit x-coordinates
                fit_yw = fitter(model_y, xy_1c[:,0],xy_1c[:,1],  xy_2c[:,1],weights= 1/np.sqrt(l1_clip['dx1']**2 + l1_clip['dy1']**2)) 
            
            elif f_mode == 'NW':
                #==============
                # Fit not weighted
                #==============
                fit_xw = fitter(model_x, xy_1c[:,0],xy_1c[:,1], xy_2c[:,0])  # Fit x-coordinates
                fit_yw = fitter(model_y, xy_1c[:,0],xy_1c[:,1],  xy_2c[:,1]) 
            
            elif f_mode == 'WnC':
                #==============
                #Fit weighted and clipped
                #==============
                or_fit = fitting.FittingWithOutlierRemoval(fitter, sigma_clip, niter=3, sigma=3.0)
                
                fit_xw, fm_x = or_fit(model_x, xy_1c[:,0],xy_1c[:,1], xy_2c[:,0], weights= 1/np.sqrt(l1_clip['dx1']**2 + l1_clip['dy1']**2))  # Fit x-coordinates
                fit_yw, fm_y = or_fit(model_y, xy_1c[:,0],xy_1c[:,1],  xy_2c[:,1],weights= 1/np.sqrt(l1_clip['dx1']**2 + l1_clip['dy1']**2)) 
                
                fm_xy = np.logical_not(np.logical_and(np.logical_not(fm_x),np.logical_not(fm_y)))
                
                fig, ax = plt.subplots(1,1)
                plt_per = 20
                ax.set_title(f'Degree = {deg}')
                ax.scatter(xy_1c[:,0][::plt_per],xy_1c[:,1][::plt_per], label = 'Used (%s%%)'%(plt_per))
                ax.scatter(xy_1c[:,0][fm_xy],xy_1c[:,1][fm_xy],s =200, label = 'Clipped')
                ax.legend()
            elif f_mode == 'NWnC':
                or_fit = fitting.FittingWithOutlierRemoval(fitter, sigma_clip, niter=3, sigma=3.0)
                
                fit_xw, fm_x = or_fit(model_x, xy_1c[:,0],xy_1c[:,1], xy_2c[:,0])  # Fit x-coordinates
                fit_yw, fm_y = or_fit(model_y, xy_1c[:,0],xy_1c[:,1],  xy_2c[:,1]) 
                
                fm_xy = np.logical_not(np.logical_and(np.logical_not(fm_x),np.logical_not(fm_y)))
                
                fig, ax = plt.subplots(1,1)
                plt_per = 20
                ax.set_title(f'Degree = {deg}')
                ax.scatter(xy_1c[:,0][::plt_per],xy_1c[:,1][::plt_per], label = 'Used (%s%%)'%(plt_per))
                ax.scatter(xy_1c[:,0][fm_xy],xy_1c[:,1][fm_xy],s =200, label = 'Clipped')
                ax.legend()
                
            # sys.exit(308)
            
            xi = fit_xw(gns_A['x'], gns_A['y'])
            yi = fit_yw(gns_A['x'], gns_A['y'])# Fit y-coordinates
            
        dic_xy[f'trans_{loop+1}'] = np.array([xi,yi]).T
        
        # print(Kx[0][0])
        gns_A['x'] = xi
        gns_A['y'] = yi
        
        if use_grid == 'yes':
            check_2 = np.array([gns_A['x'],gns_A['y']]).T
            check_1 = np.array([gns_B['x'],gns_B['y']]).T
            check = compare_lists(check_2, check_1, d_m)
            print(30*'-'+'\nTotal common stars (not only grid):%s\n'%(len(check))+30*'-')
    
    gns1 = gns_A

    return gns1
# =============================================================================
# def alignator(survey,gns,gaia,s_ls, d_m,max_deg, align_by = None,f_mode = None, plot = None, clipping = None, sig_clip = None):
#     # Proceed with the iterative alignment process from an initial common lists
#     
#     # id1 = f'x{survey}'
#     # id2 = f'y{survey}'
#     # dx = f'dx{survey}'
#     # dy = f'dy{survey}'
#     id1 = 'x'
#     id2 = 'y'
#     dx = 'sl'
#     dy = 'sb‡'
#     loop =1
#     deg = 1
# 
#     comom_ls =[]
#     comom_ls.append(len(s_ls))
#     dic_xy = {} 
#     dic_xy['trans_0'] = np.array([gns[id1],gns[id2]]).T
#     # for loop in range(1,10):
#     if sig_clip == None:
#         sig_cl = 3#!!!
#     elif sig_clip is not None:
#         sig_cl = sig_clip
#     
#     while deg < max_deg:
#         
#         diff_x =s_ls['l2_x'] - s_ls['l1_x']  
#         diff_y =s_ls['l2_y'] - s_ls['l1_y']  
#         
#         mask_x, lx_lim,hx_lim = sigma_clip(diff_x, sigma=sig_cl, masked = True, return_bounds= True)
#         mask_y, ly_lim,hy_lim = sigma_clip(diff_y, sigma=sig_cl, masked = True, return_bounds= True)
#         
#         # mask_xy = mask_x & mask_y # mask_xy = np.logical(mx, my)
#         mask_xy = np.logical_and(np.logical_not(mask_x.mask), np.logical_not(mask_y.mask))
#         
#         
#         
#         if plot == 'yes':
#             fig, (ax,ax1)  = plt.subplots(1,2)
#             ax1.set_title(f'GNS{survey}. Degree = %s'%(deg))
#             ax.set_title(f'Loop = {loop-1}. Matching stars = {len(s_ls)}')
#             ax.hist(diff_x, histtype = 'step', label = '$\overline{x} = %.2f$\n$\sigma x$ =%.2f'%(np.mean(diff_x),np.std(diff_x)))
#             ax.axvline(lx_lim, color = 'red', ls = 'dashed', label = '$\pm$%s$\sigma$'%(sig_cl))
#             ax.axvline(hx_lim, color = 'red', ls = 'dashed')
#             ax1.axvline(ly_lim, color = 'red', ls = 'dashed')
#             ax1.axvline(hy_lim, color = 'red', ls = 'dashed')
#            
#             ax1.hist(diff_y, histtype = 'step', label = '$\overline{y} = %.2f$\n$\sigma y$ =%.2f'%(np.mean(diff_y),np.std(diff_y)))
#             
#             ax.set_xlabel('$\Delta x$ [mas]')
#             ax1.set_xlabel('$\Delta y$ [mas]')
#            
#             if np.all(mask_xy) == False:
#                 
#                 diff_mx = diff_x[np.logical_not(mask_x.mask)]
#                 diff_my = diff_y[np.logical_not(mask_y.mask)]
#                 ax.hist(diff_mx , color = 'k', alpha = 0.5, lw = 10,label = '$\overline{x} = %.2f$\n$\sigma x$ =%.2f'%(np.mean(diff_mx),np.std(diff_mx)))
#                 ax1.hist(diff_my, color = 'k', alpha = 0.5, lw = 10,label = '$\overline{y} = %.2f$\n$\sigma y$ =%.2f'%(np.mean(diff_my),np.std(diff_my)))
#             ax.legend()
#             ax1.legend()
#         if clipping is not None:
#             if np.all(mask_xy) == False:
#                 # This line elimates the 3sigma alignment stars 
#                 s_ls = s_ls[mask_xy]
#                 print(10*'💀' + f'\nThere are {sig_cl} \u03C3 alignmet stars\n' + 10*'💀')
#                     
#     
#         if align_by == 'Polywarp':
#             Kx,Ky=pw.polywarp(s_ls['l2_x'],s_ls['l2_y'],s_ls['l1_x'],s_ls['l1_y'],degree=deg)
#             
#             xi=np.zeros(len(gns))
#             yi=np.zeros(len(gns))
#             
#             for k in range(deg+1):
#                         for m in range(deg+1):
#                             xi=xi+Kx[k,m]*gns[id1]**k*gns[id2]**m
#                             yi=yi+Ky[k,m]*gns[id1]**k*gns[id2]**m
#             
#             # dic_xy[f'trans_{loop}'] = np.array([xi,yi]).T    
#         elif align_by == '2DPoly':
#             model_x = Polynomial2D(degree=deg)
#             model_y = Polynomial2D(degree=deg)
#             
#             # Linear least-squares fitter
#             fitter = LinearLSQFitter()
#             # fitter = fitting.LMLSQFitter()
#             
#             if f_mode == 'W':
#                 #==============
#                 # Fit weighted
#                 #==============
#                 fit_xw = fitter(model_x, s_ls['l1_x'],s_ls['l1_y'], s_ls['l2_x'], weights= 1/np.sqrt(gns[dx][s_ls['ind_1']]**2 + gns[dy][s_ls['ind_1']]**2))  # Fit x-coordinates
#                 fit_yw = fitter(model_y, s_ls['l1_x'],s_ls['l1_y'], s_ls['l2_y'],weights= 1/np.sqrt(gns[dx][s_ls['ind_1']]**2 + gns[dy][s_ls['ind_1']]**2)) 
#             
#             elif f_mode == 'NW':
#                 #==============
#                 # Fit not weighted
#                 #==============
#                 fit_xw = fitter(model_x, s_ls['l1_x'],s_ls['l1_y'], s_ls['l2_x']) # Fit x-coordinates
#                 fit_yw = fitter(model_y, s_ls['l1_x'],s_ls['l1_y'], s_ls['l2_y']) 
#             
#             elif f_mode == 'WnC':
#                 #==============
#                 #Fit weighted and clipped
#                 #==============
#                 or_fit = fitting.FittingWithOutlierRemoval(fitter, sigma_clip, niter=3, sigma=3.0)
#                 
#                 fit_xw, fm_x = or_fit(model_x, s_ls['l1_x'],s_ls['l1_y'], s_ls['l2_x'], weights= 1/np.sqrt(gns[dx][s_ls['ind_1']]**2 + gns[dy][s_ls['ind_1']]**2)) # Fit x-coordinates
#                 fit_yw, fm_y = or_fit(model_y, s_ls['l1_x'],s_ls['l1_y'], s_ls['l2_y'],weights= 1/np.sqrt(gns[dx][s_ls['ind_1']]**2 + gns[dy][s_ls['ind_1']]**2))
#                 
#                 fm_xy = np.logical_not(np.logical_and(np.logical_not(fm_x),np.logical_not(fm_y)))
#                 
#                 fig, ax = plt.subplots(1,1)
#                 plt_per = 20
#                 ax.set_title(f'Degree = {deg}')
#                 ax.scatter(s_ls['l1_x'][::plt_per],s_ls['l1_y'][::plt_per], label = 'Used (%s%%)'%(plt_per))
#                 ax.scatter(s_ls['l1_x'][fm_xy],s_ls['l1_y'][fm_xy],s =200, label = 'Clipped')
#                 ax.legend()
#             elif f_mode == 'NWnC':
#                 or_fit = fitting.FittingWithOutlierRemoval(fitter, sigma_clip, niter=3, sigma=3.0)
#                 
#                 fit_xw, fm_x = or_fit(model_x, s_ls['l1_x'],s_ls['l1_y'], s_ls['l2_x']) # Fit x-coordinates
#                 fit_yw, fm_y = or_fit(model_y, s_ls['l1_x'],s_ls['l1_y'], s_ls['l2_y']) 
#                 
#                 fm_xy = np.logical_not(np.logical_and(np.logical_not(fm_x),np.logical_not(fm_y)))
#                 
#                 fig, ax = plt.subplots(1,1)
#                 plt_per = 20
#                 ax.set_title(f'Degree = {deg}')
#                 ax.scatter(s_ls['l1_x'][::plt_per],s_ls['l1_y'][::plt_per], label = 'Used (%s%%)'%(plt_per))
#                 ax.scatter(s_ls['l1_x'][fm_xy],s_ls['l1_y'][fm_xy],s =200, label = 'Clipped')
#                 ax.legend()
#                 
#             # sys.exit(308)
#             
#             xi = fit_xw(gns[id1], gns[id2])
#             yi = fit_yw(gns[id1], gns[id2])# Fit y-coordinates
#         
#         dic_xy[f'trans_{loop}'] = np.array([xi,yi]).T    
#             
#         gns[id1] = xi
#         gns[id2] = yi
#             
#             # l1_xy = np.array([gns[id1],gns[id2]]).T
#             # g1_xy = np.array([gaia['x'], gaia['y']]).T
#         
#         
#         
#         l_xy = np.array([gns[id1],gns[id2]]).T
#         ga_xy = np.array([gaia['x'], gaia['y']]).T
#         
#         
#         
#         s_ls = compare_lists(l_xy,ga_xy,d_m)
#         print(f'\nCommon GNS{survey} and Gaia after loop{loop} = {len(s_ls)}')
#                 
#         comom_ls.append(len(s_ls))
#         # if comom_ls[-1] <= comom_ls[-2] :
#         # # if comom_ls[-1] < comom_ls[-2]:
#         #     if len(comom_ls)>2:
#                 
#         #         gns[id1] = dic_xy[f'trans_{loop-1}'][:,0]
#         #         gns[id2] = dic_xy[f'trans_{loop-1}'][:,1]
#         #         deg +=1
#         #         print(30*'-'+f'\nBreaking after loop = {loop-1}\nStarting alignment degree = {deg}\n' + 30*'-')
#         #         continue
#         #     else:
#         #         print(f'Polynomial of degree {deg} does not work')
#         #         gns[id1] = dic_xy['trans_0'][:,0]
#         #         gns[id2] = dic_xy['trans_0'][:,1]
#         #         break
#         if comom_ls[-1] <= comom_ls[-2] :
#             if len(comom_ls)>2:
#             # if len(comom_ls)>10:
#                 
#                 # gns[id1] = dic_xy[f'trans_{loop-1}'][:,0]
#                 # gns[id2] = dic_xy[f'trans_{loop-1}'][:,1]
#                 deg +=1
#                 print(30*'-'+f'\nBreaking after loop = {loop-1}\nStarting alignment degree = {deg}\n' + 30*'-')
#                 continue
#             # else:
#             #     print(f'Polynomial of degree {deg} does not work')
#             #     gns[id1] = dic_xy['trans_0'][:,0]
#             #     gns[id2] = dic_xy['trans_0'][:,1]
#             #     break
#        
#         
#         
#         # ax.legend()
#         # ax1.legend()
#         loop +=1     
#         
#     
#     return gns
# =============================================================================
