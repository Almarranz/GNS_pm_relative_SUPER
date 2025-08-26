#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 13:18:16 2023

@author: amartinez
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import sys
from astropy.table import Table
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FormatStrFormatter
from astropy.io import ascii
import astropy.coordinates as ap_coor
from astropy.io import fits
from astropy import wcs
from astropy.wcs import WCS
import alphashape
from astropy.table import Table
# import species
# %%plotting pa    metres
from matplotlib import rc
from matplotlib import rcParams
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '25.5'})
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
rcParams.update({'font.size': 40})
rcParams.update({'figure.figsize': (10, 5)})
rcParams.update({
    "text.usetex": False,
    "font.family": "sans",
    "font.sans-serif": ["Palatino"]})
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
plt.rcParams.update({'figure.max_open_warning': 0})


def finder(pmra, pmdec, x, y, Ra, Dec, clustered_by,color_A, color_B,
           samples_dist, gen_sim, sim_lim, save_reg = None):
    
    clus_dic = {}
    coordenadas = SkyCoord(ra=Ra, dec=Dec, unit='degree')
    pmra_kernel, pmdec_kernel = gaussian_kde(pmra), gaussian_kde(pmdec)
    x_kernel, y_kernel = gaussian_kde(x), gaussian_kde(y)
    colorines  = color_A-color_B
   
    if clustered_by == 'pm_xy_color':
        X = np.array([pmra, pmdec, x, y, colorines]).T
        X = np.array([pmra, pmdec, x, y, colorines]).T
        X_stad = StandardScaler().fit_transform(X)
        tree = KDTree(X_stad, leaf_size=2)
        # DistNnce to the 1,2,3...k neighbour
        dist, ind = tree.query(X_stad, k=samples_dist)
        d_KNN = sorted(dist[:, -1])  # distance to the Kth neighbour
    elif clustered_by == 'pm_xy':
        X = np.array([pmra, pmdec, x, y]).T
        X_stad = StandardScaler().fit_transform(X)
        tree = KDTree(X_stad, leaf_size=2)
        # DistNnce to the 1,2,3...k neighbour
        dist, ind = tree.query(X_stad, k=samples_dist)
        d_KNN = sorted(dist[:, -1])  # distance to the Kth neighbour
    elif clustered_by == 'pm_color':
        X = np.array([pmra, pmdec, colorines]).T
        X_stad = StandardScaler().fit_transform(X)
        tree = KDTree(X_stad, leaf_size=2)
        # DistNnce to the 1,2,3...k neighbour
        dist, ind = tree.query(X_stad, k=samples_dist)
        d_KNN = sorted(dist[:, -1])  # distance to the Kth neighbour

    elif clustered_by == 'pm':
        X = np.array([pmra, pmdec]).T
        X_stad = StandardScaler().fit_transform(X)
        tree = KDTree(X_stad, leaf_size=2)
        # DistNnce to the 1,2,3...k neighbour
        dist, ind = tree.query(X_stad, k=samples_dist)
        d_KNN = sorted(dist[:, -1])  # distance to the Kth neighbour

    lst_d_KNN_sim = []
    if gen_sim == 'kernnel':
        for d in range(20):
            mudec_sim,  mura_sim = pmdec_kernel.resample(
                len(pmdec)), pmra_kernel.resample(len(pmra))
            x_sim, y_sim = x_kernel.resample(len(x)), y_kernel.resample(len(y))
            
            if clustered_by == 'pm_xy_color':
                color_kernel = gaussian_kde(colorines)
                color_sim = color_kernel.resample(len(pmdec))
                X_sim = np.array([mura_sim[0], mudec_sim[0],
                                 x_sim[0], y_sim[0], color_sim[0]]).T
                X_stad_sim = StandardScaler().fit_transform(X_sim)
                tree_sim = KDTree(X_stad_sim, leaf_size=2)

                # DistNnce to the 1,2,3...k neighbour
                dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
                # distance to the Kth neighbour
                d_KNN_sim = sorted(dist_sim[:, -1])

                lst_d_KNN_sim.append(min(d_KNN_sim))
            elif clustered_by == 'pm_xy':
                X_sim = np.array(
                    [mura_sim[0], mudec_sim[0], x_sim[0], y_sim[0]]).T
                X_stad_sim = StandardScaler().fit_transform(X_sim)
                tree_sim = KDTree(X_stad_sim, leaf_size=2)

                # DistNnce to the 1,2,3...k neighbour
                dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
                # distance to the Kth neighbour
                d_KNN_sim = sorted(dist_sim[:, -1])

                lst_d_KNN_sim.append(min(d_KNN_sim))
            elif clustered_by == 'pm_color':
                color_kernel = gaussian_kde(colorines)
                color_sim = color_kernel.resample(len(pmdec))
                X_sim = np.array([mura_sim[0], mudec_sim[0], color_sim[0]]).T
                X_stad_sim = StandardScaler().fit_transform(X_sim)
                tree_sim = KDTree(X_stad_sim, leaf_size=2)

                # DistNnce to the 1,2,3...k neighbour
                dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
                # distance to the Kth neighbour
                d_KNN_sim = sorted(dist_sim[:, -1])

                lst_d_KNN_sim.append(min(d_KNN_sim))

            elif clustered_by == 'pm':
                X_sim = np.array([mura_sim[0], mudec_sim[0]]).T
                X_stad_sim = StandardScaler().fit_transform(X_sim)
                tree_sim = KDTree(X_stad_sim, leaf_size=2)

                # DistNnce to the 1,2,3...k neighbour
                dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist)
                # distance to the Kth neighbour
                d_KNN_sim = sorted(dist_sim[:, -1])

                lst_d_KNN_sim.append(min(d_KNN_sim))

    d_KNN_sim_av = np.mean(lst_d_KNN_sim)
    if sim_lim == 'mean':
        eps_av = round((min(d_KNN)+d_KNN_sim_av)/2, 3)
        valor = d_KNN_sim_av
    elif sim_lim == 'minimun':
        eps_av = round((min(d_KNN)+min(lst_d_KNN_sim))/2, 6)
        valor = min(lst_d_KNN_sim)
    elif sim_lim == 'maximun':
        eps_av = round((min(d_KNN)+max(lst_d_KNN_sim))/2, 3)
        valor = min(lst_d_KNN_sim)

# =============================================================================
#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     ax.hist(d_KNN, bins='auto', histtype='step', color='k')
#     ax.hist(d_KNN_sim, bins='auto', histtype='step', color='r')
#     ax.set_xlabel('%s-NN distance' % (samples_dist))
#     ax.set_xlim(0,2)
# =============================================================================
    
    clus_method = 'dbs'

    clustering = DBSCAN(eps=eps_av, min_samples=samples_dist).fit(X_stad)
    # clustering = DBSCAN(eps=0.56, min_samples=samples_dist).fit(X_stad)
    l = clustering.labels_

    n_clusters = len(set(l)) - (1 if -1 in l else 0)
    
    print('n_clusters =',n_clusters)
    n_noise = list(l).count(-1)
    # %
    u_labels = set(l)
    # Returns a color for each cluster. Each color consists in four number, RGBA, red, green, blue and alpha. Full opacity black would be then 0,0,0,1
    colors = [plt.cm.rainbow(i) for i in np.linspace(0, 1, len(set(l)))]
    # %

    # %
    for k in range(len(colors)):  # give noise color black with opacity 0.1
        if list(u_labels)[k] == -1:
            colors[k] = [0, 0, 0, 0.1]
    # %
    colores_index = []

    for c in u_labels:
        cl_color = np.where(l == c)
        colores_index.append(cl_color)

    gr_alpha, gr_color, gr_size = 0.2, 'r', 50  # TODO
    bg_alpha = 0.05
    cl_size = 300
    ms_size, ms_color = 100, '#1f77b4'
    if len(set(l)) > 1:
        for i in range(len(set(l))-1):
# =============================================================================
#             fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# 
#             ax.hist(d_KNN, bins='auto', histtype='step', color='k')
#             ax.hist(d_KNN_sim, bins='auto', histtype='step', color='r')
#             ax.set_xlabel('%s-NN distance' % (samples_dist))
# 
#             texto = '\n'.join(('min real d_KNN = %s' % (round(min(d_KNN), 3)),
#                                'limit set for sim d_KNN =%s' % (
#                                    round(valor, 3)),
#                                'average = %s' % (eps_av), '%s' % (sim_lim), '%s' % (gen_sim), clustered_by))
# 
#             props = dict(boxstyle='round', facecolor='w', alpha=0.5)
#             # place a text box in upper left in axes coords
#             ax.text(0.25, 0.50, texto, transform=ax.transAxes, fontsize=20,
#                     verticalalignment='top', bbox=props)
# 
#             ax.set_ylabel('N')
# =============================================================================
            if 'color' in clustered_by:
                fig, ax = plt.subplots(1, 3, figsize=(30, 10))
            else:
                fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            
            ax[0].tick_params(axis='both', labelsize=30)
            ax[1].tick_params(axis='both', labelsize=30)
           
            # color_de_cluster = '#ff7f0e'
            color_de_cluster = 'lime'  # TODO
            ax[0].scatter(X[:, 0][colores_index[-1]][::500], X[:, 1][colores_index[-1]][::500],
                          color=colors[-1], s=50, zorder=1, alpha=bg_alpha)

            ax[0].scatter(X[:, 0], X[:, 1], color=colors[-1], s=50, zorder=1)
            # ax[1].quiver(t_gal['l'][colores_index[-1]].value,t_gal['b'][colores_index[-1]].value, X[:,0][colores_index[-1]]-pms[2], X[:,1][colores_index[-1]]-pms[3], alpha=0.5, color=colors[-1])

            ax[0].scatter(X[:, 0][colores_index[i]], X[:, 1][colores_index[i]],
                          color=color_de_cluster, s=cl_size, zorder=2, edgecolor='k')
            
            ax[0].set_xlim(-15, 10)
            ax[0].set_ylim(-20, 10)
            ax[0].set_xlabel('$\\mu_{l*}$ (mas yr$^{-1})$', fontsize=30)
            ax[0].set_ylabel('$\\mu_{b}$ (mas yr$^{-1})$', fontsize=30)
            ax[0].invert_xaxis()
            ax[0].axis('scaled')
            
            # ax[0].hlines(0,-10,10,linestyle = 'dashed', color ='red')

            mul_sig, mub_sig = np.std(X[:, 0][colores_index[i]]), np.std(
                X[:, 1][colores_index[i]])
            mul_mean, mub_mean = np.mean(X[:, 0][colores_index[i]]), np.mean(
                X[:, 1][colores_index[i]])

            mul_sig_all, mub_sig_all = np.std(X[:, 0]), np.std(X[:, 1])
            mul_mean_all, mub_mean_all = np.mean(X[:, 0]), np.mean(X[:, 1])

            # Print pm in galactic

            vel_txt = '\n'.join(('$\overline{\mu}_{ra*}$ = %.2f, $\overline{\mu}_{dec}$  =  %.2f' % (mul_mean, mub_mean),
                                 '$\sigma_{\mu ra*}$ =  %.2f, $\sigma_{\mu dec}$ =  %.2f' % (mul_sig, mub_sig)))
            vel_txt_all = '\n'.join(('mura =  %.2f, $\mu_{dec}$ =  %.2f' % (round(mul_mean_all, 2), round(mub_mean_all, 2)),
                                     '$\sigma_{\mu ra*}$ =  %.2f, $\sigma_{\mu dec}$ =  %.2f' % (round(mul_sig_all, 2), round(mub_sig_all, 2))))

            propiedades = dict(
                boxstyle='round', facecolor=color_de_cluster, alpha=0.2)
            propiedades_all = dict(
                boxstyle='round', facecolor=colors[-1], alpha=0.1)
            propiedades_gr = dict(
                boxstyle='round', facecolor=gr_color, alpha=0.2)
            ax[0].text(0.05,0.95, vel_txt, transform=ax[0].transAxes, fontsize=30,
                       verticalalignment='top', bbox=propiedades)
            # ax[0].text(0.05, 0.15, vel_txt_all, transform=ax[0].transAxes, fontsize=20,
            #     verticalalignment='top', bbox=propiedades_all)

            # This calcualte the maximun distance between cluster members to have a stimation of the cluster radio
            c2 = SkyCoord(ra=Ra[colores_index[i]]*u.deg,
                          dec=Dec[colores_index[i]]*u.deg)
            sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
            rad = max(sep)/2

            radio_MS = max(sep)

            # Added this bit to calculate the half-light radio (eff radio)
            # arches_dbs = arches[colores_index[i]]
            # all_mag_clus = arches_dbs['F153M']
            clus_cent = SkyCoord(ra=[np.median(Ra[colores_index[i]])], dec=[np.median(Dec[colores_index[i]])],
                                 unit='degree')
            clus_coord = SkyCoord(
                ra=Ra[colores_index[i]], dec=Dec[colores_index[i]], unit='degree')

            clus_mag = color_B[colores_index[i]]

          

            m_point = SkyCoord(ra=[np.mean(c2.ra)], dec=[np.mean(c2.dec)])

            idxc, group_md, d2d, d3d = ap_coor.search_around_sky(
                m_point, coordenadas, rad*1.5)

            mul_sig_gr, mub_sig_gr = np.std(
                X[:, 0][group_md]), np.std(X[:, 1][group_md])
            mul_mean_gr, mub_mean_gr = np.mean(
                X[:, 0][group_md]), np.mean(X[:, 1][group_md])

            vel_txt_gr = '\n'.join(('$\overline{\mu}_{l*}$ = %.2f, $\overline{\mu}_{b}$ = %.2f' % (mul_mean_gr, mub_mean_gr),
                                    '$\sigma_{\mu l*}$ = %.2f, $\sigma_{\mu b}$ = %.2f' % (mul_sig_gr, mub_sig_gr)))

            
            ax[0].text(0.05, 0.15, vel_txt_gr, transform=ax[0].transAxes, fontsize=20,
                       verticalalignment='top', bbox=propiedades_gr)


            # if tipo == 'MultiPolygon':
            
            

           
            ax[1].axis('equal')
            ax[1].scatter(Ra[group_md], Dec[group_md], color=gr_color,
                          s=gr_size, zorder=1, marker='x', alpha=gr_alpha)
            ax[0].scatter(X[:, 0][group_md], X[:, 1][group_md], color=gr_color,
                          s=gr_size, zorder=1, marker='x', alpha=1)
            prop = dict(boxstyle='round',
                        facecolor=color_de_cluster, alpha=0.2)
            # ax[1].text(0.05, 0.15, 'hl radius $\sim$ %.2f"\n stars = %s '%(eff_rad,len(colores_index[i][0])),len(colores_index[i][0])), transform=ax[1].transAxes, fontsize=30,
            #                         verticalalignment='top', bbox=prop)
            # ax[1].text(0.05, 0.15, 'Radius $\sim$ %.0f"\n # stars = %s '%(rad.to(u.arcsec).value,len(colores_index[i][0])), transform=ax[1].transAxes, fontsize=30,
            #                         verticalalignment='top', bbox=prop)
            ax[1].text(0.20, 0.95, 'Radius $\sim$ %.0f"\n # stars = %s\n l = %.2f b = %.2f ' % (rad.to(u.arcsec).value, len(colores_index[i][0]),clus_cent.ra.value,clus_cent.dec.value), transform=ax[1].transAxes, fontsize=30,
                       verticalalignment='top', bbox=prop)

            

            ax[1].scatter(Ra, Dec, color=colors[-1], s=50,
                          zorder=1, alpha=bg_alpha)  # plots in galactic
            
            ax[1].invert_xaxis()

          

            ax[1].scatter(Ra[colores_index[i]], Dec[colores_index[i]], color=color_de_cluster,
                          s=cl_size, zorder=2, edgecolor='k')  # plots in galactic
           
            ax[1].set_xlabel('l (°)', fontsize=30)
            ax[1].set_ylabel('b (°)', fontsize=30)
            # ax[1].yaxis.set_label_coords(-.05, .58)
            ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            
            if save_reg is not None:
                with open(save_reg + f'cluster_{i}.reg', 'w') as f:
                      f.write('# Region file format: DS9 version 4.1'+"\n"+'global color=cyan dashlist=8 3 width=4 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+"\n"+'galactic'+'\n')
                      f.close
                for j in range(len((Ra[colores_index[i]]))):
                               with open(save_reg + f'cluster_{i}.reg', 'a') as f:
                                   f.write('\n'.join(('point(%s,%s) # point=x'%(Ra[colores_index[i]][j],Dec[colores_index[i]][j]),'\n')))  
                               f.close

        
            # x_ticks = np.round(ax[1].get_xticks(), 2)
            # ax[1].set_xticks(np.unique(x_ticks))

            # y_ticks = np.round(ax[1].get_yticks(), 2)
            # # ax[1].set_yticklabels(y_ticks, rotation=90)
            # ax[1].set_yticks(y_ticks)
           
# =============================================================================
#             p2d = np.array([Ra[colores_index[i]],Dec[colores_index[i]]]).T
#             
#             print(p2d)
#             al_sh = alphashape.alphashape(p2d,10)
#             print('al_sh.area',al_sh.area)
#             tipo = al_sh.geom_type
#             print(tipo)
#             if tipo != 'Polygon':
#                 contorno = {}
#                 for ci,cont in enumerate(al_sh.geoms):
#                     contorno['cont%s'%(ci)] = al_sh.geoms[ci].exterior.coords[:]
#             elif tipo == 'Polygon':
#                 contorno = np.array(al_sh.exterior.coords[:])
#             supf = al_sh.area*3600
#             densidad = len(colores_index[i][0])/supf
#             if tipo == 'MultiPolygon':
#                 for cd in range(len(contorno)):
#                     ax[1].plot(np.array(contorno['cont%s'%(cd)])[:,0],np.array(contorno['cont%s'%(cd)])[:,1], label = 'Dens = %.0f stars/min2'%(densidad), color = 'k',lw =3)
#                
#             elif tipo != 'MultiPolygon':
#                 ax[1].plot(contorno[:,0],contorno[:,1], label = 'Dens = %.0f stars/min2'%(densidad),color = 'k',lw =3)
#             ax[1].legend(loc = 3,fontsize=30)
#             
# =============================================================================
            if 'color' in clustered_by:
                ax[2].tick_params(axis='both', labelsize=30)
                ax[2].scatter(color_A[colores_index[i]] -color_B[colores_index[i]], color_B[colores_index[i]], color=color_de_cluster,
                              s=cl_size, zorder=2, edgecolor='k')
                
                ax[2].scatter(color_A[group_md] -color_B[group_md], color_B[group_md],color=gr_color,
                              s=gr_size, zorder=1, marker='x', alpha=gr_alpha)
                ax[2].invert_yaxis()
                ax[2].set_xlabel('H-Ks',fontsize=30)
                ax[2].set_ylabel('Ks', fontsize=30)
                txt_color = '\n'.join(('H-Ks =%.3f' % (np.median(color_A[colores_index[i]]-color_B[colores_index[i]])),
                                       '$\sigma_{H-Ks}$ = %.3f' % (
                                           np.std(color_A[colores_index[i]]-color_B[colores_index[i]])),
                                       'diff_color = %.3f' % (max(color_A[colores_index[i]]-color_B[colores_index[i]])-min(color_A[colores_index[i]]-color_B[colores_index[i]]))))
                props = dict(boxstyle='round',
                             facecolor=color_de_cluster, alpha=0.3)
                
                ax[2].text(0.50, 0.95, txt_color, transform=ax[2].transAxes, fontsize=30,
                                       verticalalignment='top', bbox=props)
                # ax[2].set_xlim(np.min(color_A[colores_index[i]] -color_B[colores_index[i]]),np.max(color_A[colores_index[i]] -color_B[colores_index[i]]))
            clus_dic[f'clus_{i}'] = Table({
                    'RA': Ra[colores_index[i]],
                    'Dec': Dec[colores_index[i]],
                    'x': X[:, 2][colores_index[i]],
                    'y': X[:, 3][colores_index[i]],
                    'pm_x': X[:, 0][colores_index[i]],
                    'pm_y': X[:, 1][colores_index[i]],
                    'color_A': color_A[colores_index[i]],
                    'color_B': color_B[colores_index[i]],
                    })
            fig.tight_layout()
            plt.savefig('/Users/amartinez/Desktop/PhD/Charlas/EAS_Cork/images/sgrB1_clus.png',bbox_inches = 'tight', transparent = True)
            plt.show()
    return clus_dic        
            # guarda = input('Do ypu want to save this cluster? (y/n)')
            # if guarda == 'y':
            #     clus_np = np.c_[Ra[colores_index[i]], Dec[colores_index[i]],
            #                     X[:, 2][colores_index[i]], X[:, 3][colores_index[i]],
            #                    X[:, 0][colores_index[i]], X[:, 1][colores_index[i]],
            #                    color_A[colores_index[i]]]
                               
            #     clu_t = Table(clus_np, names = ('RA','Dec','x','y','pm_x(mas/yr)','pm_y(mas(yr)','IB224'))
            #     pruebas = '/Users/amartinez/Desktop/Projects/GNS_gd/pruebas/'
            #     clu_t.write(pruebas + f'clus_{i}_knn{samples_dist}.txt', format='ascii', overwrite=True)
            #     return clu_t
            #     sys.exit('Bye')
            
 # sys.exit(161)
#