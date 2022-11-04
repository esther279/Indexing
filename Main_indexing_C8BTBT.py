#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import fabio
from random import *
from fun_index import *

# Ruipeng Li, Esther Tsai
# 2019
#
# Ref
# http://lampx.tugraz.at/~hadley/ss1/crystalstructure/structures/hcp/hcp.php
# https://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-hkl?fchoose=choose
# http://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-hkl?gnum=194

## INPUT parameters
param_beam = {
    "geometry": 'GI', # Specify GI or Transmission
    "lambda_A": 12.4/13.5,    # (Currently unused)
    "inc_theta_deg": [0.12, 0.1, 0.15], # (Currently unused) [Inci, Inci_c, Inci_s]
    "verbose": 1
}

'''
C8BTBT P21/c
'''
param1 = { 
    "name": "C8BTBT",
    "a": 5.93,
    "b": 7.88,
    "c": 29.18, 
    "alp_deg": 90,
    "beta_deg": 92.4,
    "gam_deg": 90, 
    "spacegroup": 14, # 194(HCP), 225(FCC), 229(BCC), 62(pnma) # See fun_index.py -> check_ref(h, k, l, spacegroup)
    "ori_hkl": [0, 0, 1],  
    "range_hkl": [[0, 3], [-1, 3], [0, 9]],
    "filename": '/home/etsai/BNL/Research/GIWAXS_tomo_2019C2/BTBT_TOMO_test3_25_average/BTBT_TOMO_test3_25_average.npz', 
}


param2 = { 
    "name": "C8BTBT",
    "a": 5.83,
    "b": 7.88,
    "c": 29.18, 
    "alp_deg": 90,
    "beta_deg": 99.4,
    "gam_deg": 90, 
    "spacegroup": 14, # 194(HCP), 225(FCC), 229(BCC), 62(pnma) # See fun_index.py -> check_ref(h, k, l, spacegroup)
    "ori_hkl": [0, 0, 1],  
    "range_hkl": [[-2, 3], [0, 3], [0, 11]],
    "filename": '/home/etsai/BNL/Research/GIWAXS_tomo_2019C2/BTBT_TOMO_test3_25_average/BTBT_TOMO_test3_25_average.npz',

}

param0 = { 
    "name": "C8-C12-BTBT",
    "a": 5.86,
    "b": 7.72,
    "c": 33.7, 
    "alp_deg": 90,
    "beta_deg": 93.2,
    "gam_deg": 90, 
    "spacegroup": 14, 
    "ori_hkl": [0, 0, 1],  
    "range_hkl": [[-1,1], [-1, 0], [0, 1]],
    "filename": '/home/etsai/BNL/Research/GIWAXS_tomo_2019C2/BTBT_TOMO_test3_25_average/BTBT_TOMO_test3_25_average.npz',

}


# to-do: vesta intensity? search through hkl
######

param = param0
data = []   
#get_hint(param['spacegroup'])

## Index
hkl_list, Qxy_list, Qz_list, q_data = get_index(**param,**param_beam)


## Load and plot data
#%matplotlib inline 
flag_plot=0
if 'filename' in param:
    fn = param['filename']
    if fn.find('tiff')>0 or fn.find('.npz')>0:
        data = load_data(fn)
    
    if fn.find('.tiff')>0:
        temp = fabio.open(fn)
        if flag_plot:
            plt.figure()
            plt.imshow(temp.data, vmin=0, vmax=2000)
            plt.colorbar(); plt.title(fn)
    elif fn.find('png')>0 and fn.find('.npz')<0:        
        img = mpimg.imread(fn)
        if flag_plot:
            plt.figure()
            plt.imshow(img); plt.title(fn)
  
## Plot index
param_plot = {
    "log10": 0,
    "lim1": [-2, 3+1],  # axis limits
    "lim2": [-2, 2.75+1],  # axis limits
    "vmin": 4,  # colorbar range
    "vmax": 24,
    "textcolor": 'r',
    "cmap": 'magma', #'jet',
    "FS": 10
}
plt.figure(10, figsize=(12,10)); plt.clf()
plot_index(data, Qxy_list, Qz_list, hkl_list, **param_plot) 
plt.title("{}: a{},b{},c{},alpha{},beta{},gam{},spacegroup{}\n orientation {}\n{}".format(
          param['name'],param['a'],param['b'],param['c'],param['alp_deg'],param['beta_deg'],
          param['gam_deg'],param['spacegroup'],param["ori_hkl"],param['filename'][-100:-1]), size=10,fontweight='bold')




### Calculate angle 
def calc_rot(thx, thz):
    theta_x = 2*thx/180*np.pi 
    theta_z = 2*thz/180*np.pi 
    
    temp = 1-np.cos(theta_z)*np.cos(theta_x)
    alpha_p = np.arctan(np.sin(theta_x)*np.cos(theta_z)/temp)/np.pi*180
    
    alpha = 90-alpha_p
    print("2theta_x = {:.2f} \n2theta_z = {:.2f}\n 2alpha = {:.2f}\n----".format(thx*2, thz*2,2*alpha))
    return alpha

theta = QtoTheta(Q=1.7, wavelength=12.4/13.5)
print("2theta = {:.2f}".format(2*theta))

alpha = calc_rot(QtoTheta(Q=2.16, wavelength=12.4/13.5), QtoTheta(Q=1.55, wavelength=12.4/13.5)) 

alpha = calc_rot(QtoTheta(Q=2.3, wavelength=12.4/13.5), QtoTheta(Q=1.75, wavelength=12.4/13.5))



###

img = data['img']
line_x = data['x_axis']
line_y = np.log10(np.sum(img[330:-200,:],0))
plt.figure(11); plt.clf(); ax = plt.subplot(111)
plt.plot(line_x, line_y, 'k')
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
if 0:
    peaks, _ = find_peaks(line_y, height=0.1, distance=0.5/(line_x[1]-line_x[0])) 
    for idx, peak in enumerate(peaks):
        plt.plot([line_x[peak], line_x[peak]], [1.5, line_y[peak]], '--k')
        plt.text(line_x[peak], line_y[peak]+0.1, str(np.round(line_x[peak],3)),fontweight='normal', fontsize=16, color='b')
else:
     #qpeaks = [0.797, 1.0777, 1.340, 1.92474, 1.5947, 2.298220, 2.155, 2.68126]
     qpeaks = [np.round(xx,2) for xx in Qxy_list]   
     qpeaks = list(dict.fromkeys(qpeaks))     
     qpeaks = np.asarray(qpeaks)
     qpeaks.sort()
     qpeaks = qpeaks[[1, 3,4,5,6,7,8,9,10]]
     labels=['01L', '11L', '02L', '12L', '20L', '21L', '03L', '13L', '22L']
     for ii, q in enumerate(qpeaks):
         idx = np.abs(line_x-q).argmin()
         plt.plot([q, q], [2.6, line_y[idx]], '--b')
         offset=0.04
         if ii==6 or ii==8:
             offset = -0.01
         tt=plt.text(q-offset, line_y[idx]+0.18, "{} ({})".format(np.round(q,2),labels[ii]),fontweight='bold', fontsize=15, color='b')
         tt.set_rotation(90)
plt.ylim(2.8, 5.6); plt.xlim(-1, 3.9)
plt.yticks([]); plt.xticks(fontsize=15,fontweight='bold')
plt.xticks([-1, 0, 1, 2, 3]); plt.xlabel('q')


## List peaks (d = 2pi/q)
q_data.sort_values(by=['q'], ascending=True)
