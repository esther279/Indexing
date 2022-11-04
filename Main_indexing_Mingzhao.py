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
Co-Si system:
CoSi2, cubic, Fm3m, a = 5.36 A
CoSi, cubic, P213, a = 4.433 A
Co2Si, orthorhombic, Pnma, a = 3.71 A, b = 4.904 A, c = 7.066 A
Co, hexgonal, P63/mmc, a = b = 2.501 A, c = 4.033 A, alpha = beta = 90, gamma = 120
 
Pt-Si system:
PtSi, orthohombic, Pnma, a = 3.638 A, b = 5.667 A, c = 5.982 A
Pt2Si, tetragonal, I4/mmm, a = b = 3.983 A, c = 4.111 A, alpha = beta = 118.977, gamma = 90
Pt, cubic, Fm3m, a = 3.924 A
'''

x = 1
y = [-0, -0, 0]
param0 = { 
    "name": "-",
    "a": 2.501/x +y[0],
    "b": 2.501/x +y[1],
    "c": 4.033/x +y[2], 
    "alp_deg": 90,
    "beta_deg": 90,
    "gam_deg": 120, 
    "spacegroup": 1, 
    "ori_hkl": [0,  0 , 1],  
    "range_hkl": [[-1, 3], [-1, 3], [0, 3]],
    "filename": '/home/etsai/BNL/Users/CMS/2019C3/ETsai2/waxs/analysis/qr_image/Eli_sample_Co_Si_13.5kev_th0.100_x9.000_15.00s_2666500_waxs.npz', 
}


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
    "lim1": [-1.3, 3.5],  # axis limits
    "lim2": [-0.5, 3.5],  # axis limits
    "vmin": -1,  # colorbar range
    "vmax": 400,
    "textcolor": 'c',
    "cmap": 'magma', #'jet',
    "FS": 10
}
plt.figure(10, figsize=(12,10)); plt.clf()
plot_index(data, Qxy_list, Qz_list, hkl_list, **param_plot) 
plt.title("{}: a{},b{},c{},alpha{},beta{},gam{},spacegroup{}\n orientation {}\n{}".format(
          param['name'],param['a'],param['b'],param['c'],param['alp_deg'],param['beta_deg'],
          param['gam_deg'],param['spacegroup'],param["ori_hkl"],param['filename'][-100:-1]), size=10,fontweight='bold')


