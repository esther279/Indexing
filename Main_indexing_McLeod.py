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
_cell_length_a   47.38400000
_cell_length_b   48.54184887
_cell_length_c   13.60778670
_cell_angle_alpha   94.23056945
_cell_angle_beta   89.63368277
_cell_angle_gamma   82.75229794
'''
x = 12
y = [-0, -0, 0]
param0 = { 
    "name": "-",
    "a": 47.384/x +y[0],
    "b": 48.542/x +y[1],
    "c": 13.608/x +y[2], 
    "alp_deg": 94.2306,
    "beta_deg": 89.6337,
    "gam_deg": 82.7523, 
    "spacegroup": 1, 
    "ori_hkl": [1, 1, 0],  
    "range_hkl": [[-1, 3], [-1, 3], [0, 3]],
    "filename": '/home/etsai/BNL/Users/CMS/RVerduzco_2020_3/waxs/analysis/qr_image_swaxs/DM_dcm-1-134-3_pos1_th0.100_x0.000_10.00s_130151_waxs_stitched.npz', 
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
    "lim1": [-1.3, 2.8],  # axis limits
    "lim2": [-0.5, 3],  # axis limits
    "vmin": -10,  # colorbar range
    "vmax": 150,
    "textcolor": 'w',
    "cmap": 'magma', #'jet',
    "FS": 10
}
plt.figure(10, figsize=(12,10)); plt.clf()
plot_index(data, Qxy_list, Qz_list, hkl_list, **param_plot) 
plt.title("{}: a{},b{},c{},alpha{},beta{},gam{},spacegroup{}\n orientation {}\n{}".format(
          param['name'],param['a'],param['b'],param['c'],param['alp_deg'],param['beta_deg'],
          param['gam_deg'],param['spacegroup'],param["ori_hkl"],param['filename'][-100:-1]), size=10,fontweight='bold')


