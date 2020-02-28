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
    "lambda_A": 1,    # (Currently unused)
    "inc_theta_deg": [0.12, 0.1, 0.15], # (Currently unused) [Inci, Inci_c, Inci_s]
    "verbose": 1
}
param1 = { #CoSi2
    "a": 5.36,
    "b": 5.36,
    "c": 5.36, 
    "alp_deg": 90,
    "beta_deg": 90,
    "gam_deg": 90, 
    "spacegroup": 225, # 194(HCP), 225(FCC), 229(BCC), 62(pnma) # See fun_index.py -> check_ref(h, k, l, spacegroup)
    "ori_hkl": [0, 0, 1],  # Beam direction if Transmission, normal to film if GI
    "range_hkl": [[0, 3], [0, 3], [0, 3]],
    #"filename": './xray_data/SMI/q_map_SMI_example_waxs.tiff', # SMI (png or tiff); CMS (png or npz)
    #"filename": './xray_data/CMS/Eli_sample_Co_Si_13.5kev_th0.100_x-10.000_15.00s_2666462_waxs.npz',
    #"filename": './xray_data/CMS/Eli_sample_Co_Si_13.5kev_th0.100_x0.000_15.00s_2666482_waxs.npz',
    "filename": './xray_data/CMS/Eli_sample_Co_Si_13.5kev_th0.100_x10.000_15.00s_2666502_waxs.npz', 
}

param2 = { #CoSi
    "a": 4.433,
    "b": 4.433,
    "c": 4.433, 
    "alp_deg": 90,
    "beta_deg": 90,
    "gam_deg": 90, 
    "spacegroup": 229, # 194(HCP), 225(FCC), 229(BCC), 62(pnma) # See fun_index.py -> check_ref(h, k, l, spacegroup)
    "ori_hkl": [1, 0, 0],  # Beam direction if Transmission, normal to film if GI
    "range_hkl": [[0, 3], [0, 3], [0, 3]],
    #"filename": './xray_data/SMI/q_map_SMI_example_waxs.tiff', # SMI (png or tiff); CMS (png or npz)
    "filename": './xray_data/CMS/Eli_sample_Co_Si_13.5kev_th0.100_x-10.000_15.00s_2666462_waxs.npz',
    "filename": './xray_data/CMS/Eli_sample_Co_Si_13.5kev_th0.100_x0.000_15.00s_2666482_waxs.npz',
    #"filename": './xray_data/CMS/Eli_sample_Co_Si_13.5kev_th0.100_x10.000_15.00s_2666502_waxs.npz', 
}


param3 = { #Co2Si
    "a": 3.71,
    "b": 4.904,
    "c": 7.066, 
    "alp_deg": 90,
    "beta_deg": 90,
    "gam_deg": 90, 
    "spacegroup": 62, # 194(HCP), 225(FCC), 229(BCC), 62(pnma) # See fun_index.py -> check_ref(h, k, l, spacegroup)
    "ori_hkl": [0, 1, 0],  # Beam direction if Transmission, normal to film if GI
    "range_hkl": [[0, 3], [0, 3], [0, 3]],
    #"filename": './xray_data/SMI/q_map_SMI_example_waxs.tiff', # SMI (png or tiff); CMS (png or npz)
    #"filename": './xray_data/CMS/Eli_sample_Co_Si_13.5kev_th0.100_x-10.000_15.00s_2666462_waxs.npz',
    "filename": './xray_data/CMS/Eli_sample_Co_Si_13.5kev_th0.100_x0.000_15.00s_2666482_waxs.npz',
    #"filename": './xray_data/CMS/Eli_sample_Co_Si_13.5kev_th0.100_x10.000_15.00s_2666502_waxs.npz', 
}

param4 = { #Co
    "a": 2.501,
    "b": 2.501,
    "c": 4.033, 
    "alp_deg": 90,
    "beta_deg": 90,
    "gam_deg": 120, 
    "spacegroup": 194, # 194(HCP), 225(FCC), 229(BCC), 62(pnma) # See fun_index.py -> check_ref(h, k, l, spacegroup)
    "ori_hkl": [0, 1, 0],  # Beam direction if Transmission, normal to film if GI
    "range_hkl": [[-1, 3], [-1, 3], [0, 3]],
    #"filename": './xray_data/SMI/q_map_SMI_example_waxs.tiff', # SMI (png or tiff); CMS (png or npz)
    "filename": './xray_data/CMS/Eli_sample_Co_Si_13.5kev_th0.100_x-10.000_15.00s_2666462_waxs.npz',
    #"filename": './xray_data/CMS/Eli_sample_Co_Si_13.5kev_th0.100_x0.000_15.00s_2666482_waxs.npz',
    #"filename": './xray_data/CMS/Eli_sample_Co_Si_13.5kev_th0.100_x10.000_15.00s_2666502_waxs.npz', 
}

######

param10 = { #PtSi
    "a": 3.638,
    "b": 5.667,
    "c": 5.982, 
    "alp_deg": 90,
    "beta_deg": 90,
    "gam_deg": 90, 
    "spacegroup": 62, # 194(HCP), 225(FCC), 229(BCC), 62(pnma) # See fun_index.py -> check_ref(h, k, l, spacegroup)
    "ori_hkl": [0, 1, 0],  # Beam direction if Transmission, normal to film if GI
    "range_hkl": [[-1, 3], [-1, 3], [0, 3]],
    #"filename": './xray_data/SMI/q_map_SMI_example_waxs.tiff', # SMI (png or tiff); CMS (png or npz)
    #"filename": './xray_data/CMS/Eli_sample_Co_Si_13.5kev_th0.100_x-10.000_15.00s_2666462_waxs.npz',
    "filename": './xray_data/CMS/Eli_sample_Co_Si_13.5kev_th0.100_x0.000_15.00s_2666482_waxs.npz',
    #"filename": './xray_data/CMS/Eli_sample_Co_Si_13.5kev_th0.100_x10.000_15.00s_2666502_waxs.npz', 
}

param11 = { #Pt2Si
    "a": 3.983,
    "b": 3.983,
    "c": 4.111, 
    "alp_deg": 118.977,
    "beta_deg": 118.977,
    "gam_deg": 90, 
    "spacegroup": 62, # 194(HCP), 225(FCC), 229(BCC), 62(pnma) # See fun_index.py -> check_ref(h, k, l, spacegroup)
    "ori_hkl": [0, 1, 0],  # Beam direction if Transmission, normal to film if GI
    "range_hkl": [[-1, 3], [-1, 3], [0, 3]],
    #"filename": './xray_data/SMI/q_map_SMI_example_waxs.tiff', # SMI (png or tiff); CMS (png or npz)
    #"filename": './xray_data/CMS/Eli_sample_Co_Si_13.5kev_th0.100_x-10.000_15.00s_2666462_waxs.npz',
    "filename": './xray_data/CMS/Eli_sample_Co_Si_13.5kev_th0.100_x0.000_15.00s_2666482_waxs.npz',
    #"filename": './xray_data/CMS/Eli_sample_Co_Si_13.5kev_th0.100_x10.000_15.00s_2666502_waxs.npz', 
}


param = param4
data = []   
get_hint(param['spacegroup'])

## Index
hkl_list, Qxy_list, Qz_list, q_data = get_index(**param,**param_beam)


## Load and plot data
#%matplotlib inline 
if 'filename' in param:
    fn = param['filename']
    if fn.find('tiff')>0 or fn.find('.npz')>0:
        data = load_data(fn)
        
    if fn.find('.tiff')>0:
        temp = fabio.open(fn)
        plt.figure()
        plt.imshow(temp.data, vmin=0, vmax=2000)
        plt.colorbar(); plt.title(fn)
    elif fn.find('png')>0 and fn.find('.npz')<0:        
        img = mpimg.imread(fn)
        plt.figure()
        plt.imshow(img); plt.title(fn)

        
## Plot index
param_plot = {
    "log10": 0,
    "lim1": [-0.5, 3.5],  # axis limits
    "vmin": -5,  # colorbar range
    "vmax": 200,
    "textcolor": 'c',
}
plt.figure(10, figsize=(12,10)); plt.clf()
plot_index(data, Qxy_list, Qz_list, hkl_list, **param_plot) 
plt.title(param['filename'])


## List peaks (d = 2pi/q)
q_data.sort_values(by=['q'], ascending=True)
