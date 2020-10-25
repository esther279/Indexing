import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import fabio
from random import *
import scipy
from fun_index import *

def rand_color(a, b):
    r = b-a
    color = (random.random()*r+a, random.random()*r+a, random.random()*r+a)
    return color

def plot_1D(x, y, axis2=[], fignum=100, log=0, color='b', xlabel = '', verbose=1):
    if fignum>0:
        plt.figure(fignum,figsize=(15,7)); plt.clf()
        #plt.subplot(2,1,1)

        #plt.subplot(2,1,2)
    ax1 = plt.subplot(111)
    plt.plot(x, y, color = color, alpha=0.6)
    locs = np.arange(0.5, 3.5, 0.5)
    ax1.set_xticks(locs)

    if log: plt.yscale('log')
    plt.xlabel(xlabel)
    plt.grid()
    #plt.title(fn_rad)
    peaks, _ = scipy.signal.find_peaks(y, height=0, width=1, prominence=(0.2, None))
    ylim = [np.nanmin(y[y != -np.inf]), np.nanmax(y)]
    yrange = ylim[1]-ylim[0]
    if verbose:
        for idx_p, peak in enumerate(peaks):
                plt.plot([x[peak], x[peak]], [ylim[0], y[peak]], '--', color='k') #rand_color(0.3, 0.9)
                if verbose>1:
                    plt.text(x[peak], y[peak], str(np.round(x[peak],2)),fontweight='bold')
                    plt.text(x[peak]+0.12, y[peak], '('
                            +str(np.round(2*np.pi/x[peak],2))+')', color='b')
                    #plt.text(x[peak], y[0]+(idx_p%5+1)*yrange*0.09, str(np.round(x[peak],2)),fontweight='bold')
                    #plt.text(x[peak]+0.12, y[0]+(idx_p%5+1)*yrange*0.09, '('
                    #        +str(np.round(2*np.pi/x[peak],2))+')', color='b')

    plt.legend(['polymorph1 (H)', 'polymorph2 (J)'], loc=1)
    if axis2 != []:
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(locs)
        #locs, labels = xticks()
        d = 2*np.pi/locs
        ax2.set_xticklabels(d.round(decimals=2))
        #ax2.set_xticks(idx.astype(int))
        #ax2.set_xticklabels(np.asarray(axis2)[idx.astype(int)].round(decimals=2))
        ax2.set_xlabel('d (A)')
        plt.grid()
    

# ========================================================================
# Plot integration and label peaks
# ========================================================================

## Vesta
home_dir = '/home/etsai/BNL/Users/SMI/ABraunschweig/'
fn_rad = 'Polymorph 1.int'
temp = np.loadtxt(home_dir+fn_rad, skiprows=2, unpack=True)
x2 = temp[0,:]    
x1 = [TwoThetatoQ(TwoTheta=xx, wavelength=12.4/16.1) for xx in x2]
idx = np.nonzero(x1>x[-1])[0][0]
x1 = x1[0:idx]  
y2 = temp[1,:][0:idx]
y2 = y2/np.mean(y2)
plot_1D(x1, y2, fignum=0, log=1, color=[0.7,0.0,0], xlabel='q(A^-1)', verbose=0)

## Vesta
home_dir = '/home/etsai/BNL/Users/SMI/ABraunschweig/'
fn_rad = 'Polymorph 2.int'
temp = np.loadtxt(home_dir+fn_rad, skiprows=2, unpack=True)
x2 = temp[0,:]    
x1 = [TwoThetatoQ(TwoTheta=xx, wavelength=12.4/16.1) for xx in x2]
idx = np.nonzero(x1>x[-1])[0][0]
x1 = x1[0:idx]  
y2 = temp[1,:][0:idx]
y2 = y2/np.mean(y2)
plot_1D(x1, y2, axis2=1, fignum=0, log=1, color=[0,0.7,0], xlabel='q(A^-1)', verbose=0)
    

## Data
home_dir = '/home/etsai/BNL/Users/SMI/ABraunschweig/2020C1/306008_analysis/Results/AB2/txt/'
fn_rad = 'Radint_sum_waxs_AB2_1_1_MeDPP_glass_thermal_none_0.0800deg_x49750_.txt'
#fn_rad = 'Radint_sum_waxs_AB2_1_7_MeDPP_glass_thermal_240minVSADCM_0.0800deg_x-31250_.txt'
x, y = np.loadtxt(home_dir+fn_rad, delimiter=' ', usecols=(0, 2), unpack=True)
x = x[0:-1000]
y = y[0:-1000]
y = y/np.mean(y)
#plot_1D(x/1.04-0.005, y, fignum=100, log=1, color='k', xlabel='q', verbose=2)
plot_1D(x, y, axis2=1, fignum=0, log=1, color='k', xlabel='q(A^-1)', verbose=2)
plt.title('{}'.format(fn_rad))





