import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import fabio
import random 

def rand_color(a, b):
    r = b-a
    color = (random.random()*r+a, random.random()*r+a, random.random()*r+a)
    return color

def plot_1D(x, y, fignum=100, log=0, color='b', xlabel = '', verbose=1):
    if fignum>0:
        plt.figure(fignum,figsize=(15,7)); #plt.clf()
        #plt.subplot(2,1,1)

        #plt.subplot(2,1,2)
    plt.plot(x, y, color = color, alpha=0.5)
    if log: plt.yscale('log')
    plt.xlabel(xlabel)
    plt.grid()
    plt.title(fn_rad)
    peaks, _ = find_peaks(y, height=0, width=2, prominence=(0.4, None))
    ylim = [np.nanmin(y[y != -np.inf]), np.nanmax(y)]
    yrange = ylim[1]-ylim[0]
    if verbose:
        for idx_p, peak in enumerate(peaks):
                plt.plot([x[peak], x[peak]], ylim, '--', color='k') #rand_color(0.3, 0.9)
                if verbose>1:
                    plt.text(x[peak], y[0]+(idx_p%5+1)*yrange*0.09, str(np.round(x[peak],2)),fontweight='bold')
                    plt.text(x[peak]+0.12, y[0]+(idx_p%5+1)*yrange*0.09, '('
                            +str(np.round(2*np.pi/x[peak],2))+')', color='b')


# ========================================================================
# Plot integration and label peaks
# ========================================================================
home_dir = '/home/etsai/BNL/Users/SMI/ABraunschweig/2020C1/306008_analysis/Results/AB2/txt/'
fn_rad = 'Radint_sum_waxs_AB2_1_1_MeDPP_glass_thermal_none_0.0800deg_x49750_.txt'
x, y = np.loadtxt(home_dir+fn_rad, delimiter=' ', usecols=(0, 2), unpack=True)
x = x[0:-1000]
y = y[0:-1000]
y = y/np.mean(y)
plot_1D(x, y, fignum=100, log=1, xlabel='q', verbose=2)

## Vesta
home_dir = '/home/etsai/BNL/Users/SMI/ABraunschweig/'
fn_rad = 'Polymorph 2.int'
temp = np.loadtxt(home_dir+fn_rad, skiprows=2, unpack=True)
x2 = temp[0,:]    
x2 = [TwoThetatoQ(TwoTheta=xx, wavelength=12.4/16.1) for xx in x2]
idx = np.nonzero(x2>x[-1])[0][0]
x2 = x2[1:idx]  
y2 = temp[1,:][1:idx]
y2 = y2/np.mean(y2)
plot_1D(x2, y2, fignum=0, log=1, color=[0,0.7,0], xlabel='q(A^-1)', verbose=0)
    
## Vesta
home_dir = '/home/etsai/BNL/Users/SMI/ABraunschweig/'
fn_rad = 'Polymorph 1.int'
temp = np.loadtxt(home_dir+fn_rad, skiprows=2, unpack=True)
x2 = temp[0,:]    
x2 = [TwoThetatoQ(TwoTheta=xx, wavelength=12.4/16.1) for xx in x2]
idx = np.nonzero(x2>x[-1])[0][0]
x2 = x2[1:idx]  
y2 = temp[1,:][1:idx]
y2 = y2/np.mean(y2)
plot_1D(x2, y2, fignum=0, log=1, color=[0.7,0.0,0], xlabel='q(A^-1)', verbose=0)




