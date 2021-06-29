#!/usr/bin/python
# -*- coding: utf-8 -*-
################################################################################
# reciprocal_peaks.py
#    version 0.9.1
################################################################################
# Performs various computatiosn on reciprocal space peaks, vectors, etc.
################################################################################

if False:
    # Determine which version of pyXS code to use
    import sys
    #PYXS_PATH='./'
    PYXS_PATH='/home/kyager/BNL/X9/beamline_software/pyXS_Analysis/main'
    PYXS_PATH in sys.path or sys.path.append(PYXS_PATH)
    from pyXSanalysis.processing import *



import os, glob
from math import sin, cos, radians, pi, sqrt
import numpy as np
import pylab as plt

import matplotlib as mpl

SciAnalysis_PATH='/nsls2/xf11bm/software/SciAnalysis/'
SciAnalysis_PATH = '/home/etsai/BNL/Users/software/SciAnalysis/'
SciAnalysis_PATH in sys.path or sys.path.append(SciAnalysis_PATH)
import glob
from SciAnalysis import tools
from SciAnalysis.XSAnalysis.Data import *
from SciAnalysis.XSAnalysis import Protocols



# Fitter    
###################################################################    
class Fitter(object):
    
    
    def __init__(self):
        self.fit_vals = []
    
    def fit_function_exp_rise_to_max(self, p, x ):
        """Fit function."""
        
        # Note that x and y are arrays, so we need numpy versions
        # of functions...
        y_initial, xi, plateau = p
        
        I = y_initial + (plateau-y_initial)*(1.0 - np.exp( -( x )/(xi) ) )
        
        return I
        
    def fit_function_linear(self, p, x ):
        """Fit function."""
        
        # Note that x and y are arrays, so we need numpy versions
        # of functions...
        m, b = p
        I = np.multiply(m,x) + b
        
        return I        

    def fit_function(self, p, x):
        
        return self.fit_function_linear( p, x )

    def residuals(self, p, x, I ):
        """Residuals function."""
        
        return I-self.fit_function(p, x)

    def residuals_with_limits(self, p, x, I, out_of_bounds_penalty=1e30):
        """This is an example of how to edit the residuals function so as
        to enforce limits on various values."""
        
        adjustment = 1.0
        m, b = p
        
        if m<0:
            adjustment = out_of_bounds_penalty
        
        return adjustment*( I-fit_function(p, x) )        
        
    def fit(self, x, y, initial_guess=None):
        
        if initial_guess==None:
            # Automated guess
            initial_guess = self.best_initial_guess( x, y )

        x = np.asarray(x)

        # Do the fit
        plsq = leastsq( self.residuals, initial_guess, args=(x, y) )

        self.fit_vals = plsq[0]

        return self.fit_vals

    def fit_line(self, xi=0.0, xf=1.0, num=200):
        
        xfit = np.linspace(xi, xf, num)
        yfit = []
        yfit = self.fit_function( self.fit_vals, xfit ) 
            
        return xfit, yfit


    def set_fit_vals(self, p):
        self.fit_vals = p
        
        
    def best_initial_guess(self, x, y ):
        
        b = x[0]
        m = (y[-1]-y[0])/(x[-1]-x[0])
        
        initial_guess = m, b
        
        return initial_guess


    def fit_string(self):
        
        m, b = self.fit_vals
        s = 'y = %.2fx + %.2f' % (m, b)
        
        return s

        
        
        
    # END OF: class Fitter(object)
    ############################################
        

        
# UnitCell    
###################################################################    
class FitterSphereSurfaceIntersect(Fitter):
    
    
    def fit_function(self, p, x):
        
        c, sigma, xc = p
        
        prefactor = c/( sigma*np.sqrt(2*np.pi) )
        I = prefactor*np.exp( -np.square(x-xc)/(2*(sigma**2)) )
        
        return I


    def best_initial_guess(self, x, y ):
        
        sigma = 0.5*np.std(x)
        xc = (x[-1]+x[0])/2.0 # Use midpoint
        #xc = np.average(x) # Use ~middle of x-values
        #xc = np.sum( x*y )/np.sum(y) # Use mean
        c = max(y)*np.sqrt(2*np.pi)*sigma
        
        initial_guess = c, sigma, xc
        
        return initial_guess
        
        
        
    # END OF: class FitterSphereSurfaceIntersect(Fitter)
    ############################################
        
             

    
# UnitCell    
###################################################################    
class UnitCell(object):
    
    def __init__(self, lattice_spacing_a=1.0, lattice_spacing_b=None, lattice_spacing_c=None, alpha=90.0, beta=None, gamma=None, sigma_D=0.01):

        self.lattice_spacing_a = lattice_spacing_a
        if lattice_spacing_b==None:
            self.lattice_spacing_b = lattice_spacing_a
        else:
            self.lattice_spacing_b = lattice_spacing_b
        if lattice_spacing_c==None:
            self.lattice_spacing_c = lattice_spacing_a
        else:
            self.lattice_spacing_c = lattice_spacing_c
            
        self.alpha = radians(alpha)
        if beta==None:
            self.beta = radians(alpha)
        else:
            self.beta = radians(beta)
        if gamma==None:
            self.gamma = radians(alpha)
        else:
            self.gamma = radians(gamma)
    
    
        self.sigma_D = sigma_D          # Lattice disorder
        
        self.rotation_matrix_exp = np.identity(3)

        
        
    # Realspace
    ############################################
    def v_abc(self, a, b, c):
        """Determines the position in realspace for the given vector,
        in multiples of the unitcell vectors."""
        
        # The 'unitcell' coordinate system assumes:
        #  a-axis lies along x-axis
        #  b-axis is in x-y plane
        #  c-axis is vertical (or at a tilt, depending on beta)
        
        # Convert from (unitcell) Cartesian to (unitcell) fractional coordinates
        reduced_volume = sqrt( 1 - (cos(self.alpha))**2 - (cos(self.beta))**2 - (cos(self.gamma))**2 + 2*cos(self.alpha)*cos(self.beta)*cos(self.gamma) )
        #volume = reduced_volume*self.lattice_spacing_a*self.lattice_spacing_b*self.lattice_spacing_c
        av = ( self.lattice_spacing_a , \
                0.0 , \
                0.0  )
        bv = ( self.lattice_spacing_b*cos(self.gamma) , \
                self.lattice_spacing_b*sin(self.gamma) , \
                0.0 )
        cv = ( self.lattice_spacing_c*cos(self.beta) , \
                self.lattice_spacing_c*( cos(self.alpha) - cos(self.beta)*cos(self.gamma) )/( sin(self.gamma) ) , \
                self.lattice_spacing_c*reduced_volume/( sin(self.gamma) ) )
                
        v = a*av + b*bv +c *cv
        v_len = sqrt( v[0]**2 + v[1]**2 + v[2]**2 )
        
        return (v_len, v)
    
    
    def v_abc_length(self, a, b, c):
        
        vlen, v_vector = self.v_abc(a,b,c)
        #vlen = sqrt( v_vector[0]**2 + v_vector[1]**2 + v_vector[2]**2 )
        
        return vlen

        
    def v_abc_exp(self, a, b, c):
        """Returns the realspace position for the vector, in the experimental coordinate
        system (which includes whatever rotations have been set)."""
        
        vlen, v_vector = self.v_abc(a,b,c)
        
        x, y, z = v_vector
        x, y, z = self.rotate_q_exp(x, y, z)
        
        
        return (vlen, (x, y, z) )

        
    def print_v_abc_exp(self, a, b, c):
        # Usage:
        # vlen, (x, y, z), angle_wrt_x, angle_wrt_z = .print_v_abc_exp( a, b, c )
        
        print( '(abc) = ( %d %d %d )' % (a,b,c) )
        vabc = self.v_abc_exp(a, b, c)
        vlen, (x, y, z) = vabc
        print( '\t|v| = %.2f \t\t (x,y,z) = (%.4f, %.4f, %.4f)' % (vlen, x, y, z) )

        # Angle w.r.t. x-axis (of vector projection into x-y plane):
        angle_wrt_x = np.degrees( np.arctan2( y, x ) )
        print( '\tangle_inplane = %.3f degrees' % (angle_wrt_x) )
        # Angle w.r.t. z-axis
        vxy = sqrt( x**2 + y**2 )
        angle_wrt_z = np.degrees( np.arctan2( vxy, z ) )
        print( '\tangle_vertical = %.3f degrees' % (angle_wrt_z) )
        
        return (vlen, (x, y, z), angle_wrt_x, angle_wrt_z )
        
        
    def get_unitcell_volume(self):
        """Returns volume of unitcell."""
        
        reduced_volume = sqrt( 1 - (cos(self.alpha))**2 - (cos(self.beta))**2 - (cos(self.gamma))**2 + 2*cos(self.alpha)*cos(self.beta)*cos(self.gamma) )
        volume = reduced_volume*self.lattice_spacing_a*self.lattice_spacing_b*self.lattice_spacing_c
        
        return volume


             
        
    
    # Reciprocal-space
    ############################################
    def iterate_over_hkl_compute(self, max_hkl=6):
        """Returns a sequence of hkl lattice peaks (reflections)."""
        
        # r will contain the return value, an array with rows that contain:
        # h, k, l, qhkl, qhkl_vector
        r = []
        
        for h in range(-max_hkl,max_hkl+1):
            for k in range(-max_hkl,max_hkl+1):
                for l in range(-max_hkl,max_hkl+1):
                    
                    # Don't put a reflection at origin
                    if not (h==0 and k==0 and l==0):
                        qhkl, qhkl_vector = self.q_hkl_exp(h,k,l)
                        r.append( [ h, k, l, qhkl, qhkl_vector ] )
        
        return r
        
        
    def iterate_over_hkl(self, max_hkl=6):
        
        return self.iterate_over_hkl_compute(max_hkl=max_hkl)
            
    
    
    def q_hkl_rectangular(self, h, k, l):
        """Determines the position in reciprocal space for the given reflection."""
        
        # NOTE: This is assuming cubic/rectangular only!
        qhkl_vector = ( 2*pi*h/(self.lattice_spacing_a), \
                        2*pi*k/(self.lattice_spacing_b), \
                        2*pi*l/(self.lattice_spacing_c) ) 
        qhkl = sqrt( qhkl_vector[0]**2 + qhkl_vector[1]**2 + qhkl_vector[2]**2 )
        
        return (qhkl, qhkl_vector)

        
    def q_hkl(self, h, k, l):
        """Determines the position in reciprocal space for the given reflection."""
        
        # The 'unitcell' coordinate system assumes:
        #  a-axis lies along x-axis
        #  b-axis is in x-y plane
        #  c-axis is vertical (or at a tilt, depending on beta)
        
        # Convert from (unitcell) Cartesian to (unitcell) fractional coordinates
        reduced_volume = sqrt( 1 - (cos(self.alpha))**2 - (cos(self.beta))**2 - (cos(self.gamma))**2 + 2*cos(self.alpha)*cos(self.beta)*cos(self.gamma) )
        #volume = reduced_volume*self.lattice_spacing_a*self.lattice_spacing_b*self.lattice_spacing_c
        a = ( self.lattice_spacing_a , \
                0.0 , \
                0.0  )
        b = ( self.lattice_spacing_b*cos(self.gamma) , \
                self.lattice_spacing_b*sin(self.gamma) , \
                0.0 )
        c = ( self.lattice_spacing_c*cos(self.beta) , \
                self.lattice_spacing_c*( cos(self.alpha) - cos(self.beta)*cos(self.gamma) )/( sin(self.gamma) ) , \
                self.lattice_spacing_c*reduced_volume/( sin(self.gamma) ) )
        
        # Compute (unitcell) reciprocal-space lattice vectors
        volume = np.dot( a, np.cross(b,c) )
        u = np.cross( b, c ) / volume # Along qx
        v = np.cross( c, a ) / volume # Along qy
        w = np.cross( a, b ) / volume # Along qz
        
        qhkl_vector = 2*pi*( h*u + k*v + l*w )
        qhkl = sqrt( qhkl_vector[0]**2 + qhkl_vector[1]**2 + qhkl_vector[2]**2 )
        
        return (qhkl, qhkl_vector)

                
        
    def q_hkl_length(self, h, k, l):
        
        qhkl, qhkl_vector = self.q_hkl(h,k,l)
        #qhkl = sqrt( qhkl_vector[0]**2 + qhkl_vector[1]**2 + qhkl_vector[2]**2 )
        
        return qhkl
        
    
    def q_hkl_exp(self, h, k, l):
        """Returns the q position for the unit cell in the experimental coordinate
        system."""
        
        qhkl, qhkl_vector = self.q_hkl(h,k,l)
        
        qx, qy, qz = qhkl_vector
        qx, qy, qz = self.rotate_q_exp(qx, qy, qz)
        
        
        return (qhkl, (qx, qy, qz) )


    def print_q_hkl_exp(self, h, k, l):
        print( '[hkl] = [ %d %d %d ]' % (h,k,l) )
        qhkl = self.q_hkl_exp(h, k, l)
        qtot, (qx, qy, qz) = qhkl
        qxy = sqrt( qx**2 + qy**2 )
        print( '\t|q%d%d%d| = %.2f \t\t (qx,qy,qz) = (%.4f, %.4f, %.4f)' % (h, k, l, qtot, qx, qy, qz) )
        print( '\t              \t\t (qxy,qz) = (%.4f, %.4f)' % (qxy, qz) )

        # Angle w.r.t. x-axis (of vector projection into x-y plane):
        angle_wrt_x = np.degrees( np.arctan2( qy, qx ) )
        print( '\tangle_inplane = %.3f degrees' % (angle_wrt_x) )
        # Angle w.r.t. z-axis
        angle_wrt_z = np.degrees( np.arctan2( qxy, qz ) )
        print( '\tangle_vertical = %.3f degrees' % (angle_wrt_z) )
        
        return (qhkl, (qx, qy, qz), qxy, angle_wrt_x, angle_wrt_z )

        
    def q_hkl_exp_inplane_powder(self, h, k, l):
        
        qhkl, qhkl_vector = self.q_hkl_exp(h,k,l)
        
        qz = qhkl_vector[2]
        qxy = sqrt( (qhkl_vector[0])**2 + (qhkl_vector[1])**2 )
        
        return qxy, qz

                
        
    # Orientation (angles, rotation, etc.)
    ############################################
    def set_exp_vertical_hkl(self, h, k, l ):
        """Reorients the unit cell such that the given (hkl) peak is pointing
        out-of-plane (vertical; along qz)."""
        
        # TODO
        pass

        
    def set_rotation_angles(self, eta=0.0, phi=0.0, theta=0.0):
        """Set rotations."""
            
        self.rotation_matrix_exp = self.rotation_elements( eta, phi, theta )
       
       
    def apply_rotation(self, eta=0.0, phi=0.0, theta=0.0):
        """Apply a rotation (adds to what has already been done)."""
            
        new_rotation_matrix = self.rotation_elements( eta, phi, theta )
        
        #self.rotation_matrix_exp = np.dot( self.rotation_matrix_exp , new_rotation_matrix )
        self.rotation_matrix_exp = np.dot( new_rotation_matrix, self.rotation_matrix_exp  )

        
    def apply_rotation_x(self, eta=0.0 ):
        """Apply a rotation (adds to what has already been done)."""
            
        eta = radians(eta)
        new_rotation_matrix =   [[  1 ,                 0 ,             0           ],
                                [   0  ,                +cos(eta) ,     -sin(eta)   ],
                                [   0 ,                 +sin(eta) ,     +cos(eta)   ]]        
        
        self.rotation_matrix_exp = np.dot( new_rotation_matrix, self.rotation_matrix_exp  )

        
    def apply_rotation_y(self, phi=0.0 ):
        """Apply a rotation (adds to what has already been done)."""
            
        phi = radians(phi)
        new_rotation_matrix =   [[  +cos(phi) ,         0 ,             +sin(phi)   ],
                                [   0  ,                1 ,             0           ],
                                [   -sin(phi) ,         0 ,             +cos(phi)   ]]        
        
        self.rotation_matrix_exp = np.dot( new_rotation_matrix, self.rotation_matrix_exp  )

    def apply_rotation_z(self, theta=0.0 ):
        """Apply a rotation (adds to what has already been done)."""
            
        theta = radians(theta)
        new_rotation_matrix =   [[  +cos(theta) ,       -sin(theta) ,     0         ],
                                [   +sin(theta) ,       +cos(theta) ,     0         ],
                                [   0 ,                 0 ,               1         ]]        
        
        self.rotation_matrix_exp = np.dot( new_rotation_matrix, self.rotation_matrix_exp  )
            
        

    def rotation_elements(self, eta, phi, theta):
        """Converts angles into an appropriate rotation matrix."""
        
        # Three-axis rotation:
        # 1. Rotate about +z by eta (follows RHR; rotation is mathematical and thus counter-clockwise)
        # 2. Tilt by phi with respect to +z (rotation about y-axis) then
        # 3. rotate by theta in-place (rotation about z-axis) ### BUG: This isn't a conceptual rotation about z (influenced by other rotations)
        

        eta = radians( eta )        # eta is orientation around the z axis (before reorientation)
        phi = radians( phi )        # phi is grain tilt (with respect to +z axis)
        theta = radians( theta )    # grain orientation (around the z axis)
        
        rotation_elements = [[  cos(eta)*cos(phi)*cos(theta)-sin(eta)*sin(theta) ,
                                    -cos(eta)*cos(phi)*sin(theta)-sin(eta)*cos(theta) ,
                                    -cos(eta)*sin(phi)                                   ],
                            [  sin(eta)*cos(phi)*cos(theta)+cos(eta)*sin(theta) ,
                                    -sin(eta)*cos(phi)*sin(theta)+cos(eta)*cos(theta) ,
                                    sin(eta)*sin(phi)                                    ],
                            [ -sin(phi)*cos(theta) ,
                                sin(phi)*sin(theta) ,
                                cos(phi)                                              ]]
        
        return rotation_elements

        
    def rotate_q_exp(self, qx, qy, qz):
        """Rotates the q-vector in the way given by the internal
        rotation_matrix, which should have been set using "set_angles"
        or the appropriate pargs (eta, phi, theta)."""
        # qx, qy, qz = self.rotate_q_exp(qx, qy, qz)
        
        q_vector = np.array( [[qx],[qy],[qz]] )
        
        q_rotated = np.dot( self.rotation_matrix_exp, q_vector )
        qx = q_rotated[0,0]
        qy = q_rotated[1,0]
        qz = q_rotated[2,0]
        
        return qx, qy, qz

        
        
    # Plotting
    ############################################
    def plot_exp_inplane_powder(self, filename='output.png', plot_region=[0.0, None, 0.0, None], plot_buffers=[0.16, 0.035, 0.16, 0.03], label_peaks=False, peaks_present=None, blanked_figure=False, max_hkl=10, thresh=1e-10):
        
        peaks_x = []
        peaks_y = []
        hkls = []
        for h, k, l, qhkl, qhkl_vector in self.iterate_over_hkl(max_hkl=max_hkl):
            qz = qhkl_vector[2]
            qxy = sqrt( (qhkl_vector[0])**2 + (qhkl_vector[1])**2 )
            
            if (plot_region[0]!=None and qxy>=plot_region[0]) and (plot_region[1]!=None and qxy<=plot_region[1]) and (plot_region[2]!=None and qz>=plot_region[2]) and (plot_region[3]!=None and qz<=plot_region[3]):
                
                if peaks_present==None or [abs(h), abs(k), abs(l)] in peaks_present:
                    peaks_x.append( qxy )
                    peaks_y.append( qz )
                    hkls.append( [qxy, qz, (h,k,l)] )
            
            
        if label_peaks:
            
            labels_x = []
            labels_y = []
            labels_s = []
            for qxy, qz, hkl in hkls:
                h, k, l = hkl
                if peaks_present==None or [abs(h), abs(k), abs(l)] in peaks_present:

                    # Create new label
                    if False:
                        s = r'$'
                        if h<0:
                            s += '\overline{'+str(abs(h))+'}'
                        else:
                            s += str(h)
                        if k<0:
                            s += '\overline{'+str(abs(k))+'}'
                        else:
                            s += str(k)
                        if l<0:
                            s += '\overline{'+str(abs(l))+'}'
                        else:
                            s += str(l)
                        s += r'$'
                    else:
                        #s = '%d%d%d' % (h, k, l)
                        s = '%d%d%d' % (abs(h), abs(k), abs(l))
                        #if (abs(h)+abs(k)+abs(l))%3==2:
                            #s = '%d%d%d' % (abs(h), abs(k), abs(l))
                        #else:
                            #s = ''
                    
                    # Check if this peak already exists
                    imatch = -1
                    for i in range(len(labels_x)):
                        if abs(qxy-labels_x[i])<thresh and abs(qz-labels_y[i])<thresh:
                            imatch = i
                            
                    if imatch==-1:
                        # New peak
                        labels_x.append(qxy)
                        labels_y.append(qz)
                        labels_s.append(s)
                    else:
                        # Existing peak
                        #labels_s[imatch] += ' ' + s
                        pass

                
            
            
            
        # Plot styling
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.labelsize'] = 30
        plt.rcParams['xtick.labelsize'] = 'xx-large'
        plt.rcParams['ytick.labelsize'] = 'xx-large'

        #plt.rcParams['axes.labelsize'] = 35
        #plt.rcParams['xtick.labelsize'] = 28
        #plt.rcParams['ytick.labelsize'] = 28


        fig = plt.figure(figsize=(7,7))
        #fig.subplots_adjust(left=0.17, bottom=0.15, right=0.97, top=0.94, wspace=0.2, hspace=0.2)
        #ax = plt.subplot(111)
        left_buf, right_buf, bottom_buf, top_buf = plot_buffers
        fig_width = 1.0-right_buf-left_buf
        fig_height = 1.0-top_buf-bottom_buf
        ax = fig.add_axes( [left_buf, bottom_buf, fig_width, fig_height], aspect='equal' )
        
        plt.scatter( peaks_x, peaks_y, s=80, facecolor='none', edgecolor=(0,1,0), linewidth=0.5 )
        if label_peaks:
            for x, y, s in zip( labels_x, labels_y, labels_s ):
                if blanked_figure:
                    plt.text( x, y, s, size=9, color='1.0', horizontalalignment='left', verticalalignment='bottom' )
                else:
                    plt.text( x, y, s, size=6, color='0.0', horizontalalignment='left', verticalalignment='bottom' )

        # Axis scaling
        xi, xf, yi, yf = ax.axis()
        if plot_region[1]==None and plot_region[3]!=None:
            xf = plot_region[3]
        if plot_region[3]==None and plot_region[1]!=None:
            yf = plot_region[1]
            
        if plot_region[0] != None: xi = plot_region[0]
        if plot_region[1] != None: xf = plot_region[1]
        if plot_region[2] != None: yi = plot_region[2]
        if plot_region[3] != None: yf = plot_region[3]
        ax.axis( [xi, xf, yi, yf] )
                
        if blanked_figure:
            plt.xticks( [] )
            plt.yticks( [] )
        else:
            plt.xlabel( r'$q_{xy} \, (\mathrm{\AA^{-1}})$', size=30 )
            plt.ylabel( r'$q_z \, (\mathrm{\AA^{-1}})$', size=30  )
        
        plt.savefig(filename, transparent=blanked_figure)
        #plt.show()
        plt.close()
        
        
        
    def plot_exp(self, filename='output.png', qy_plane=0.0, plot_region=[None, None, None, None], plot_buffers=[0.16, 0.035, 0.16, 0.03], label_peaks=False, max_hkl=10, thresh = 1e-7):
        
        
        
        peaks_x = []
        peaks_y = []
        names = []
        for h, k, l, qhkl, qhkl_vector in self.iterate_over_hkl(max_hkl=max_hkl):
            qx, qy, qz = qhkl_vector
            if abs(qy-qy_plane)<thresh:
                peaks_x.append( qx )
                peaks_y.append( qz )
                if label_peaks:
                    if True:
                        s = r'$'
                        if h<0:
                            s += '\overline{'+str(abs(h))+'}'
                        else:
                            s += str(h)
                        if k<0:
                            s += '\overline{'+str(abs(k))+'}'
                        else:
                            s += str(k)
                        if l<0:
                            s += '\overline{'+str(abs(l))+'}'
                        else:
                            s += str(l)
                        s += r'$'
                    else:
                        s = '%d%d%d' % (abs(h), abs(k), abs(l))
                    names.append( s )
            
            
            
        # Plot styling
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.labelsize'] = 30
        plt.rcParams['xtick.labelsize'] = 'xx-large'
        plt.rcParams['ytick.labelsize'] = 'xx-large'

        #plt.rcParams['axes.labelsize'] = 35
        #plt.rcParams['xtick.labelsize'] = 28
        #plt.rcParams['ytick.labelsize'] = 28


        fig = plt.figure(figsize=(8,8))
        #fig.subplots_adjust(left=0.17, bottom=0.15, right=0.97, top=0.94, wspace=0.2, hspace=0.2)
        #ax = plt.subplot(111)
        left_buf, right_buf, bottom_buf, top_buf = plot_buffers
        fig_width = 1.0-right_buf-left_buf
        fig_height = 1.0-top_buf-bottom_buf
        ax = fig.add_axes( [left_buf, bottom_buf, fig_width, fig_height], aspect='equal' )
        
        plt.scatter( peaks_x, peaks_y, s=80, facecolor='none', edgecolor='b' )
        if label_peaks:
            for x, y, s in zip( peaks_x, peaks_y, names ):
                plt.text( x, y, s, size=10, color='0.5', horizontalalignment='left', verticalalignment='bottom' )

        # Axis scaling
        xi, xf, yi, yf = ax.axis()
        if plot_region[0] != None: xi = plot_region[0]
        if plot_region[1] != None: xf = plot_region[1]
        if plot_region[2] != None: yi = plot_region[2]
        if plot_region[3] != None: yf = plot_region[3]
        if plot_region[0]==None and plot_region[1]==None and plot_region[2]==None and plot_region[3]==None:
            xf = max( xi, xf, yi, yf )
            yf = xf
            xi = -xf
            yi = -yf
        ax.axis( [xi, xf, yi, yf] )
                
        
        plt.xlabel( r'$q_{x} \, (\mathrm{\AA^{-1}})$', size=30 )
        plt.ylabel( r'$q_z \, (\mathrm{\AA^{-1}})$', size=30  )
        
        plt.savefig(filename)
        plt.close()

        
    def plot_ewald(self, ewaldsphere, filename='output.png', plot_region=[None, None, None, None], plot_buffers=[0.16, 0.035, 0.16, 0.03], label_peaks=False, blanked_figure=False, peaks_present=None, max_hkl=10, thresh=0.01, dpi=100):
        

        # Plot styling
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.labelsize'] = 30
        plt.rcParams['xtick.labelsize'] = 'xx-large'
        plt.rcParams['ytick.labelsize'] = 'xx-large'

        #plt.rcParams['axes.labelsize'] = 35
        #plt.rcParams['xtick.labelsize'] = 28
        #plt.rcParams['ytick.labelsize'] = 28


        fig = plt.figure(figsize=(10,10))
        #fig.subplots_adjust(left=0.17, bottom=0.15, right=0.97, top=0.94, wspace=0.2, hspace=0.2)
        #ax = plt.subplot(111)
        left_buf, right_buf, bottom_buf, top_buf = plot_buffers
        fig_width = 1.0-right_buf-left_buf
        fig_height = 1.0-top_buf-bottom_buf
        ax = fig.add_axes( [left_buf, bottom_buf, fig_width, fig_height], aspect='equal' )

        

        peaks_x = []
        peaks_y = []
        names = []
        
        q_rings = []
        
        for h, k, l, qhkl, qhkl_vector in self.iterate_over_hkl(max_hkl=max_hkl):
            qx, qy, qz = qhkl_vector
            qxy = np.sqrt(qx**2 + qy**2)
            
            # Make sure peaks appears on correct side of image
            if qx<0:
                qxy *= -1
                
            if (plot_region[0]!=None and qxy>=plot_region[0]) and (plot_region[1]!=None and qxy<=plot_region[1]) and (plot_region[2]!=None and qz>=plot_region[2]) and (plot_region[3]!=None and qz<=plot_region[3]):
                d = ewaldsphere.distance_to_ewald_surface(qx, qy, qz)
                if d<thresh:
                    if peaks_present==None or [abs(h), abs(k), abs(l)] in peaks_present:
                        peaks_x.append( qxy )
                        peaks_y.append( qz )
                        if label_peaks:
                            if False:
                                s = r'$'
                                if h<0:
                                    s += '\overline{'+str(abs(h))+'}'
                                else:
                                    s += str(h)
                                if k<0:
                                    s += '\overline{'+str(abs(k))+'}'
                                else:
                                    s += str(k)
                                if l<0:
                                    s += '\overline{'+str(abs(l))+'}'
                                else:
                                    s += str(l)
                                s += r'$'
                            else:
                                s = '%d%d%d' % (abs(h), abs(k), abs(l))
                            names.append( s )
                            
                            
                        xs = []
                        ys = []
                        for rot_angle in np.linspace(0, 2*np.pi, num=200):
                            qx_rot = qx*np.cos(rot_angle) + qz*np.sin(rot_angle)
                            qy_rot = qy
                            qz_rot = -qx*np.sin(rot_angle) + qz*np.cos(rot_angle)
                            qxy_rot = np.sqrt(np.square(qx_rot)+np.square(qy_rot))
                            if qx_rot<0:
                                qxy_rot *= -1
                            
                            xs.append( qxy_rot )
                            ys.append( qz_rot )
                        q_rings.append( [xs, ys] )
                
        
        plt.scatter( peaks_x, peaks_y, s=80, facecolor='none', edgecolor=(0,1,0), linewidth=1.5 )
        
        for ring in q_rings:
            x, y = ring
            plt.plot(x, y, '-', color='g', alpha=0.5)
        
        if label_peaks:
            for x, y, s in zip( peaks_x, peaks_y, names ):
                if blanked_figure:
                    plt.text( x, y, s, size=12, color='1.0', horizontalalignment='left', verticalalignment='bottom' )
                else:
                    plt.text( x, y, s, size=12, color='0.0', horizontalalignment='left', verticalalignment='bottom' )
                
                
                
        # Axis scaling
        xi, xf, yi, yf = ax.axis()
        if plot_region[0] != None: xi = plot_region[0]
        if plot_region[1] != None: xf = plot_region[1]
        if plot_region[2] != None: yi = plot_region[2]
        if plot_region[3] != None: yf = plot_region[3]
        if plot_region[0]==None and plot_region[1]==None and plot_region[2]==None and plot_region[3]==None:
            xf = max( xi, xf, yi, yf )
            yf = xf
            xi = -xf
            yi = -yf
            
            
        # Show central meridian of Ewald sphere
        qxys, qzs = ewaldsphere.central_meridian_arc()
        plt.plot( qxys, qzs, '-', color='0.5', linewidth=0.5 )
        plt.plot( -1*qxys, qzs, '-', color='0.5', linewidth=0.5 )
        
        
            
        ax.axis( [xi, xf, yi, yf] )
        
        if blanked_figure:
            plt.xticks( [] )
            plt.yticks( [] )
        else:
            plt.xlabel( r'$q_{xy} \, (\mathrm{\AA^{-1}})$', size=30 )
            plt.ylabel( r'$q_{z} \, (\mathrm{\AA^{-1}})$', size=30  )
            
        
        plt.savefig( filename, transparent=blanked_figure, dpi=dpi )        
        plt.close()


    def plot_ewald_qxqz(self, ewaldsphere, filename='output.png', plot_region=[None, None, None, None], plot_buffers=[0.16, 0.035, 0.16, 0.03], label_peaks=False, blanked_figure=False, peaks_present=None, max_hkl=10, thresh=0.01, dpi=100):
        

        # Plot styling
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.labelsize'] = 30
        plt.rcParams['xtick.labelsize'] = 'xx-large'
        plt.rcParams['ytick.labelsize'] = 'xx-large'

        #plt.rcParams['axes.labelsize'] = 35
        #plt.rcParams['xtick.labelsize'] = 28
        #plt.rcParams['ytick.labelsize'] = 28


        fig = plt.figure(figsize=(10,10))
        #fig.subplots_adjust(left=0.17, bottom=0.15, right=0.97, top=0.94, wspace=0.2, hspace=0.2)
        #ax = plt.subplot(111)
        left_buf, right_buf, bottom_buf, top_buf = plot_buffers
        fig_width = 1.0-right_buf-left_buf
        fig_height = 1.0-top_buf-bottom_buf
        ax = fig.add_axes( [left_buf, bottom_buf, fig_width, fig_height], aspect='equal' )

        

        peaks_x = []
        peaks_y = []
        names = []
        
        q_rings = []
        intersection_rings = []
        shell_rings = []
        
        for h, k, l, qhkl, qhkl_vector in self.iterate_over_hkl(max_hkl=max_hkl):
            qx, qy, qz = qhkl_vector
            qxy = np.sqrt(qx**2 + qy**2)
            q_total = np.sqrt(qx**2 + qy**2 +qz**2)
            
            # Make sure peaks appears on correct side of image
            if qx<0:
                qxy *= -1
                
            #if (plot_region[0]!=None and qxy>=plot_region[0]) and (plot_region[1]!=None and qxy<=plot_region[1]) and (plot_region[2]!=None and qz>=plot_region[2]) and (plot_region[3]!=None and qz<=plot_region[3]):
            if True:
                
                d = ewaldsphere.distance_to_ewald_surface(qx, qy, qz)
                
                if d<thresh:
                    #if peaks_present==None or [abs(h), abs(k), abs(l)] in peaks_present:
                    if peaks_present==None or [h,k,l] in peaks_present or [-h,-k,-l] in peaks_present:
                        peaks_x.append( qx )
                        peaks_y.append( qz )
                        if label_peaks:
                            if False:
                                s = r'$'
                                if h<0:
                                    s += '\overline{'+str(abs(h))+'}'
                                else:
                                    s += str(h)
                                if k<0:
                                    s += '\overline{'+str(abs(k))+'}'
                                else:
                                    s += str(k)
                                if l<0:
                                    s += '\overline{'+str(abs(l))+'}'
                                else:
                                    s += str(l)
                                s += r'$'
                            else:
                                s = '%d%d%d' % (abs(h), abs(k), abs(l))
                            names.append( s )
                            
                            
                            
                        # Put a q-ring for each peak
                        xs = []
                        ys = []
                        for rot_angle in np.linspace(0, 2*np.pi, num=200):
                            qx_rot = qx*np.cos(rot_angle) + qz*np.sin(rot_angle)
                            qy_rot = qy
                            qz_rot = -qx*np.sin(rot_angle) + qz*np.cos(rot_angle)
                            qxy_rot = np.sqrt(np.square(qx_rot)+np.square(qy_rot))
                            if qx_rot<0:
                                qxy_rot *= -1
                            
                            xs.append( qx_rot )
                            ys.append( qz_rot )
                        q_rings.append( [xs, ys] )
                        
                        intersection_rings.append( ewaldsphere.intersection_ring(q_total) )
                        
                #if peaks_present==None or [abs(h), abs(k), abs(l)] in peaks_present:
                if peaks_present==None or [h,k,l] in peaks_present or [-h,-k,-l] in peaks_present:
                #if [abs(h), abs(k), abs(l)]==[1,2,0]:
                #if [h, k, l]==[1,2,0]:
                    target = [qx, qy, qz]
                    shell_rings.append( ewaldsphere.intersection_q_shell(q_total, target) )
                
        
        
        
        #for ring in q_rings:
            #x, y = ring
            #plt.plot(x, y, '-', color='g', alpha=0.5)
        
        #for ring in intersection_rings:
            #qx, qy, qz = ring
            #plt.plot(qx, qz, '-', color='b', alpha=0.5)

        for ring in shell_rings:
            qx, qy, qz, d = ring
            for i in range(len(qx)):

                marker, linewidth, alpha = self.d_to_marker(d[i], thresh=1.5)
                #marker, linewidth, alpha = 80, 1.5, 0.5
                plt.scatter([qx[i]], [qz[i]], s=marker, facecolor='none', edgecolor=(0,1,0), alpha=alpha, linewidth=linewidth )
        
        plt.scatter( peaks_x, peaks_y, s=120, facecolor='none', edgecolor=(0,1,0), linewidth=2.5 )
        if label_peaks:
            for x, y, s in zip( peaks_x, peaks_y, names ):
                if blanked_figure:
                    plt.text( x, y, s, size=12, color='1.0', horizontalalignment='left', verticalalignment='bottom' )
                else:
                    plt.text( x, y, s, size=12, color='0.0', horizontalalignment='left', verticalalignment='bottom' )
                
                
                
        # Axis scaling
        xi, xf, yi, yf = ax.axis()
        if plot_region[0] != None: xi = plot_region[0]
        if plot_region[1] != None: xf = plot_region[1]
        if plot_region[2] != None: yi = plot_region[2]
        if plot_region[3] != None: yf = plot_region[3]
        if plot_region[0]==None and plot_region[1]==None and plot_region[2]==None and plot_region[3]==None:
            xf = max( xi, xf, yi, yf )
            yf = xf
            xi = -xf
            yi = -yf
            
            
        # Show central meridian of Ewald sphere
        #qxys, qzs = ewaldsphere.central_meridian_arc()
        #plt.plot( qxys, qzs, '-', color='0.5', linewidth=0.5 )
        #plt.plot( -1*qxys, qzs, '-', color='0.5', linewidth=0.5 )
        plt.plot( [0,0], [plot_region[2], plot_region[3]], '-', color='0.5', linewidth=0.5 )
        
        
            
        ax.axis( [xi, xf, yi, yf] )
        
        if blanked_figure:
            plt.xticks( [] )
            plt.yticks( [] )
        else:
            plt.xlabel( r'$q_{x} \, (\mathrm{\AA^{-1}})$', size=30 )
            plt.ylabel( r'$q_{z} \, (\mathrm{\AA^{-1}})$', size=30  )
            
        
        plt.savefig( filename, transparent=blanked_figure, dpi=dpi )        
        plt.close()

    def peak_list(self, ewaldsphere, peaks_present=None, plot_region=[None, None, None, None], max_hkl=10, thresh=0.01):
        # usage:
        #peaks_x, peaks_y, names, distances = self.peak_list(ewaldsphere, peaks_present=peaks_present, plot_region=plot_region, max_hkl=max_hkl, thresh=thresh)
        
        peaks_x = []
        peaks_y = []
        names = []
        distances = []
        for h, k, l, qhkl, qhkl_vector in self.iterate_over_hkl(max_hkl=max_hkl):
            qx, qy, qz = qhkl_vector
            qxy = np.sqrt(qx**2 + qy**2)
            
            # Make sure peaks appears on correct side of image
            if qx<0:
                qxy *= -1
                
            if (plot_region[0]!=None and qxy>=plot_region[0]) and (plot_region[1]!=None and qxy<=plot_region[1]) and (plot_region[2]!=None and qz>=plot_region[2]) and (plot_region[3]!=None and qz<=plot_region[3]):
                d = ewaldsphere.distance_to_ewald_surface(qx, qy, qz)
                if d<thresh:
                    if peaks_present==None or ([abs(h), abs(k), abs(l)] in peaks_present):
                        peaks_x.append( qxy )
                        peaks_y.append( qz )
                        distances.append( d )
                        if True:
                            s = r'$'
                            if h<0:
                                s += '\overline{'+str(abs(h))+'}'
                            else:
                                s += str(h)
                            if k<0:
                                s += '\overline{'+str(abs(k))+'}'
                            else:
                                s += str(k)
                            if l<0:
                                s += '\overline{'+str(abs(l))+'}'
                            else:
                                s += str(l)
                            s += r'$'
                        else:
                            s = '%d%d%d' % (abs(h), abs(k), abs(l))
                        names.append( s )
                        
        return peaks_x, peaks_y, names, distances
        
        
    def d_to_marker(self, d, thresh=0.01, marker_max=250, marker_min=40, lw_max=5.0, lw_min=1.0, alpha_max=0.9, alpha_min=0.2):
        # Usage:
        # size, linewidth, alpha = self.d_to_marker(d, thresh=thresh)
        
        split1 = 0.25
        split2 = 0.50
        
        
        if abs(d)>thresh:
            
            return marker_min*0.2, lw_min, 0
            #return marker_min*0.2, lw_min, alpha_min
            


        intensity = 1.0 - abs(d)/thresh
        if intensity<0:
            print('ERROR intensity < 0')
        if intensity>1.0:
            print('ERROR intensity > 1')
            
        if intensity>split2:
            
            extent = (intensity-split2)/(1-split2) # extent is a # from 0 to 1
            
            marker = marker_max
            linewidth = lw_min + (lw_max-lw_min)*extent
            alpha = alpha_max
            
        elif intensity>split1:
            
            extent = (intensity-split1)/(split2-split1) # extent is a # from 0 to 1

            marker = marker_min + (marker_max-marker_min)*extent
            linewidth = lw_min
            alpha = alpha_max
        
        else:

            extent = (intensity-0.0)/split1 # extent is a # from 0 to 1
            
            marker = marker_min
            linewidth = lw_min
            alpha = alpha_min + (alpha_max-alpha_min)*extent
            
        
        if alpha<alpha_min or alpha>alpha_max:
            print('ERROR alpha')
        if linewidth<lw_min or linewidth>lw_max:
            print('ERROR lw')
            
        return marker, linewidth, alpha

        
    def plot_ewald_peak_distances(self, ewaldsphere, filename='output.png', plot_region=[None, None, None, None], plot_buffers=[0.16, 0.035, 0.16, 0.03], label_peaks=False, blanked_figure=False, peaks_present=None, max_hkl=10, thresh=0.01):
        """Plots peak positions as seen experimentally (taking into account Ewald sphere). The peak markers are sized based on their distance from the Ewald sphere."""
        

        # Plot styling
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.labelsize'] = 30
        plt.rcParams['xtick.labelsize'] = 'xx-large'
        plt.rcParams['ytick.labelsize'] = 'xx-large'

        #plt.rcParams['axes.labelsize'] = 35
        #plt.rcParams['xtick.labelsize'] = 28
        #plt.rcParams['ytick.labelsize'] = 28


        fig = plt.figure(figsize=(7,7))
        #fig.subplots_adjust(left=0.17, bottom=0.15, right=0.97, top=0.94, wspace=0.2, hspace=0.2)
        #ax = plt.subplot(111)
        left_buf, right_buf, bottom_buf, top_buf = plot_buffers
        fig_width = 1.0-right_buf-left_buf
        fig_height = 1.0-top_buf-bottom_buf
        ax = fig.add_axes( [left_buf, bottom_buf, fig_width, fig_height], aspect='equal' )

        

        if True:
            # Symmetry-peaks

            self.apply_rotation_z( -120.0 )
            peaks_x, peaks_y, names, distances = self.peak_list(ewaldsphere, peaks_present=peaks_present, plot_region=plot_region, max_hkl=max_hkl, thresh=thresh)
            
            #plt.scatter( peaks_x, peaks_y, s=80, facecolor='none', edgecolor=(0.7,0.7,0), linewidth=1.5 ) # Yellow peaks
            for x, y, d in zip(peaks_x, peaks_y, distances):
                size, linewidth, alpha = self.d_to_marker(d, thresh=thresh)
                plt.scatter( x, y, s=size, facecolor='none', edgecolor=(0.7,0.7,0), linewidth=linewidth, alpha=alpha ) # Yellow peaks
            if label_peaks:
                for x, y, s in zip( peaks_x, peaks_y, names ):
                    plt.text( x, y, s, size=12, color='0.6', horizontalalignment='left', verticalalignment='bottom' )


            self.apply_rotation_z( +240.0 )
            peaks_x, peaks_y, names, distances = self.peak_list(ewaldsphere, peaks_present=peaks_present, plot_region=plot_region, max_hkl=max_hkl, thresh=thresh)
            
            #plt.scatter( peaks_x, peaks_y, s=80, facecolor='none', edgecolor=(0,0.7,0.7), linewidth=1.5 ) # Blue-green peaks
            for x, y, d in zip(peaks_x, peaks_y, distances):
                size, linewidth, alpha = self.d_to_marker(d, thresh=thresh)
                plt.scatter( x, y, s=size, facecolor='none', edgecolor=(0,0.7,0.7), linewidth=linewidth, alpha=alpha ) # Blue-green peaks
            
            if label_peaks:
                for x, y, s in zip( peaks_x, peaks_y, names ):
                    plt.text( x, y, s, size=12, color='0.6', horizontalalignment='left', verticalalignment='bottom' )

            self.apply_rotation_z( -120.0 )

            
            
        # Regular peaks
        peaks_x, peaks_y, names, distances = self.peak_list(ewaldsphere, peaks_present=peaks_present, plot_region=plot_region, max_hkl=max_hkl, thresh=thresh)
        
        #plt.scatter( peaks_x, peaks_y, s=80, facecolor='none', edgecolor=(0,1,0), linewidth=1.5 ) # Green peaks
        for x, y, d in zip(peaks_x, peaks_y, distances):
            size, linewidth, alpha = self.d_to_marker(d, thresh=thresh)
            plt.scatter( x, y, s=size, facecolor='none', edgecolor=(0,1,0), linewidth=linewidth, alpha=alpha ) # Green peaks
        
        if label_peaks:
            for x, y, s in zip( peaks_x, peaks_y, names ):
                if blanked_figure:
                    plt.text( x, y, s, size=12, color='1.0', horizontalalignment='left', verticalalignment='bottom' )
                else:
                    plt.text( x, y, s, size=12, color='0.0', horizontalalignment='left', verticalalignment='bottom' )
                
                
                
        # Axis scaling
        xi, xf, yi, yf = ax.axis()
        if plot_region[0] != None: xi = plot_region[0]
        if plot_region[1] != None: xf = plot_region[1]
        if plot_region[2] != None: yi = plot_region[2]
        if plot_region[3] != None: yf = plot_region[3]
        if plot_region[0]==None and plot_region[1]==None and plot_region[2]==None and plot_region[3]==None:
            xf = max( xi, xf, yi, yf )
            yf = xf
            xi = -xf
            yi = -yf
            
            
        # Show central meridian of Ewald sphere
        qxys, qzs = ewaldsphere.central_meridian_arc()
        plt.plot( qxys, qzs, '-', color='0.5', linewidth=0.5 )
        plt.plot( -1*qxys, qzs, '-', color='0.5', linewidth=0.5 )
        
        
            
        ax.axis( [xi, xf, yi, yf] )
        
        if blanked_figure:
            plt.xticks( [] )
            plt.yticks( [] )
        else:
            plt.xlabel( r'$q_{xy} \, (\mathrm{\AA^{-1}})$', size=30 )
            plt.ylabel( r'$q_{z} \, (\mathrm{\AA^{-1}})$', size=30  )
            
        
        plt.savefig( filename, transparent=blanked_figure )     
        plt.close()

           

    def plot_ewald_two_beam(self, ewaldsphere, Material_ambient, Material_film, Material_substrate, filename='output.png', plot_region=[None, None, None, None], plot_buffers=[0.16, 0.035, 0.16, 0.03], label_peaks=False, blanked_figure=False, peaks_present=None, max_hkl=10, thresh=0.01):
        
        
        # Prepare for refraction correction computations
        k_xray =e.get_k() 
        
        ambient_n = np.real( Material_ambient.get_xray_n(energy=e.get_beam_energy()) )
        film_n = np.real( Material_film.get_xray_n(energy=e.get_beam_energy()) )
        substrate_n = np.real( Material_substrate.get_xray_n(energy=e.get_beam_energy()) )
        
        film_crit_rad = np.radians( Material_film.get_xray_critical_angle(energy=e.get_beam_energy()) )
        substrate_crit_rad = np.radians( Material_substrate.get_xray_critical_angle(energy=e.get_beam_energy()) )

        alpha_incident_rad = np.radians(e.get_theta_incident())
        alpha_incident_effective_rad = np.arccos( cos(alpha_incident_rad)*ambient_n/film_n ) # Snell's law (cosine form)
        
        
        
        
        
        

        # Plot styling
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.labelsize'] = 30
        plt.rcParams['xtick.labelsize'] = 'xx-large'
        plt.rcParams['ytick.labelsize'] = 'xx-large'

        #plt.rcParams['axes.labelsize'] = 35
        #plt.rcParams['xtick.labelsize'] = 28
        #plt.rcParams['ytick.labelsize'] = 28


        fig = plt.figure(figsize=(7,7))
        #fig.subplots_adjust(left=0.17, bottom=0.15, right=0.97, top=0.94, wspace=0.2, hspace=0.2)
        #ax = plt.subplot(111)
        left_buf, right_buf, bottom_buf, top_buf = plot_buffers
        fig_width = 1.0-right_buf-left_buf
        fig_height = 1.0-top_buf-bottom_buf
        ax = fig.add_axes( [left_buf, bottom_buf, fig_width, fig_height], aspect='equal' )

        

        # BA term
        peaks_x = []
        peaks_y = []
        names = []
        
        # Reflected-beam scattering
        peaksTR_x = []
        peaksTR_y = []
        namesTR = []

        # Scattering is reflected
        #peaksRT_x = []
        #peaksRT_y = []
        #namesRT = []

        # Beam is reflected, scattering is reflected
        #peaksRR_x = []
        #peaksRR_y = []
        #namesRR = []
        
        for h, k, l, qhkl, qhkl_vector in self.iterate_over_hkl(max_hkl=max_hkl):
            qx, qy, qz = qhkl_vector
            qxy = np.sqrt(qx**2 + qy**2)
            
            # Make sure peaks appears on correct side of image
            if qx<0:
                qxy *= -1
                
            #if (plot_region[0]!=None and qxy>=plot_region[0]) and (plot_region[1]!=None and qxy<=plot_region[1]) and (plot_region[2]!=None and qz>=plot_region[2]) and (plot_region[3]!=None and qz<=plot_region[3]):
            #if (plot_region[0]!=None and qxy>=plot_region[0]) and (plot_region[1]!=None and qxy<=plot_region[1]) and (plot_region[3]!=None and qz<=plot_region[3]):
            if True:
                d = ewaldsphere.distance_to_ewald_surface(qx, qy, qz)
                if d<thresh:
                    if peaks_present==None or [abs(h), abs(k), abs(l)] in peaks_present:
                        # This peak will be plotted; now compute the refraction correction for it
                        qzorig = qz
                        

                        # Shift of direct beam (due to film-air interface)
                        direct_beam_angle_shift_rad = (alpha_incident_rad-alpha_incident_effective_rad)

                        # Scattering event occurs inside film, with angle 2theta_B
                        two_theta_B_rad = 2.0*np.arcsin(qz/(2.0*k_xray))
                        
                        # Scattered ray refracts as it exits
                        scattered_incident_angle_rad = two_theta_B_rad - alpha_incident_effective_rad
                        scattered_exit_angle_rad = np.arccos( cos(scattered_incident_angle_rad)*film_n/ambient_n ) # Snell's law (cosine form)
                        
                        if (scattered_incident_angle_rad) > 0:
                            # GISAXS (scattering above horizon)
                            
                            if True:
                                # Direct computation of refraction-induced angle shifts
                                # Inspired by Figure 3 in:
                                # Breiby et al.  J. Appl. Cryst. (2008). 41, 262–271 doi: 10.1107/S0021889808001064
                                scatter_beam_angle_shift_rad = (scattered_exit_angle_rad-scattered_incident_angle_rad)
                                #two_theta_B_final_rad = two_theta_B_rad + direct_beam_angle_shift_rad + scatter_beam_angle_shift_rad
                                two_theta_B_final_rad = alpha_incident_rad + scattered_exit_angle_rad
                                qz = 2*k_xray*sin(two_theta_B_final_rad/2.0)
                            
                            if False:
                                # Equation 2 in:
                                # Xinhui Lu et al. 
                                qz_in_film = k_xray*( 
                                                np.sqrt( (sin(alpha_incident_rad))**2 - (sin(film_crit_rad))**2 ) 
                                                + np.sqrt( ( (qz/k_xray)-sin(alpha_incident_rad) )**2 - (sin(film_crit_rad))**2 )
                                                )
                                qz_shift = qz-qz_in_film
                                qz += qz_shift

                            if False:
                                # Equation 22 in:
                                # Byeongdu Lee Macromolecules 2005, 38, 4311 doi: 10.1021/ma047562d 
                                scattered_exit_angle_rad = np.arccos( np.sqrt( film_n**2 -((qz/k_xray) + np.sqrt(film_n**2 - (cos(alpha_incident_rad))**2 ) )**2 ) )
                                two_theta_B_final_rad = alpha_incident_rad + scattered_exit_angle_rad
                                qz = 2*k_xray*sin(two_theta_B_final_rad/2.0)

                            if False:
                                # Equation 26 in:
                                # Busch et al. J. Appl. Cryst. J. Appl. Cryst. (2006). 39, 433–442 doi:10.1107/S0021889806012337
                                qz = k_xray*( 
                                        - sin(alpha_incident_rad) 
                                        - np.sqrt( 
                                                (sin(film_crit_rad)**2) 
                                                + ( (qz/k_xray) - ( sin(alpha_incident_rad)**2 - sin(film_crit_rad)**2 )**2 )
                                                )
                                        )
 
                            
                        else:
                            # GTSAXS (scattering below horizon)
                            
                            if True:
                                # Direct computation of refraction-induced angle shifts
                                substrate_incident_effective_rad = alpha_incident_effective_rad - two_theta_B_rad
                                if cos(substrate_incident_effective_rad)*film_n/substrate_n > 1.0:
                                    # Scattering ray intersects substrate below its critical angle
                                    #substrate_exit_angle_rad = 0.0 # peak appears along horizon
                                    substrate_exit_angle_rad = np.nan # peak is not seen (total reflection)
                                else:
                                    substrate_exit_angle_rad = np.arccos( cos(substrate_incident_effective_rad)*film_n/substrate_n ) # Snell's law (cosine form)
                                
                                scatter_beam_angle_shift_rad = substrate_incident_effective_rad - substrate_exit_angle_rad
                                two_theta_B_final_rad = two_theta_B_rad + direct_beam_angle_shift_rad + scatter_beam_angle_shift_rad
                                qz = 2*k_xray*sin(two_theta_B_final_rad/2.0)

                            if False:
                                # Xinhui Lu et al. 
                                qz_in_film = k_xray*( 
                                                np.sqrt( (sin(alpha_incident_rad))**2 - (sin(film_crit_rad))**2 ) 
                                                - np.sqrt( 
                                                    ( (qz/k_xray)-sin(alpha_incident_rad) )**2 
                                                    + (sin(substrate_crit_rad))**2
                                                    - (sin(film_crit_rad))**2 
                                                    )
                                                )
                                qz_shift = qz-qz_in_film
                                qz += qz_shift
                                

                            
                        if np.isnan(qz):
                            pass
                        elif (plot_region[0]!=None and qxy>=plot_region[0]) and (plot_region[1]!=None and qxy<=plot_region[1]) and (plot_region[2]!=None and qz>=plot_region[2]) and (plot_region[3]!=None and qz<=plot_region[3]):
                            peaks_x.append( qxy )
                            peaks_y.append( qz )
                            if label_peaks:
                                if True:
                                    s = r'$'
                                    if h<0:
                                        s += '\overline{'+str(abs(h))+'}'
                                    else:
                                        s += str(h)
                                    if k<0:
                                        s += '\overline{'+str(abs(k))+'}'
                                    else:
                                        s += str(k)
                                    if l<0:
                                        s += '\overline{'+str(abs(l))+'}'
                                    else:
                                        s += str(l)
                                    s += r'$'
                                else:
                                    s = '%d%d%d' % (abs(h), abs(k), abs(l))
                                names.append( s )

                                
                                
                                
                        # Plot peaks from reflected beam
                        qz = qzorig
                        
                        # Specular reflection inside the film (reflection from film-substrate interface)
                        specular_beam_in_film_rad = alpha_incident_rad + alpha_incident_effective_rad

                        # Scattering event occurs inside film, with angle 2theta_B
                        two_theta_B_rad = 2.0*np.arcsin(qz/(2.0*k_xray))
                        
                        
                        # Scattered ray refracts as it exits
                        #scattered_incident_angle_rad = two_theta_B_rad + specular_beam_in_film_rad # WRONG
                        scattered_incident_angle_rad = alpha_incident_effective_rad + two_theta_B_rad
                        scattered_exit_angle_rad = np.arccos( cos(scattered_incident_angle_rad)*film_n/ambient_n ) # Snell's law (cosine form)
                        
                        if (scattered_incident_angle_rad) > 0:
                            # GISAXS (scattering above horizon)
                            
                            # Direct computation of refraction-induced angle shifts
                            # Inspired by Figure 3 in:
                            # Breiby et al.  J. Appl. Cryst. (2008). 41, 262–271 doi: 10.1107/S0021889808001064
                            scatter_beam_angle_shift_rad = (scattered_exit_angle_rad-scattered_incident_angle_rad)
                            #two_theta_B_final_rad = two_theta_B_rad + direct_beam_angle_shift_rad + scatter_beam_angle_shift_rad
                            two_theta_B_final_rad = alpha_incident_rad + scattered_exit_angle_rad
                            qz = 2*k_xray*sin(two_theta_B_final_rad/2.0)
                            
                            
                        else:
                            # GTSAXS (scattering below horizon)
                            
                            scattered_incident_angle_rad = -alpha_incident_effective_rad + abs(two_theta_B_rad)
                            if cos(scattered_incident_angle_rad)*film_n/substrate_n > 1.0:
                                # Scattering ray intersects substrate below its critical angle
                                #substrate_exit_angle_rad = 0.0 # peak appears along horizon
                                substrate_exit_angle_rad = np.nan # peak is not seen (total reflection)
                            else:
                                scattered_exit_angle_rad = np.arccos( cos(scattered_incident_angle_rad)*film_n/substrate_n ) # Snell's law (cosine form)
                            
                            two_theta_B_final_rad = alpha_incident_rad - scattered_exit_angle_rad
                            qz = 2*k_xray*sin(two_theta_B_final_rad/2.0)
                            

                        if np.isnan(qz):
                            pass
                        elif (plot_region[0]!=None and qxy>=plot_region[0]) and (plot_region[1]!=None and qxy<=plot_region[1]) and (plot_region[2]!=None and qz>=plot_region[2]) and (plot_region[3]!=None and qz<=plot_region[3]):
                            peaksTR_x.append( qxy )
                            peaksTR_y.append( qz )
                            if label_peaks:
                                if True:
                                    s = r'$'
                                    if h<0:
                                        s += '\overline{'+str(abs(h))+'}'
                                    else:
                                        s += str(h)
                                    if k<0:
                                        s += '\overline{'+str(abs(k))+'}'
                                    else:
                                        s += str(k)
                                    if l<0:
                                        s += '\overline{'+str(abs(l))+'}'
                                    else:
                                        s += str(l)
                                    s += r'$'
                                else:
                                    s = '%d%d%d' % (abs(h), abs(k), abs(l))
                                namesTR.append( s )
                        
                        
        plt.scatter( peaks_x, peaks_y, s=80, facecolor='none', edgecolor=(0,1,0), linewidth=1.5 )
        plt.scatter( peaksTR_x, peaksTR_y, s=80, facecolor='none', edgecolor=(1,1,0), linewidth=1.5 )
        if label_peaks:
            for x, y, s in zip( peaks_x, peaks_y, names ):
                if blanked_figure:
                    plt.text( x, y, s, size=12, color='1.0', horizontalalignment='left', verticalalignment='bottom' )
                else:
                    plt.text( x, y, s, size=12, color='0.0', horizontalalignment='left', verticalalignment='bottom' )
            for x, y, s in zip( peaksTR_x, peaksTR_y, namesTR ):
                if blanked_figure:
                    plt.text( x, y, s, size=12, color='1.0', horizontalalignment='left', verticalalignment='top' )
                else:
                    plt.text( x, y, s, size=12, color='0.0', horizontalalignment='left', verticalalignment='top' )
                
                
        if True:
            # Show various guides in the figure
            
            # Direct beam
            qz = 0
            plt.scatter( [0], [qz], s=120, facecolor='none', edgecolor='0.75', linewidth=1.5 )
            plt.axhline( qz, color='0.75' )
            
            # Refracted beam
            alpha_exit_direct_rad = np.arccos( cos(alpha_incident_effective_rad)*film_n/substrate_n )
            qz = 2*k_xray*sin( (alpha_incident_rad-alpha_exit_direct_rad)/2.0)
            plt.scatter( [0], [qz], s=120, facecolor='none', edgecolor='r', linewidth=1.5 )
            plt.axhline( qz, color='r' )
            
            # Horizon
            qz = 2*k_xray*sin(alpha_incident_rad/2.0)
            l = plt.axhline( qz, color='0.5' )
            l.set_dashes( [10,3] )
            
            # Yoneda
            yoneda_rad = alpha_incident_rad + film_crit_rad
            qz = 2*k_xray*sin(yoneda_rad/2.0)
            plt.axhline( qz, color='orange', linewidth=3.0 )
            
            # Specular beam
            qz = 2*k_xray*sin( (2.0*alpha_incident_rad)/2.0)
            plt.scatter( [0], [qz], s=120, facecolor='none', edgecolor='r', linewidth=1.5 )
            plt.axhline( qz, color='r' )
                
            # Show central meridian of Ewald sphere
            qxys, qzs = ewaldsphere.central_meridian_arc()
            plt.plot( qxys, qzs, '-', color='0.5', linewidth=0.5 )
            plt.plot( -1*qxys, qzs, '-', color='0.5', linewidth=0.5 )

                
        # Axis scaling
        xi, xf, yi, yf = ax.axis()
        if plot_region[0] != None: xi = plot_region[0]
        if plot_region[1] != None: xf = plot_region[1]
        if plot_region[2] != None: yi = plot_region[2]
        if plot_region[3] != None: yf = plot_region[3]
        if plot_region[0]==None and plot_region[1]==None and plot_region[2]==None and plot_region[3]==None:
            xf = max( xi, xf, yi, yf )
            yf = xf
            xi = -xf
            yi = -yf
            
            
        
        
            
        ax.axis( [xi, xf, yi, yf] )
        
        if blanked_figure:
            plt.xticks( [] )
            plt.yticks( [] )
        else:
            plt.xlabel( r'$q_{xy} \, (\mathrm{\AA^{-1}})$', size=30 )
            plt.ylabel( r'$q_{z} \, (\mathrm{\AA^{-1}})$', size=30  )
            
        
        plt.savefig( filename, transparent=blanked_figure )        
        plt.close()
        
        
        
    def compute_DWBA_corrections(self, qz, ewaldsphere, Material_ambient, Material_film, Material_substrate):
        # Usage:
        #qzDD, qzRD, qzDR, qzRR = compute_DWBA_corrections(qz, ewaldsphere, Material_ambient, Material_film, Material_substrate)
        # Outputs:
        # qzDD: Scattering from direct (refracted) beam (Born approximation)
        # qzRD: Scattering from substrate-reflected beam
        # qzDR: Scattering from direct (refracted) beam reflects from substrate
        # qzRR: Scattering from substrate-reflected beam reflects from substrate 
        
        
        qzDD = np.nan # Born approximation
        qzRD = np.nan # Scattering from reflected beam
        qzDR = np.nan # Scattering is reflected
        qzRR = np.nan # Scattering from reflected beam is also reflected

        # Prepare for refraction correction computations
        k_xray =e.get_k() 
        
        ambient_n = np.real( Material_ambient.get_xray_n(energy=e.get_beam_energy()) )
        film_n = np.real( Material_film.get_xray_n(energy=e.get_beam_energy()) )
        substrate_n = np.real( Material_substrate.get_xray_n(energy=e.get_beam_energy()) )
        
        film_crit_rad = np.radians( Material_film.get_xray_critical_angle(energy=e.get_beam_energy()) )
        substrate_crit_rad = np.radians( Material_substrate.get_xray_critical_angle(energy=e.get_beam_energy()) )

        alpha_incident_rad = np.radians(e.get_theta_incident())
        if cos(alpha_incident_rad)*ambient_n/film_n > 1.0:
            alpha_incident_effective_rad = 0.0
        else:
            alpha_incident_effective_rad = np.arccos( cos(alpha_incident_rad)*ambient_n/film_n ) # Snell's law (cosine form)

        # Shift of direct beam (due to film-air interface)
        direct_beam_angle_shift_rad = (alpha_incident_rad-alpha_incident_effective_rad)

        # Scattering event occurs inside film, with angle 2theta_B
        two_theta_B_rad = 2.0*np.arcsin(qz/(2.0*k_xray))

        

        # qzDD: Scattering from direct (refracted) beam (Born approximation)
        ########################################
        
        # Scattered ray refracts as it exits
        scattered_incident_angle_rad = two_theta_B_rad - alpha_incident_effective_rad
        scattered_exit_angle_rad = np.arccos( cos(scattered_incident_angle_rad)*film_n/ambient_n ) # Snell's law (cosine form)
        
        if (scattered_incident_angle_rad) > 0:
            # GISAXS (scattering above horizon)
            
            # Direct computation of refraction-induced angle shifts
            # Inspired by Figure 3 in:
            # Breiby et al.  J. Appl. Cryst. (2008). 41, 262–271 doi: 10.1107/S0021889808001064
            scatter_beam_angle_shift_rad = (scattered_exit_angle_rad-scattered_incident_angle_rad)
            #two_theta_B_final_rad = two_theta_B_rad + direct_beam_angle_shift_rad + scatter_beam_angle_shift_rad
            two_theta_B_final_rad = alpha_incident_rad + scattered_exit_angle_rad
            qzDD = 2*k_xray*sin(two_theta_B_final_rad/2.0)
            
        else:
            # GTSAXS (scattering below horizon)
            
            # Direct computation of refraction-induced angle shifts
            substrate_incident_effective_rad = alpha_incident_effective_rad - two_theta_B_rad
            if cos(substrate_incident_effective_rad)*film_n/substrate_n > 1.0:
                # Scattering ray intersects substrate below its critical angle
                #substrate_exit_angle_rad = 0.0 # peak appears along horizon
                substrate_exit_angle_rad = np.nan # peak is not seen (total reflection)
            else:
                substrate_exit_angle_rad = np.arccos( cos(substrate_incident_effective_rad)*film_n/substrate_n ) # Snell's law (cosine form)
            
            scatter_beam_angle_shift_rad = substrate_incident_effective_rad - substrate_exit_angle_rad
            two_theta_B_final_rad = two_theta_B_rad + direct_beam_angle_shift_rad + scatter_beam_angle_shift_rad
            qzDD = 2*k_xray*sin(two_theta_B_final_rad/2.0)
        

        
        # qzDR: Scattering from direct (refracted) beam reflects from substrate
        ########################################
        
        if (scattered_incident_angle_rad) < 0:
            # GISAXS (Scattering hits film-substrate interface, reflects, and exits above horizon)
            
            scattered_exit_angle_rad = np.arccos( cos(abs(scattered_incident_angle_rad))*film_n/ambient_n ) # Snell's law (cosine form)
            
            two_theta_B_final_rad = alpha_incident_rad + scattered_exit_angle_rad
            qzDR = 2*k_xray*sin(two_theta_B_final_rad/2.0)
            
            
        
        
        
        
        # qzRD: Scattering from substrate-reflected beam
        ########################################
        
        # Specular reflection inside the film (reflection from film-substrate interface)
        specular_beam_in_film_rad = alpha_incident_rad + alpha_incident_effective_rad

        # Scattering event occurs inside film, with angle 2theta_B
        two_theta_B_rad = 2.0*np.arcsin(qz/(2.0*k_xray))
        
        
        # Scattered ray refracts as it exits
        #scattered_incident_angle_rad = two_theta_B_rad + specular_beam_in_film_rad # WRONG
        scattered_incident_angle_rad = alpha_incident_effective_rad + two_theta_B_rad
        scattered_exit_angle_rad = np.arccos( cos(scattered_incident_angle_rad)*film_n/ambient_n ) # Snell's law (cosine form)
        
        if (scattered_incident_angle_rad) > 0:
            # GISAXS (scattering above horizon)
            
            # Direct computation of refraction-induced angle shifts
            # Inspired by Figure 3 in:
            # Breiby et al.  J. Appl. Cryst. (2008). 41, 262–271 doi: 10.1107/S0021889808001064
            scatter_beam_angle_shift_rad = (scattered_exit_angle_rad-scattered_incident_angle_rad)
            #two_theta_B_final_rad = two_theta_B_rad + direct_beam_angle_shift_rad + scatter_beam_angle_shift_rad
            two_theta_B_final_rad = alpha_incident_rad + scattered_exit_angle_rad
            qzRD = 2*k_xray*sin(two_theta_B_final_rad/2.0)
            
            
        else:
            # GTSAXS (scattering below horizon)
            
            scattered_incident_angle_rad = -alpha_incident_effective_rad + abs(two_theta_B_rad)
            if cos(scattered_incident_angle_rad)*film_n/substrate_n > 1.0:
                # Scattering ray intersects substrate below its critical angle
                #substrate_exit_angle_rad = 0.0 # peak appears along horizon
                substrate_exit_angle_rad = np.nan # peak is not seen (total reflection)
            else:
                scattered_exit_angle_rad = np.arccos( cos(scattered_incident_angle_rad)*film_n/substrate_n ) # Snell's law (cosine form)
            
            two_theta_B_final_rad = alpha_incident_rad - scattered_exit_angle_rad
            qzRD = 2*k_xray*sin(two_theta_B_final_rad/2.0)
        
        
        
        
        
        # qzRR: Scattering from substrate-reflected beam reflects from substrate 
        ########################################

        # Scattered ray refracts as it exits
        scattered_incident_angle_rad = alpha_incident_effective_rad + two_theta_B_rad
        
        if (scattered_incident_angle_rad) < 0:
            # GISAXS (Scattering hits film-substrate interface, reflects, and exits above horizon)
        
            scattered_exit_angle_rad = np.arccos( cos(abs(scattered_incident_angle_rad))*film_n/ambient_n ) # Snell's law (cosine form)
            
            two_theta_B_final_rad = alpha_incident_rad + scattered_exit_angle_rad
            qzRR = 2*k_xray*sin(two_theta_B_final_rad/2.0)
        
        
        return qzDD, qzRD, qzDR, qzRR
        
        
        
    def append_to_peak_list(self, qxy, qz, h, k, l, peaks_x, peaks_y, names, plot_region=[None, None, None, None]):
        
        if np.isnan(qz):
            pass
        elif (plot_region[0]!=None and qxy>=plot_region[0]) and (plot_region[1]!=None and qxy<=plot_region[1]) and (plot_region[2]!=None and qz>=plot_region[2]) and (plot_region[3]!=None and qz<=plot_region[3]):
            peaks_x.append( qxy )
            peaks_y.append( qz )
            if True:
                s = r'$'
                if h<0:
                    s += '\overline{'+str(abs(h))+'}'
                else:
                    s += str(h)
                if k<0:
                    s += '\overline{'+str(abs(k))+'}'
                else:
                    s += str(k)
                if l<0:
                    s += '\overline{'+str(abs(l))+'}'
                else:
                    s += str(l)
                s += r'$'
            else:
                s = '%d%d%d' % (abs(h), abs(k), abs(l))
            names.append( s )

                        
    
    def plot_ewald_DWBA(self, ewaldsphere, Material_ambient, Material_film, Material_substrate, filename='output.png', plot_region=[None, None, None, None], plot_buffers=[0.16, 0.035, 0.16, 0.03], label_peaks=False, blanked_figure=False, peaks_present=None, max_hkl=10, thresh=0.01):
        


        # Plot styling
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.labelsize'] = 30
        plt.rcParams['xtick.labelsize'] = 'xx-large'
        plt.rcParams['ytick.labelsize'] = 'xx-large'
        #plt.rcParams['axes.labelsize'] = 35
        #plt.rcParams['xtick.labelsize'] = 28
        #plt.rcParams['ytick.labelsize'] = 28


        fig = plt.figure(figsize=(7,7))
        #fig.subplots_adjust(left=0.17, bottom=0.15, right=0.97, top=0.94, wspace=0.2, hspace=0.2)
        #ax = plt.subplot(111)
        left_buf, right_buf, bottom_buf, top_buf = plot_buffers
        fig_width = 1.0-right_buf-left_buf
        fig_height = 1.0-top_buf-bottom_buf
        ax = fig.add_axes( [left_buf, bottom_buf, fig_width, fig_height], aspect='equal' )

        

        # BA term
        peaksDD_x = []
        peaksDD_y = []
        namesDD = []
        
        # Reflected-beam scattering
        peaksRD_x = []
        peaksRD_y = []
        namesRD = []

        # Scattering is reflected
        peaksDR_x = []
        peaksDR_y = []
        namesDR = []

        # Beam is reflected, scattering is reflected
        peaksRR_x = []
        peaksRR_y = []
        namesRR = []
        
        
        
        for h, k, l, qhkl, qhkl_vector in self.iterate_over_hkl(max_hkl=max_hkl):
            qx, qy, qz = qhkl_vector
            qzorig = qz
            qxy = np.sqrt(qx**2 + qy**2)
            
            # Make sure peaks appears on correct side of image
            if qx<0:
                qxy *= -1
                
                
            # TODO: Consider each refraction-corrected peak separately
            d = ewaldsphere.distance_to_ewald_surface(qx, qy, qz)
            if d<thresh:
                if peaks_present==None or [abs(h), abs(k), abs(l)] in peaks_present:
                    
                    # This peak will be plotted; now compute the refraction correction for it
                    qzDD, qzRD, qzDR, qzRR = self.compute_DWBA_corrections(qz, ewaldsphere, Material_ambient, Material_film, Material_substrate)
                        
                    self.append_to_peak_list(qxy, qzDD, (h, k, l), peaksDD_x, peaksDD_y, namesDD, plot_region=plot_region)
                    self.append_to_peak_list(qxy, qzRD, (h, k, l), peaksRD_x, peaksRD_y, namesRD, plot_region=plot_region)
                    self.append_to_peak_list(qxy, qzDR, (h, k, l), peaksDR_x, peaksDR_y, namesDR, plot_region=plot_region)
                    self.append_to_peak_list(qxy, qzRR, (h, k, l), peaksRR_x, peaksRR_y, namesRR, plot_region=plot_region)
                        

        if blanked_figure:
            label_color='1.0'
        else:
            label_color='0.0'
                            
        if True:
            # qzDD: Scattering from direct (refracted) beam (Born approximation)
            # Green
            plt.scatter( peaksDD_x, peaksDD_y, s=80, facecolor='none', edgecolor=(0,1,0), linewidth=1.5 )
            if label_peaks:
                for x, y, s in zip( peaksDD_x, peaksDD_y, namesDD ):
                    plt.text( x, y, s, size=12, color=label_color, horizontalalignment='left', verticalalignment='bottom' )
            
        if True:
            # qzRD: Scattering from substrate-reflected beam
            # Yellow
            plt.scatter( peaksRD_x, peaksRD_y, s=80, facecolor='none', edgecolor=(1,1,0), linewidth=1.5 )
            if label_peaks:
                for x, y, s in zip( peaksRD_x, peaksRD_y, namesRD ):
                    plt.text( x, y, s, size=12, color=label_color, horizontalalignment='left', verticalalignment='bottom' )
            
        if False:
            # qzDR: Scattering from direct (refracted) beam reflects from substrate
            # Orange
            # Peaks appear at same location as qzRD (reflection symmetry)
            plt.scatter( peaksDR_x, peaksDR_y, s=80, facecolor='none', edgecolor=(1,0.5,0), linewidth=1.5 )
            if label_peaks:
                for x, y, s in zip( peaksDR_x, peaksDR_y, namesDR ):
                    plt.text( x, y, s, size=12, color=label_color, horizontalalignment='left', verticalalignment='center' )
            
        if False:
            # qzRR: Scattering from substrate-reflected beam reflects from substrate 
            # Red
            # Peaks appear at same location as qzDD (reflection symmetry)
            plt.scatter( peaksRR_x, peaksRR_y, s=80, facecolor='none', edgecolor=(1,0,0), linewidth=1.5 )
            if label_peaks:
                for x, y, s in zip( peaksRR_x, peaksRR_y, namesRR ):
                    plt.text( x, y, s, size=12, color=label_color, horizontalalignment='left', verticalalignment='center' )
        
        
        
                
                
        if True:
            # Show various guides in the figure
            
            k_xray =e.get_k() 
            
            ambient_n = np.real( Material_ambient.get_xray_n(energy=e.get_beam_energy()) )
            film_n = np.real( Material_film.get_xray_n(energy=e.get_beam_energy()) )
            substrate_n = np.real( Material_substrate.get_xray_n(energy=e.get_beam_energy()) )
            
            film_crit_rad = np.radians( Material_film.get_xray_critical_angle(energy=e.get_beam_energy()) )

            alpha_incident_rad = np.radians(e.get_theta_incident())
            if cos(alpha_incident_rad)*ambient_n/film_n > 1.0:
                alpha_incident_effective_rad = 0.0
            else:
                alpha_incident_effective_rad = np.arccos( cos(alpha_incident_rad)*ambient_n/film_n ) # Snell's law (cosine form)

            
            # Direct beam
            qz = 0
            plt.scatter( [0], [qz], s=120, facecolor='none', edgecolor='0.75', linewidth=1.5 )
            plt.axhline( qz, color='0.75' )
            
            # Refracted beam
            if cos(alpha_incident_effective_rad)*film_n/substrate_n > 1.0:
                alpha_exit_direct_rad = 0.0
            else:
                alpha_exit_direct_rad = np.arccos( cos(alpha_incident_effective_rad)*film_n/substrate_n )
            qz = 2*k_xray*sin( (alpha_incident_rad-alpha_exit_direct_rad)/2.0)
            plt.scatter( [0], [qz], s=120, facecolor='none', edgecolor='r', linewidth=1.5 )
            plt.axhline( qz, color='r' )
            
            # Horizon
            qz = 2*k_xray*sin(alpha_incident_rad/2.0)
            l = plt.axhline( qz, color='0.5' )
            l.set_dashes( [10,3] )
            
            # Yoneda
            yoneda_rad = alpha_incident_rad + film_crit_rad
            qz = 2*k_xray*sin(yoneda_rad/2.0)
            plt.axhline( qz, color='orange', linewidth=3.0 )
            
            # Specular beam
            qz = 2*k_xray*sin( (2.0*alpha_incident_rad)/2.0)
            plt.scatter( [0], [qz], s=120, facecolor='none', edgecolor='r', linewidth=1.5 )
            plt.axhline( qz, color='r' )
                
            # Show central meridian of Ewald sphere
            qxys, qzs = ewaldsphere.central_meridian_arc()
            plt.plot( qxys, qzs, '-', color='0.5', linewidth=0.5 )
            plt.plot( -1*qxys, qzs, '-', color='0.5', linewidth=0.5 )

                
                
                
        # Axis scaling
        xi, xf, yi, yf = ax.axis()
        if plot_region[0] != None: xi = plot_region[0]
        if plot_region[1] != None: xf = plot_region[1]
        if plot_region[2] != None: yi = plot_region[2]
        if plot_region[3] != None: yf = plot_region[3]
        if plot_region[0]==None and plot_region[1]==None and plot_region[2]==None and plot_region[3]==None:
            xf = max( xi, xf, yi, yf )
            yf = xf
            xi = -xf
            yi = -yf
            
            
        
        
            
        ax.axis( [xi, xf, yi, yf] )
        
        if blanked_figure:
            plt.xticks( [] )
            plt.yticks( [] )
        else:
            plt.xlabel( r'$q_{xy} \, (\mathrm{\AA^{-1}})$', size=30 )
            plt.ylabel( r'$q_{z} \, (\mathrm{\AA^{-1}})$', size=30  )
            
        
        plt.savefig( filename, transparent=blanked_figure )        
        plt.close()
        
          
          
          
    def plot_ewald_inplane_powder(self, ewaldsphere, filename='output.png', plot_region=[None, None, None, None], plot_buffers=[0.16, 0.035, 0.16, 0.03], label_peaks=False, blanked_figure=False, peaks_present=None, max_hkl=10, thresh=0.001, dpi=100):
        

        # Plot styling
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.labelsize'] = 30
        plt.rcParams['xtick.labelsize'] = 'xx-large'
        plt.rcParams['ytick.labelsize'] = 'xx-large'

        #plt.rcParams['axes.labelsize'] = 35
        #plt.rcParams['xtick.labelsize'] = 28
        #plt.rcParams['ytick.labelsize'] = 28


        fig = plt.figure(figsize=(10,10))
        #fig.subplots_adjust(left=0.17, bottom=0.15, right=0.97, top=0.94, wspace=0.2, hspace=0.2)
        #ax = plt.subplot(111)
        left_buf, right_buf, bottom_buf, top_buf = plot_buffers
        fig_width = 1.0-right_buf-left_buf
        fig_height = 1.0-top_buf-bottom_buf
        ax = fig.add_axes( [left_buf, bottom_buf, fig_width, fig_height], aspect='equal' )

        

        peaks_x = []
        peaks_y = []
        names = []
        
        q_rings = []
        
        for h, k, l, qhkl, qhkl_vector in self.iterate_over_hkl(max_hkl=max_hkl):
            qx, qy, qz = qhkl_vector
            qxy = np.sqrt(qx**2 + qy**2)
            
            # Make sure peaks appears on correct side of image
            if qx<0:
                qxy *= -1

                
            #if (plot_region[0]!=None and qxy>=plot_region[0]) and (plot_region[1]!=None and qxy<=plot_region[1]) and (plot_region[2]!=None and qz>=plot_region[2]) and (plot_region[3]!=None and qz<=plot_region[3]):
            if True:
                #d = ewaldsphere.distance_to_ewald_surface(qx, qy, qz)
                #if d<thresh:
                if True:
                    
                    #if [abs(h), abs(k), abs(l)] == [1,2,0]:
                        #print(h,k,l, qxy, qz)
                    
                    if peaks_present==None or [abs(h), abs(k), abs(l)] in peaks_present:
                        
                        if label_peaks:
                            if True:
                                s = r'$'
                                if h<0:
                                    s += '\overline{'+str(abs(h))+'}'
                                else:
                                    s += str(h)
                                if k<0:
                                    s += '\overline{'+str(abs(k))+'}'
                                else:
                                    s += str(k)
                                if l<0:
                                    s += '\overline{'+str(abs(l))+'}'
                                else:
                                    s += str(l)
                                s += r'$'
                            else:
                                s = '%d%d%d' % (abs(h), abs(k), abs(l))
                                
                            # Check if this peak already exists
                            imatch = -1
                            for i in range(len(peaks_x)):
                                if abs(qxy-peaks_x[i])<thresh and abs(qz-peaks_y[i])<thresh:
                                    #print qxy, peaks_x[i], qz, peaks_y[i]
                                    imatch = i

                            #if [abs(h), abs(k), abs(l)] == [1,2,0]:
                                #print(h,k,l, qxy, qz, imatch)
                                #print(peaks_x[imatch], peaks_y[imatch], names[imatch])

                                                                
                            if imatch==-1:
                                # New peak
                                peaks_x.append( +abs(qxy) )
                                peaks_y.append( qz )
                                names.append( s )
                                
                                peaks_x.append( -abs(qxy) )
                                peaks_y.append( qz )
                                names.append( s )
                                
                                
                                # Put a q-ring for each peak
                                xs = []
                                ys = []
                                for rot_angle in np.linspace(0, 2*np.pi, num=200):
                                    qx_rot = qx*np.cos(rot_angle) + qz*np.sin(rot_angle)
                                    qy_rot = qy
                                    qz_rot = -qx*np.sin(rot_angle) + qz*np.cos(rot_angle)
                                    qxy_rot = np.sqrt(np.square(qx_rot)+np.square(qy_rot))
                                    if qx_rot<0:
                                        qxy_rot *= -1
                                    
                                    xs.append( qxy_rot )
                                    ys.append( qz_rot )
                                q_rings.append( [xs, ys] )
                                
                                
                            else:
                                # Existing peak
                                #names.append( '' )
                                pass
                            
                            
                            
                        else:
                            peaks_x.append( qxy )
                            peaks_y.append( qz )
                            names.append( '' )
                            
                                
                            

        #for ring in q_rings:
            #x, y = ring
            #plt.scatter(x, y, s=4, facecolor='g', edgecolor='none', alpha=0.3)
                
        
        plt.scatter( peaks_x, peaks_y, s=80, facecolor='none', edgecolor=(0,1,0), linewidth=1.5 )
        if label_peaks:
            for x, y, s in zip( peaks_x, peaks_y, names ):
                if blanked_figure:
                    plt.text( x, y, s, size=12, color='1.0', horizontalalignment='left', verticalalignment='bottom' )
                else:
                    plt.text( x, y, s, size=12, color='0.0', horizontalalignment='left', verticalalignment='bottom' )
                
                
                
        # Axis scaling
        xi, xf, yi, yf = ax.axis()
        if plot_region[0] != None: xi = plot_region[0]
        if plot_region[1] != None: xf = plot_region[1]
        if plot_region[2] != None: yi = plot_region[2]
        if plot_region[3] != None: yf = plot_region[3]
        if plot_region[0]==None and plot_region[1]==None and plot_region[2]==None and plot_region[3]==None:
            xf = max( xi, xf, yi, yf )
            yf = xf
            xi = -xf
            yi = -yf
            
            
        # Show central meridian of Ewald sphere
        qxys, qzs = ewaldsphere.central_meridian_arc()
        plt.plot( qxys, qzs, '-', color='0.5', linewidth=0.5 )
        plt.plot( -1*qxys, qzs, '-', color='0.5', linewidth=0.5 )
        
        
            
        ax.axis( [xi, xf, yi, yf] )
        
        if blanked_figure:
            plt.xticks( [] )
            plt.yticks( [] )
        else:
            plt.xlabel( r'$q_{xy} \, (\mathrm{\AA^{-1}})$', size=30 )
            plt.ylabel( r'$q_{z} \, (\mathrm{\AA^{-1}})$', size=30  )
            
        
        plt.savefig( filename, transparent=blanked_figure, dpi=dpi )   
        #plt.show()
        plt.close()
        
        
        
    def list_powder_peaks(self, filename='peaks.dat', max_hkl=9, thresh=0.001):
        
        peaks_q = []
        names = []
        for h, k, l, qhkl, qhkl_vector in self.iterate_over_hkl(max_hkl=max_hkl):
            
            qx, qy, qz = qhkl_vector
            qxy = np.sqrt(qx**2 + qy**2)
            
            #s = '%d%d%d' % (abs(h), abs(k), abs(l))
            s = '%d%d%d' % (h, k, l)
                
            peaks_q.append( qhkl )
            names.append( s )

            
        # Sort
        peaks_q = np.asarray(peaks_q)
        names = np.asarray(names)
        indices = np.argsort( peaks_q )
        q_sorted = peaks_q[indices]
        names_sorted = names[indices]
        
        fout = open(filename, 'w')
        for q, n in zip(q_sorted, names_sorted):
            #print( '%02.4f\t%s' % (q,n) )
            fout.write( '%02.4f\t%s\n' % (q,n) )
        fout.close()

 
    def plot_ewald_two_beam_qxqz(self, ewaldsphere, Material_ambient, Material_film, Material_substrate, filename='output.png', plot_region=[None, None, None, None], plot_buffers=[0.16, 0.035, 0.16, 0.03], label_peaks=False, blanked_figure=False, peaks_present=None, max_hkl=10, thresh=0.01, dpi=100):
        
        
        # Prepare for refraction correction computations
        k_xray =e.get_k() 
        
        ambient_n = np.real( Material_ambient.get_xray_n(energy=e.get_beam_energy()) )
        film_n = np.real( Material_film.get_xray_n(energy=e.get_beam_energy()) )
        substrate_n = np.real( Material_substrate.get_xray_n(energy=e.get_beam_energy()) )
        
        film_crit_rad = np.radians( Material_film.get_xray_critical_angle(energy=e.get_beam_energy()) )
        substrate_crit_rad = np.radians( Material_substrate.get_xray_critical_angle(energy=e.get_beam_energy()) )

        alpha_incident_rad = np.radians(e.get_theta_incident())
        alpha_incident_effective_rad = np.arccos( cos(alpha_incident_rad)*ambient_n/film_n ) # Snell's law (cosine form)
            

        # Plot styling
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.labelsize'] = 30
        plt.rcParams['xtick.labelsize'] = 'xx-large'
        plt.rcParams['ytick.labelsize'] = 'xx-large'

        #plt.rcParams['axes.labelsize'] = 35
        #plt.rcParams['xtick.labelsize'] = 28
        #plt.rcParams['ytick.labelsize'] = 28


        fig = plt.figure(figsize=(10,10))
        #fig.subplots_adjust(left=0.17, bottom=0.15, right=0.97, top=0.94, wspace=0.2, hspace=0.2)
        #ax = plt.subplot(111)
        left_buf, right_buf, bottom_buf, top_buf = plot_buffers
        fig_width = 1.0-right_buf-left_buf
        fig_height = 1.0-top_buf-bottom_buf
        ax = fig.add_axes( [left_buf, bottom_buf, fig_width, fig_height], aspect='equal' )

        

        # BA term
        peaks_x = []
        peaks_y = []
        names = []
        
        # Reflected-beam scattering
        peaksTR_x = []
        peaksTR_y = []
        namesTR = []

        # Scattering is reflected
        #peaksRT_x = []
        #peaksRT_y = []
        #namesRT = []

        # Beam is reflected, scattering is reflected
        #peaksRR_x = []
        #peaksRR_y = []
        #namesRR = []
        
        for h, k, l, qhkl, qhkl_vector in self.iterate_over_hkl(max_hkl=max_hkl):
            qx, qy, qz = qhkl_vector
            qxy = np.sqrt(qx**2 + qy**2)
            qtotal = np.sqrt(qx**2 + qy**2 + qz**2)
            
            # Make sure peaks appears on correct side of image
            if qx<0:
                qxy *= -1
                
            # Treat as inplane powder, showing up on left-side (-qx)
            qxy = -abs(qxy)
            qx = -abs(qx)
                
            #if (plot_region[0]!=None and qxy>=plot_region[0]) and (plot_region[1]!=None and qxy<=plot_region[1]) and (plot_region[2]!=None and qz>=plot_region[2]) and (plot_region[3]!=None and qz<=plot_region[3]):
            #if (plot_region[0]!=None and qxy>=plot_region[0]) and (plot_region[1]!=None and qxy<=plot_region[1]) and (plot_region[3]!=None and qz<=plot_region[3]):
            if True:
                #d = ewaldsphere.distance_to_ewald_surface(qx, qy, qz)
                #if d<thresh:
                if True: # Inplane powder (plot all peaks)
                    if peaks_present==None or [abs(h), abs(k), abs(l)] in peaks_present:
                        # This peak will be plotted; now compute the refraction correction for it
                        qzorig = qz
                        

                        # Shift of direct beam (due to film-air interface)
                        direct_beam_angle_shift_rad = (alpha_incident_rad-alpha_incident_effective_rad)

                        # Scattering event occurs inside film, with angle 2theta_B
                        two_theta_B_rad = 2.0*np.arcsin(qz/(2.0*k_xray))
                        
                        # Scattered ray refracts as it exits
                        scattered_incident_angle_rad = two_theta_B_rad - alpha_incident_effective_rad
                        scattered_exit_angle_rad = np.arccos( cos(scattered_incident_angle_rad)*film_n/ambient_n ) # Snell's law (cosine form)
                        
                        if (scattered_incident_angle_rad) > 0:
                            # GISAXS (scattering above horizon)
                            
                            if True:
                                # Direct computation of refraction-induced angle shifts
                                # Inspired by Figure 3 in:
                                # Breiby et al.  J. Appl. Cryst. (2008). 41, 262–271 doi: 10.1107/S0021889808001064
                                scatter_beam_angle_shift_rad = (scattered_exit_angle_rad-scattered_incident_angle_rad)
                                #two_theta_B_final_rad = two_theta_B_rad + direct_beam_angle_shift_rad + scatter_beam_angle_shift_rad
                                two_theta_B_final_rad = alpha_incident_rad + scattered_exit_angle_rad
                                qz = 2*k_xray*sin(two_theta_B_final_rad/2.0)
                            
                            if False:
                                # Equation 2 in:
                                # Xinhui Lu et al. 
                                qz_in_film = k_xray*( 
                                                np.sqrt( (sin(alpha_incident_rad))**2 - (sin(film_crit_rad))**2 ) 
                                                + np.sqrt( ( (qz/k_xray)-sin(alpha_incident_rad) )**2 - (sin(film_crit_rad))**2 )
                                                )
                                qz_shift = qz-qz_in_film
                                qz += qz_shift

                            if False:
                                # Equation 22 in:
                                # Byeongdu Lee Macromolecules 2005, 38, 4311 doi: 10.1021/ma047562d 
                                scattered_exit_angle_rad = np.arccos( np.sqrt( film_n**2 -((qz/k_xray) + np.sqrt(film_n**2 - (cos(alpha_incident_rad))**2 ) )**2 ) )
                                two_theta_B_final_rad = alpha_incident_rad + scattered_exit_angle_rad
                                qz = 2*k_xray*sin(two_theta_B_final_rad/2.0)

                            if False:
                                # Equation 26 in:
                                # Busch et al. J. Appl. Cryst. J. Appl. Cryst. (2006). 39, 433–442 doi:10.1107/S0021889806012337
                                qz = k_xray*( 
                                        - sin(alpha_incident_rad) 
                                        - np.sqrt( 
                                                (sin(film_crit_rad)**2) 
                                                + ( (qz/k_xray) - ( sin(alpha_incident_rad)**2 - sin(film_crit_rad)**2 )**2 )
                                                )
                                        )
 
                            
                        else:
                            # GTSAXS (scattering below horizon)
                            
                            if True:
                                # Direct computation of refraction-induced angle shifts
                                substrate_incident_effective_rad = alpha_incident_effective_rad - two_theta_B_rad
                                if cos(substrate_incident_effective_rad)*film_n/substrate_n > 1.0:
                                    # Scattering ray intersects substrate below its critical angle
                                    #substrate_exit_angle_rad = 0.0 # peak appears along horizon
                                    substrate_exit_angle_rad = np.nan # peak is not seen (total reflection)
                                else:
                                    substrate_exit_angle_rad = np.arccos( cos(substrate_incident_effective_rad)*film_n/substrate_n ) # Snell's law (cosine form)
                                
                                scatter_beam_angle_shift_rad = substrate_incident_effective_rad - substrate_exit_angle_rad
                                two_theta_B_final_rad = two_theta_B_rad + direct_beam_angle_shift_rad + scatter_beam_angle_shift_rad
                                qz = 2*k_xray*sin(two_theta_B_final_rad/2.0)

                            if False:
                                # Xinhui Lu et al. 
                                qz_in_film = k_xray*( 
                                                np.sqrt( (sin(alpha_incident_rad))**2 - (sin(film_crit_rad))**2 ) 
                                                - np.sqrt( 
                                                    ( (qz/k_xray)-sin(alpha_incident_rad) )**2 
                                                    + (sin(substrate_crit_rad))**2
                                                    - (sin(film_crit_rad))**2 
                                                    )
                                                )
                                qz_shift = qz-qz_in_film
                                qz += qz_shift
                                

                            
                        
                        if np.isnan(qz):
                            pass
                        elif (plot_region[0]!=None and qxy>=plot_region[0]) and (plot_region[1]!=None and qxy<=plot_region[1]) and (plot_region[2]!=None and qz>=plot_region[2]) and (plot_region[3]!=None and qz<=plot_region[3]):
                            peaks_x.append( qxy )
                            peaks_y.append( qz )
                            if label_peaks:
                                if True:
                                    s = r'$'
                                    if h<0:
                                        s += '\overline{'+str(abs(h))+'}'
                                    else:
                                        s += str(h)
                                    if k<0:
                                        s += '\overline{'+str(abs(k))+'}'
                                    else:
                                        s += str(k)
                                    if l<0:
                                        s += '\overline{'+str(abs(l))+'}'
                                    else:
                                        s += str(l)
                                    s += r'$'
                                else:
                                    s = '%d%d%d' % (abs(h), abs(k), abs(l))
                                names.append( s )

                                
                                
                                
                        # Plot peaks from reflected beam
                        qz = qzorig
                        
                        # Specular reflection inside the film (reflection from film-substrate interface)
                        specular_beam_in_film_rad = alpha_incident_rad + alpha_incident_effective_rad

                        # Scattering event occurs inside film, with angle 2theta_B
                        two_theta_B_rad = 2.0*np.arcsin(qz/(2.0*k_xray))
                        
                        
                        # Scattered ray refracts as it exits
                        #scattered_incident_angle_rad = two_theta_B_rad + specular_beam_in_film_rad # WRONG
                        scattered_incident_angle_rad = alpha_incident_effective_rad + two_theta_B_rad
                        scattered_exit_angle_rad = np.arccos( cos(scattered_incident_angle_rad)*film_n/ambient_n ) # Snell's law (cosine form)
                        
                        if (scattered_incident_angle_rad) > 0:
                            # GISAXS (scattering above horizon)
                            
                            # Direct computation of refraction-induced angle shifts
                            # Inspired by Figure 3 in:
                            # Breiby et al.  J. Appl. Cryst. (2008). 41, 262–271 doi: 10.1107/S0021889808001064
                            scatter_beam_angle_shift_rad = (scattered_exit_angle_rad-scattered_incident_angle_rad)
                            #two_theta_B_final_rad = two_theta_B_rad + direct_beam_angle_shift_rad + scatter_beam_angle_shift_rad
                            two_theta_B_final_rad = alpha_incident_rad + scattered_exit_angle_rad
                            qz = 2*k_xray*sin(two_theta_B_final_rad/2.0)
                            
                            
                        else:
                            # GTSAXS (scattering below horizon)
                            
                            scattered_incident_angle_rad = -alpha_incident_effective_rad + abs(two_theta_B_rad)
                            if cos(scattered_incident_angle_rad)*film_n/substrate_n > 1.0:
                                # Scattering ray intersects substrate below its critical angle
                                #substrate_exit_angle_rad = 0.0 # peak appears along horizon
                                substrate_exit_angle_rad = np.nan # peak is not seen (total reflection)
                            else:
                                scattered_exit_angle_rad = np.arccos( cos(scattered_incident_angle_rad)*film_n/substrate_n ) # Snell's law (cosine form)
                            
                            two_theta_B_final_rad = alpha_incident_rad - scattered_exit_angle_rad
                            qz = 2*k_xray*sin(two_theta_B_final_rad/2.0)
                            



                        #print( '[{},{},{}] q = {:.2f} = ({:.2f}, {:.2f}, {:.2f}) qxy = {:.2f} '.format(h,k,l, qtotal, qx, qy, qz, qxy) )
                        

                        if np.isnan(qz):
                            pass
                        elif (plot_region[0]!=None and qxy>=plot_region[0]) and (plot_region[1]!=None and qxy<=plot_region[1]) and (plot_region[2]!=None and qz>=plot_region[2]) and (plot_region[3]!=None and qz<=plot_region[3]):
                            peaksTR_x.append( qxy )
                            peaksTR_y.append( qz )
                            if label_peaks:
                                if True:
                                    s = r'$'
                                    if h<0:
                                        s += '\overline{'+str(abs(h))+'}'
                                    else:
                                        s += str(h)
                                    if k<0:
                                        s += '\overline{'+str(abs(k))+'}'
                                    else:
                                        s += str(k)
                                    if l<0:
                                        s += '\overline{'+str(abs(l))+'}'
                                    else:
                                        s += str(l)
                                    s += r'$'
                                else:
                                    s = '%d%d%d' % (abs(h), abs(k), abs(l))
                                namesTR.append( s )
                        

        #print(peaks_x, peaks_y)
        plt.scatter( peaks_x, peaks_y, s=100, marker='x', facecolor=(0,1,0), linewidth=2 )
        plt.scatter( peaksTR_x, peaksTR_y, s=100, facecolor='none', edgecolor=(1,1,0), linewidth=2 )
        if label_peaks:
            for x, y, s in zip( peaks_x, peaks_y, names ):
                if blanked_figure:
                    plt.text( x, y, s, size=12, color='1.0', horizontalalignment='left', verticalalignment='bottom' )
                else:
                    plt.text( x, y, s, size=12, color='0.0', horizontalalignment='left', verticalalignment='bottom' )
            for x, y, s in zip( peaksTR_x, peaksTR_y, namesTR ):
                if blanked_figure:
                    plt.text( x, y, s, size=12, color='1.0', horizontalalignment='left', verticalalignment='top' )
                else:
                    plt.text( x, y, s, size=12, color='0.0', horizontalalignment='left', verticalalignment='top' )
                
                
        if 1:
            # Show various guides in the figure
            
            # Direct beam
            qz = 0
            plt.scatter( [0], [qz], s=120, facecolor='none', edgecolor='0.75', linewidth=1.5 )
            plt.axhline( qz, color='0.75' )
            
            # Refracted beam
            alpha_exit_direct_rad = np.arccos( cos(alpha_incident_effective_rad)*film_n/substrate_n )
            qz = 2*k_xray*sin( (alpha_incident_rad-alpha_exit_direct_rad)/2.0)
            plt.scatter( [0], [qz], s=120, facecolor='none', edgecolor='m', linewidth=1.5 )
            plt.axhline( qz, color='m' )
            
            # Horizon
            qz = 2*k_xray*sin(alpha_incident_rad/2.0)
            l = plt.axhline( qz, color='0.5' )
            l.set_dashes( [10,3] )
            
            # Yoneda
            yoneda_rad = alpha_incident_rad + film_crit_rad
            qz = 2*k_xray*sin(yoneda_rad/2.0)
            plt.axhline( qz, color='orange', linewidth=3.0 )
            
            # Specular beam
            qz = 2*k_xray*sin( (2.0*alpha_incident_rad)/2.0)
            plt.scatter( [0], [qz], s=120, facecolor='none', edgecolor='r', linewidth=1.5 )
            plt.axhline( qz, color='r' )
                
            # Show central meridian of Ewald sphere
            #qxys, qzs = ewaldsphere.central_meridian_arc()
            #plt.plot( qxys, qzs, '-', color='0.5', linewidth=0.5 )
            #plt.plot( -1*qxys, qzs, '-', color='0.5', linewidth=0.5 )
            plt.plot( [0,0], [plot_region[2], plot_region[3]], '-', color='0.5', linewidth=0.5 )

                
        # Axis scaling
        xi, xf, yi, yf = ax.axis()
        #print('axis {}'.format(ax.axis()))
        if plot_region[0] != None: xi = plot_region[0]
        if plot_region[1] != None: xf = plot_region[1]
        if plot_region[2] != None: yi = plot_region[2]
        if plot_region[3] != None: yf = plot_region[3]
        if plot_region[0]==None and plot_region[1]==None and plot_region[2]==None and plot_region[3]==None:
            xf = max( xi, xf, yi, yf )
            yf = xf
            xi = -xf
            yi = -yf         
       
        ax.axis( [xi, xf, yi, yf] )
        #print('axis {}'.format(ax.axis()))  
        
        if blanked_figure:
            plt.xticks( [] )
            plt.yticks( [] )
        else:
            plt.xlabel( r'$q_{xy} \, (\mathrm{\AA^{-1}})$', size=30 )
            plt.ylabel( r'$q_{z} \, (\mathrm{\AA^{-1}})$', size=30  )
            
        
        plt.savefig( filename, transparent=blanked_figure, dpi=dpi )        
        plt.close()
        
        
        
    # END OF: class UnitCell(object)
    ############################################

        
        
# EwaldSphere
###################################################################    
class EwaldSphere(object):
    """Describes the geometry (size/curvature) of the Ewald sphere
    in reciprocal space."""
    
    
    def __init__(self, wavelength=None, energy=None, theta_incident=None):
        
        if wavelength==None and energy==None:
            # Assume Cu K-alpha
            self.set_wavelength(1.5418)

            #self.set_wavelength(0.7107) # Molybdenum    0.71 A
            #self.set_wavelength(1.5418) # Copper        1.54 A
            #self.set_wavelength(1.7902) # Cobalt        1.79 A
            #self.set_wavelength(1.9373) # Iron          1.94 A
    
        elif wavelength==None:
            self.set_beam_energy( energy )
        else:
            self.set_wavelength( wavelength )

        self.theta_incident = theta_incident

        
    def set_theta_incident(self, theta_incident):
        self.theta_incident = theta_incident
        
            
    def get_theta_incident(self):
        return self.theta_incident
            
    def get_wavelength(self):
        """Units: Angstroms"""
        return self.wavelength


    def get_k(self):
        """Units: 1/Angstroms"""
        k = 2.0*np.pi/self.wavelength
        return k
        
        
    def get_beam_energy(self):
        h = 6.626068e-34 # m^2 kg / s
        c = 299792458 # m/s
        
        lam = self.get_wavelength()*1e-10 # m
        E = h*c/lam # Joules
        
        E *= 6.24150974e18 # electron volts
        
        E /= 1000.0 # keV
        
        return E


    def set_wavelength(self, wavelength):
        self.wavelength = wavelength
        
    
    def set_beam_energy(self, E):
        h = 6.626068e-34 # m^2 kg / s
        c = 299792458 # m/s
        
        E *= 1000.0 # eV
        E /= 6.24150974e18 # Joules
        
        lam = h*c/E # meters
        lam *= 1e10 # Angstroms
        
        self.set_wavelength(lam)
        
        
    def determine_in_plane_angle(self, qxy, qz=0.0, theta_incident=0.0):
        """Returns an angle (omega) that describes the in-plane angle between the
        pure-qx axis and the specified position (qxy,qz) in reciprocal space,
        under the assumption that this point lies on the surface of the Ewald sphere.
        
        Essentially, this is answering the question: given that I see a peak on my
        detector, corresponding to (qxy,qz), what is the in-plane angle (in the qx-qy
        plane) between qx and the qxy vector?"""
        
        k = self.get_k()
        if theta_incident==None:
            # Use internal value
            theta_incident = self.theta_incident
        theta_incident_rad = np.radians(theta_incident)
        
        from scipy.optimize import fsolve
        
        def equations(p, qxy=qxy, qz=qz, theta_incident=theta_incident, k=k):
            
            # The variable we are fitting for
            omega_rad, = p
            
            # Non-fit values: qxy, qz, k, theta_incident, k
            
            return ( (qxy*cos(omega_rad))**2 + (qxy*sin(omega_rad)+k*cos(theta_incident_rad))**2 + (qz-k*sin(theta_incident_rad))**2 - k**2 )

            
        omega_rad, =  fsolve(equations, ( np.radians(5.0) ) )
        #print( 'omega_rad = %.2f (err = %.4f)' % ( omega_rad, equations((omega_rad, )) ) )
        
        omega = abs( np.degrees(omega_rad) )
        #print( 'omega = %.2f (err = %.4f)' % ( omega, equations((omega_rad, )) ) )
        
        
        return omega

        
    def distance_to_ewald_surface(self, qx, qy, qz, theta_incident=None):
        """Returns the (shortest) distance between the given 
        point (qx,qy,qz) and the surface of the Ewald sphere."""
        
        # The Ewald sphere is centered at ( +0, -k*cos(theta_incident), +k*sin(theta_incident) )
        k = self.get_k()
        if theta_incident==None:
            theta_incident = self.get_theta_incident()
        theta_incident_rad = np.radians(theta_incident)
        
        # Move Ewald sphere to origin
        qy += +k*cos(theta_incident_rad)
        qz += -k*sin(theta_incident_rad)
        
        q = np.sqrt( qx**2 + qy**2 + qz**2 )
        
        distance = abs(k-q)
        
        return distance
                
                
    def central_meridian_arc(self):
        
        k = self.get_k()
        theta_incident_rad = np.radians(self.get_theta_incident())
        cos_theta = cos(theta_incident_rad)
        sin_theta = sin(theta_incident_rad)
        
        qzs = np.linspace( 0, k, 50 )
        qxys = -k*cos_theta + np.sqrt( k**2 - np.square(qzs-k*sin_theta) )
        
        return qxys, qzs
    
    
    def intersection_ring(self, q_total):
        """Returns points along the ring where the given reciprocal-space shell 
        (of size q_total) intersects the Ewald sphere."""
        
        # WARNING: This ignores the effect of the incident angle
        
        

        # This is a point that intersects the Ewald sphere
        # (if incident_angle = 0)
        theta = np.arcsin(q_total/(2*self.get_k()))
        qx, qy, qz = 0, -q_total*np.sin(theta), q_total*np.cos(theta)
        
        #qx, qy, qz = 0, 0, q_total
        
        qxs = []
        qys = []
        qzs = []
        
        for rot_angle in np.linspace(0, 2*np.pi, num=200):
            qx_rot = qx*np.cos(rot_angle) + qz*np.sin(rot_angle)
            qy_rot = qy
            qz_rot = -qx*np.sin(rot_angle) + qz*np.cos(rot_angle)
            qxy_rot = np.sqrt(np.square(qx_rot)+np.square(qy_rot))
            if qx_rot<0:
                qxy_rot *= -1
            
            qxs.append( qx_rot )
            qys.append( qy_rot )
            qzs.append( qz_rot )
        
        return qxs, qys, qzs
    
    
    def intersection_q_shell(self, q_total, target, num=50):
        """Returns points along the ring where the given reciprocal-space shell 
        (of size q_total) intersects the Ewald sphere.
        
        Also returns the 'shell distance' (distance along the spherical surface of
        size q_total) to a target point
        """
        
        # WARNING: This ignores the effect of the incident angle
        
        

        # This is a point that intersects the Ewald sphere
        # (if incident_angle = 0)
        theta = np.arcsin(q_total/(2*self.get_k()))
        qx, qy, qz = 0, -q_total*np.sin(theta), q_total*np.cos(theta)
        
        #qx, qy, qz = 0, 0, q_total
        
        qxs = []
        qys = []
        qzs = []
        ds = []
        
        for rot_angle in np.linspace(0, 2*np.pi, num=num):
            qx_rot = qx*np.cos(rot_angle) + qz*np.sin(rot_angle)
            qy_rot = qy
            qz_rot = -qx*np.sin(rot_angle) + qz*np.cos(rot_angle)
            qxy_rot = np.sqrt(np.square(qx_rot)+np.square(qy_rot))
            if qx_rot<0:
                qxy_rot *= -1
            
            qxs.append( qx_rot )
            qys.append( qy_rot )
            qzs.append( qz_rot )
            
            
            # Compute distance along surface of q_total shell
            # This is the 'great circle' distance
            
            qxt, qyt, qzt = target
            delta_sigma = np.arccos( (qx_rot*qxt + qy_rot*qyt + qz_rot*qzt)/(q_total**2) ) # Divide by q^2 to make them unit vectors
            d = q_total*delta_sigma
            
            if np.isnan(d):
                print('ERROR d is nan')
            
            ds.append( d )
        
        
        return qxs, qys, qzs, ds
    
    
    # END OF: class EwaldSphere(object)
    ############################################

        

        

# Testing
###################################################################    
#TestLattice = UnitCell( 1.0, 10.0, 1.0, 90.0, 90.0, 90.0 )
#h, k, l = 1, 0, 0
#h, k, l = 0, 1, 0
#h, k, l = 0, 0, 1
#h, k, l = 1, 0, 1

#qhkl, (qx, qy, qz), qxy, angle_wrt_x, angle_wrt_z = TestLattice.print_q_hkl_exp(h, k, l)
#Rubrene.apply_rotation_y(-90.0)
#qhkl, (qx, qy, qz), qxy, angle_wrt_x, angle_wrt_z = TestLattice.print_q_hkl_exp(h, k, l)


class Element(object):
    
    def __init__(self, symbol, atomic_number, atomic_weight, density=1.0 ):
        self.symbol = symbol
        self.atomic_number = atomic_number
        self.atomic_weight = atomic_weight
        self.density = density # g/cm^3
    
    def set_density(self, density=1.0):
        self.density = density # g/cm^3
    
    def get_density(self):
        return self.density
    
    def get_atomic_weight(self):
        return self.atomic_weight
        
    def get_molecular_weight(self):
        return self.get_atomic_weight()
    
    def set_xray_properties(self, properties):
        self.xray_properties = properties
        
    def get_xray_delta_beta(self, energy=13.0):
        # Usage:
        # delta, beta = self.get_xray_delta_beta(energy=13.0)
        
        try:
            delta = self.xray_properties[energy][2]
            beta = self.xray_properties[energy][3]
        except KeyError:
            # Interpolate instead
            #energy_close = min(self.xray_properties.keys(), key=lambda k: abs(k-energy))

            #keys = np.sort(self.xray_properties.keys())
            keys = list(self.xray_properties.keys())
            idx = -1
            for i, key in enumerate(keys):
                if idx==-1 and key>energy:
                    idx = i
                    
            energy_low = keys[idx-1]
            energy_high = keys[idx]
            extent = (energy-energy_low)/(energy_high-energy_low)
            
            delta = self.xray_properties[energy_high][2]*extent + self.xray_properties[energy_low][2]*(1.0-extent)
            beta = self.xray_properties[energy_high][3]*extent + self.xray_properties[energy_low][3]*(1.0-extent)
            
        return delta, beta

        
    def get_xray_delta_beta_intrinsic(self, energy=13.0):
        """The intrinsic delta and beta are those scaled for the density
        of the material. Thus they are a 'per-atom' contribution."""
        
        delta, beta = self.get_xray_delta_beta(energy)
        delta *= self.atomic_weight/self.density
        beta *= self.atomic_weight/self.density
        
        return delta, beta        

        
    def get_f1_f2(self, energy=13.0):
        """Returns this element's f1 and f2 values, for the given energy."""
        
        try:
            f1 = self.xray_properties[energy][0]
            f2 = self.xray_properties[energy][1]
        except KeyError:
            # Interpolate instead
            #energy_close = min(self.xray_properties.keys(), key=lambda k: abs(k-energy))

            keys = np.sort(self.xray_properties.keys())
            idx = -1
            for i, key in enumerate(keys):
                if idx==-1 and key>energy:
                    idx = i
                    
            energy_low = keys[idx-1]
            energy_high = keys[idx]
            extent = (energy-energy_low)/(energy_high-energy_low)
            
            f1 = self.xray_properties[energy_high][0]*extent + self.xray_properties[energy_low][0]*(1.0-extent)
            f2 = self.xray_properties[energy_high][1]*extent + self.xray_properties[energy_low][1]*(1.0-extent)
            
        return f1, f2        
        
        
    def get_xray_n(self, energy=13.0):
        delta = self.xray_properties[energy][2]
        beta = self.xray_properties[energy][3]
        
        n = 1.0 - delta + beta*1j
        
        return n
        
    def get_xray_critical_angle(self, energy=13.0, ambient_n=1.0):
        
        # Method 1: using Snell's law
        material_n = self.get_xray_n(energy)
        #crit_angle_rad = np.pi/2.0 - np.arcsin( material_n/ambient_n )
        crit_angle_rad = np.arccos( material_n/ambient_n )
        
        # Method 2: theta_critical = sqrt(2*delta)
        #delta, beta = self.get_xray_delta_beta(energy=energy)
        #crit_angle_rad = np.sqrt( 2.0*delta )
        
        return np.degrees(np.real(crit_angle_rad))
        
        
    def get_xray_attenuation_length(self, energy=13.0):
        """The depth into the material that x-rays penetrate;
        specifically where intensity falls to 1/e of its initial value
        (in microns)."""
        
        try:
            att_len = self.xray_properties[energy][-1]
        except KeyError:
            # Interpolate instead
            #energy_close = min(self.xray_properties.keys(), key=lambda k: abs(k-energy))

            keys = np.sort(self.xray_properties.keys())
            idx = -1
            for i, key in enumerate(keys):
                if idx==-1 and key>energy:
                    idx = i
                    
            energy_low = keys[idx-1]
            energy_high = keys[idx]
            extent = (energy-energy_low)/(energy_high-energy_low)
            
            att_len = self.xray_properties[energy_high][-1]*extent + self.xray_properties[energy_low][-1]*(1.0-extent)
            
        return att_len

        
    def get_xray_mass_absorption_coefficient(self, energy=13.0):
        
        att_len = self.get_xray_attenuation_length(energy=energy)
        density = self.get_density()
        
        mu = 1.0/( att_len * density )
        
        return mu
      
        
    def to_string(self):
        return self.symbol

        
        
        
# Element properties taken from:
# http://henke.lbl.gov/optical_constants/pert_form.html


Element_H = Element('H', 1, 1.008, density=0.899e-4)
#                              f1     f2         delta       beta        cri-ang    attenuation(um)
xray_properties = {    8.0 : [ 1.000, 0.1127e-5, 5.7846e-10, 6.5200e-16, 1.949e-3,  3.1357e8 ] ,
                      13.0 : [ 1.000, 0.3476e-6, 2.1906e-10, 7.6141e-17, 1.199e-3,  3.0640e8 ] ,
                      13.5 : [ 1.000, 0.3168e-6, 2.0314e-10, 6.4363e-17, 1.155e-3,  3.0603e8 ] ,
                      14.0 : [ 1.000, 0.2898e-6, 1.8888e-10, 5.473e-17, 1.114e-3,  3.0567e-8 ] ,
                      14.5 : [ 1.000, 0.2658e-6, 1.7608e-10, 4.6797e-17, 1.075e-3,  3.0531e-8 ]  }
Element_H.set_xray_properties(xray_properties)

Element_Be = Element('Be', 4, 9.012, density=1.85)
xray_properties = {    8.0 : [ 4.004, 0.1558e-2, 5.3265e-6, 2.0724e-9, 0.187,  5277.0 ] ,
                      13.0 : [ 4.001, 0.5150e-3, 2.0157e-6, 2.5946e-10, 0.115,  16877.0 ] ,
                      13.5 : [ 4.001, 0.4721e-3, 1.8691e-6, 2.2054e-10, 0.111,  18016.0 ] ,
                      14.0 : [ 4.001, 0.4341e-3, 1.7379e-6, 1.8856e-10, 0.107,  19105.0 ] ,
                      14.5 : [ 4.001, 0.4004e-3, 1.6201e-6, 1.6211e-10, 0.103,  20137.0 ]  }
Element_Be.set_xray_properties(xray_properties)

Element_B = Element('B', 5, 10.811, density=2.34)
xray_properties = {    8.0 : [ 5.010, 0.4131e-2, 7.0343e-6, 5.8004e-9, 0.215,  2007.0 ] ,
                      13.0 : [ 5.003, 0.1354e-2, 2.6605e-6, 7.1984e-10, 0.132,  7861.0 ] ,
                      13.5 : [ 5.003, 0.1239e-2, 2.4669e-6, 6.1082e-10, 0.127,  8601.0 ] ,
                      14.0 : [ 5.003, 0.1137e-2, 2.2937e-6, 5.2128e-10, 0.123,  9347.0 ] ,
                      14.5 : [ 5.003, 0.1046e-2, 2.1381e-6, 4.4726e-10, 0.118,  1.0094e4 ]  }
Element_B.set_xray_properties(xray_properties)


Element_C = Element('C', 6, 12.011, density=2.20)    
xray_properties = {   13.0 : [ 6.007, 0.3218e-2, 2.7032e-6, 1.4481e-9, 0.133,  4489 ] ,
                      13.5 : [ 6.007, 0.2946e-2, 2.5064e-6, 1.2294e-9, 0.128,  4988 ] ,
                      14.0 : [ 6.006, 0.2705e-2, 2.3304e-6, 1.0497e-9, 0.124,  5508 ] ,
                      14.5 : [ 6.006, 0.2491e-2, 2.1723e-6, 9.0111e-10, 0.119,  6048 ]  }
Element_C.set_xray_properties(xray_properties)

Element_N = Element('N', 7, 14.007, density=0.125e-2)    
xray_properties = {   13.0 : [ 7.013, 0.6329e-2, 1.5383e-9, 1.3883e-12, 3.178e-3,  4.9812e6 ] ,
                      13.5 : [ 7.012, 0.5809e-2, 1.4262e-9, 1.1816e-12, 3.060e-3,  5.5649e6 ] ,
                      14.0 : [ 7.011, 0.5348e-2, 1.3260e-9, 1.0114e-12, 2.951e-3,  6.1837e6 ] ,
                      14.5 : [ 7.010, 0.4936e-2, 1.2360e-9, 8.7033e-13, 2.849e-3,  6.8363e6 ]  }
Element_N.set_xray_properties(xray_properties)


Element_O = Element('O', 8, 15.999, density=0.143e-02)    
xray_properties = {    8.0 : [ 8.053, 0.3414e-1, 4.6659e-9, 1.9780e-11, 5.535e-3,  6.1705e5 ] ,
                      13.0 : [ 8.022, 0.1183e-1, 1.7601e-9, 2.5967e-12, 3.399e-3,  2.7618e6 ] ,
                      13.5 : [ 8.020, 0.1088e-1, 1.6319e-9, 2.2136e-12, 3.273e-3,  3.0956e6 ] ,
                      14.0 : [ 8.019, 0.1003e-1, 1.5171e-9, 1.8976e-12, 3.156e-3,  3.4527e6 ] ,
                      14.5 : [ 8.017, 0.9272e-2, 1.4140e-9, 1.6353e-12, 3.047e-3,  3.8330e6 ]  }
Element_O.set_xray_properties(xray_properties)

Element_Al = Element('Al', 13, 26.982, density=2.70)
xray_properties = {    8.0 :[ 13.21, 0.2444, 8.5746e-6, 1.5857e-7, 0.237,  77.63 ] }
Element_Al.set_xray_properties(xray_properties)

Element_Si = Element('Si', 14, 28.086, density=2.33)
#                              f1     f2      delta      beta       cri-ang  attenuation(um)
xray_properties = {    8.0 : [ 14.26, 0.3287, 7.6733e-6, 1.7688e-7, 0.224,  69.62 ] ,
                      13.0 : [ 14.13, 0.1275, 2.8804e-6, 2.5978e-8, 0.138,  289.8 ] ,
                      13.5 : [ 14.13, 0.1182, 2.6695e-6, 2.2348e-8, 0.132,  324.0 ] ,
                      14.0 : [ 14.12, 0.1100, 2.4810e-6, 1.9327e-8, 0.128,  360.8 ] ,
                      14.5 : [ 14.11, 0.1025, 2.3118e-6, 1.6796e-8, 0.123,  400.3 ]  }
Element_Si.set_xray_properties(xray_properties)


Element_S = Element('S', 16, 32.066, density=2.05)
#                              f1     f2      delta      beta       cri-ang  attenuation(um)
xray_properties = {   13.0 : [ 16.19, 0.2214, 2.5431e-6, 3.4775e-8, 0.129,  217.1 ] ,
                      13.5 : [ 16.18, 0.2058, 2.3568e-6, 2.9972e-8, 0.124,  242.4 ] ,
                      14.0 : [ 16.17, 0.1918, 2.1902e-6, 2.5968e-8, 0.120,  269.6 ] ,
                      14.5 : [ 16.16, 0.1791, 2.0407e-6, 2.2607e-8, 0.116,  298.7 ]  }
Element_S.set_xray_properties(xray_properties)



Element_Cr = Element('Cr', 24, 51.996, density=7.19)
#                              f1     f2     delta      beta       cri-ang  attenuation(um)
xray_properties = {   13.0 : [ 24.38, 1.062, 8.2806e-6, 3.6065e-7, 0.233,   21.01 ] ,
                      13.5 : [ 24.38, 0.9912, 7.6785e-6, 3.1224e-7, 0.225,   23.37 ] ,
                      14.0 : [ 24.37, 0.9274, 7.1391e-6, 2.7165e-7, 0.217,   25.89 ]  }
Element_Cr.set_xray_properties(xray_properties)


Element_Fe = Element('Fe', 26, 55.845, density=7.87)
xray_properties = {   13.0 : [ 26.33, 1.445, 9.1196e-6, 5.0054e-7, 0.245,   15.15 ] ,
                      13.5 : [ 26.34, 1.352, 8.4620e-6, 4.3413e-7, 0.236,   16.81 ] ,
                      14.0 : [ 26.36, 1.267, 7.8718e-6, 3.7833e-7, 0.227,   18.60 ]  }
Element_Fe.set_xray_properties(xray_properties)

Element_Co = Element('Co', 27, 58.933, density=8.90)
xray_properties = {   13.0 : [ 27.26, 1.633, 1.0115e-5, 6.0580e-7, 0.258,   12.52 ] ,
                      13.5 : [ 27.29, 1.529, 9.3900e-6, 5.2592e-7, 0.248,   13.88 ] ,
                      14.0 : [ 27.32, 1.434, 8.7383e-6, 4.5872e-7, 0.240,   15.34 ]  }
Element_Co.set_xray_properties(xray_properties)

Element_Ni = Element('Ni', 28, 58.693, density=8.90)
xray_properties = {   13.0 : [ 28.18, 1.889, 1.0499e-5, 7.0368e-7, 0.263,   10.78 ] ,
                      13.5 : [ 28.23, 1.770, 9.7522e-6, 6.1155e-7, 0.253,   11.94 ] ,
                      14.0 : [ 28.26, 1.662, 9.0799e-6, 5.3393e-7, 0.244,   13.18 ]  }
Element_Ni.set_xray_properties(xray_properties)

Element_Cu = Element('Cu', 29, 63.546, density=8.96)
xray_properties = {   13.0 : [ 29.05, 2.149, 1.0061e-5, 7.4449e-7, 0.257,   10.19 ] ,
                      13.5 : [ 29.12, 2.017, 9.3531e-6, 6.4773e-7, 0.248,   11.27 ] ,
                      14.0 : [ 29.18, 1.895, 8.714e-6, 5.6608e-7, 0.239,   12.44]  }
Element_Cu.set_xray_properties(xray_properties)

Element_Zn = Element('Zn', 30, 65.390, density=7.13)
xray_properties = {   13.5 : [ 29.97, 2.284, 7.4474e-6, 5.6754e-7, 0.221,   12.87 ] }
Element_Zn.set_xray_properties(xray_properties)

Element_Ga = Element('Ga', 31, 69.723, density=6.09)
xray_properties = {   13.5 : [ 30.74, 2.551, 6.1216e-6, 5.0803e-7, 0.200,   14.38 ] }
Element_Ga.set_xray_properties(xray_properties)

Element_Se = Element('Se', 34, 78.960, density=4.50)
xray_properties = {   13.5 : [ 32.11, 3.496, 4.1693e-6, 4.5385e-7, 0.165,   16.09 ] }
Element_Se.set_xray_properties(xray_properties)

Element_Mo = Element('Mo', 42, 95.940, density=10.2)
xray_properties = {   13.5 : [ 41.14, 1.108, 9.9829e-6, 2.6892e-7, 0.256,   27.12 ] }
Element_Mo.set_xray_properties(xray_properties)

Element_Ru = Element('Ru', 44, 101.070, density=12.4)
xray_properties = {   13.0 : [ 43.41, 1.408, 1.3094e-5, 4.2459e-7, 0.293,   17.85 ] ,
                      13.5 : [ 43.34, 1.317, 1.2123e-5, 3.6827e-7, 0.282,   19.81 ] ,
                      14.0 : [ 43.28, 1.234, 1.1256e-5, 3.2105e-7, 0.272,   21.90 ]  }
Element_Ru.set_xray_properties(xray_properties)

Element_Ag = Element('Ag', 47, 107.868, density=10.5)
xray_properties = {   13.5 : [ 46.59, 1.727, 1.0331e-5, 3.8306e-7, 0.260,   19.05 ] }
Element_Ag.set_xray_properties(xray_properties)

Element_Cd = Element('Cd', 48, 112.411, density=8.65)
xray_properties = {   13.5 : [ 47.64, 1.929, 8.3519e-6, 3.3816e-7, 0.234,   21.58 ] }
Element_Cd.set_xray_properties(xray_properties)

Element_In = Element('In', 49, 114.818, density=7.31)
xray_properties = {   13.0 : [ 48.77, 2.167, 7.6279e-6, 3.3889e-7, 0.224,   22.37 ] ,
                      13.5 : [ 48.71, 2.028, 7.0653e-6, 2.9417e-7, 0.215,   24.81 ] ,
                      14.0 : [ 48.66, 1.903, 6.5623e-6, 2.5665e-7, 0.208,   27.42 ]  }
Element_In.set_xray_properties(xray_properties)

Element_Sn = Element('Sn', 50, 118.710, density=7.30)
xray_properties = {   13.5 : [ 49.80, 2.245, 6.9765e-6, 3.1447e-7, 0.214,   23.21 ] }
Element_Sn.set_xray_properties(xray_properties)



Element_Ta = Element('Ta', 73, 180.948, density=16.7)
xray_properties = {   13.0 : [ 69.88, 10.37, 1.5800e-5, 2.3457e-6, 0.322,   3.235 ] ,
                      13.5 : [ 70.49, 9.812, 1.4779e-5, 2.0572e-6, 0.312,   3.552 ] ,
                      14.0 : [ 70.95, 9.291, 1.3833e-5, 1.8113e-6, 0.301,   3.890 ]  }
Element_Ta.set_xray_properties(xray_properties)

Element_W = Element('W', 74, 183.840, density=19.3)
xray_properties = {   13.0 : [ 70.30, 10.83, 1.8130e-5, 2.7934e-6, 0.345,   2.716 ] ,
                      13.5 : [ 71.08, 10.21, 1.6999e-5, 2.4428e-6, 0.334,   2.991 ] ,
                      14.0 : [ 71.65, 9.648, 1.5934e-5, 2.1456e-6, 0.323,   3.284 ]  }
Element_W.set_xray_properties(xray_properties)

Element_Pt = Element('Pt', 78, 195.078, density=21.5)
xray_properties = {   13.0 : [ 70.28, 8.474, 1.8984e-5, 2.2889e-6, 0.353,   3.315 ] ,
                      13.5 : [ 70.44, 11.08, 1.7646e-5, 2.7760e-6, 0.340,   2.632 ] ,
                      14.0 : [ 71.35, 11.79, 1.6619e-5, 2.7464e-6, 0.330,   2.565 ]  }
Element_Pt.set_xray_properties(xray_properties)

Element_Pb = Element('Pb', 82, 207.200, density=11.4)
xray_properties = {   8.0 : [ 78.29, 9.018, 2.7820e-5, 3.2046e-6, 0.427,   3.848 ] ,
                      13.0 : [ 66.97, 4.126, 9.0119e-6, 5.5522e-7, 0.243,   13.66 ] ,
                      13.5 : [ 73.68, 9.612, 9.1949e-6, 1.1995e-6, 0.246,   6.091 ] ,
                      14.0 : [ 74.86, 9.021, 8.6858e-6, 1.0468e-6, 0.239,   6.730 ]  }
Element_Pb.set_xray_properties(xray_properties)






class Material(object):
    
    avogadro = 6.0221415e23 # 1/mol
    
    def __init__(self, composition, name=None, density=None):
        
        self.composition = composition
        
        if name==None:
            self.name = ''
            for element, amount in self.composition:
                self.name += element.symbol + str(amount)
        else:
            self.name = name
        
        if density==None:
            # Guess density based on elements
            density = 0.0
            total_amt = 0.0
            for element, amount in self.composition:
                density += amount*element.get_density()
                total_amt += amount
            density /= total_amt
        
        self.density = density # g/cm^3
    
    
    def set_density(self, density=1.0):
        self.density = density # g/cm^3
    
    
    def set_density_unitcell(self, unitcell, repeats=1.0):
        
        
        MW = self.get_molecular_weight()*repeats # g/mol
        V = unitcell.get_unitcell_volume() # A^3
        V_cm3 = V*1e-24 # cm^3
        
        self.density = (MW/self.avogadro)/V_cm3
        
        return self.density
    
    
    def get_density(self):
        return self.density # g/cm^3
    
    
    def get_molecular_weight(self):
        MW_total = 0.0
        for element, amount in self.composition:
            MW_total += amount*element.get_molecular_weight()

        return MW_total # g/mol
            
            
    def get_molar_volume(self):
        
        vol_cm3 = self.get_molecular_volume(units='cm^3')*self.avogadro # cm^3
        
        return vol_cm3

        
    def get_molecular_volume(self, units='A^3'):
        
        wt = self.get_molecular_weight()/self.avogadro # g/unit-cell
        
        vol_cm3 = wt/(self.get_density()) # cm^3/unit-cell
        vol_m3 = vol_cm3*0.000001 # m^3
        vol_A3 = vol_m3*1e30 # A^3
        vol_nm3 = vol_A3*0.001 # nm^3
        
        if units=='A^3':
            return vol_A3
        elif units=='nm^3':
            return vol_nm3
        elif units=='cm^3':
            return vol_cm3
        elif units=='m^3':
            return vol_m3
            
        
           
    def get_xray_delta_beta(self, energy=13.0):
        delta = 0.0
        beta = 0.0
        MW_total = 0.0
        for element, amount in self.composition:
            deltacur, betacur = element.get_xray_delta_beta_intrinsic(energy)
            delta += amount*deltacur
            beta += amount*beta
            MW_total += amount*element.get_molecular_weight()
        
        delta *= self.density/MW_total
        beta *= self.density/MW_total
        
        return delta, beta
    
    
    def get_xray_delta_beta_intrinsic(self, energy=13.0):
        """The intrinsic delta and beta are those scaled for the density
        of the material. Thus they are a 'per-atom' contribution."""
        
        delta, beta = self.get_xray_delta_beta(energy)
        
        delta *= self.get_molecular_weight()/self.get_density()
        beta *= self.get_molecular_weight()/self.get_density()
                
        return delta, beta        
                   
                   
    def get_xray_n(self, energy=13.0):
        delta, beta = self.get_xray_delta_beta(energy)
            
        n = 1.0 - delta + beta*1j
        
        return n        

    def get_xray_critical_angle(self, energy=13.0, ambient_n=1.0):
        material_n = self.get_xray_n(energy)
        crit_ang_rad = np.pi/2.0 - np.real( np.arcsin( material_n/ambient_n ) )
        
        return np.degrees(crit_ang_rad)
        
        
    def get_xray_scattering_length(self, energy=13.0):
        
        c = 299792458 # m/s
        charge_electron = 1.60217646e-19 # C
        permitivity_free_space = 8.8541878176e-12 # C^2/Nm^2
        mass_electron = 9.10938188e-31 # kg

        prefactor_m = (charge_electron**2)/( 4*np.pi*permitivity_free_space*mass_electron*(c**2) ) # meters
        prefactor_fm = prefactor_m*1e15 # femtometers
        
        b_total = 0
        for element, amount in self.composition:
            f1cur, f2cur = element.get_f1_f2(energy=energy)
            b_cur = f1cur*prefactor_fm
            b_total += amount*b_cur
                
        return b_total # fm
        
        
    def get_xray_scattering_length_density(self, energy=13.0):
        return get_xray_SLD(energy=energy)
        
        
    def get_xray_SLD(self, energy=13.0):
        
        b_fm = self.get_xray_scattering_length()
        b_A = b_fm*0.00001 # Angstroms
        
        SLD_A = b_A/self.get_molecular_volume(units='A^3') # A^-2
        SLD = SLD_A*1e6 # 10^-6 A^-2
        
        return SLD # 10^-6 A^-2
    
    
    def get_xray_attenuation_length(self, energy=13.0):
        
        mu_total = 0.0
        
        for element, amount in self.composition:
            
            mu_cur = element.get_xray_mass_absorption_coefficient(energy=energy)

            mu_total += amount*mu_cur
            
        return 1.0/(mu_total)
    
    
    def get_xray_mass_absorption_coefficient(self, energy=13.0):
        
        att_len = self.get_xray_attenuation_length(energy=energy)
        density = self.get_density()
        
        mu = 1.0/( att_len * density )
        
        return mu
    
    
    def to_string(self):
        
        return self.name

        
class Blend(Material):
    
    def __init__(self, composition, name=None, density=None):
        self.composition = composition

        if name==None:
            self.name = ''
            for material, amount in self.composition:
                self.name += material.to_string()+ str(amount)
        else:
            self.name = name
            
        if density==None:
            # Guess density based on composition
            density = 0.0
            total_amt = 0.0
            for material, amount in self.composition:
                density += amount*material.get_density()
                total_amt += amount
            density /= total_amt
        
        self.density = density # g/cm^3

       


Material_Vacuum = Material( [[Element_H, 1.0]], name='Vacuum', density=0.00 )
Material_Water = Material( [[Element_H, 2.0], [Element_O, 1.0]], name='Water', density=1.00 )

Material_Myoglobin = Material( [[Element_C, 774.0], [Element_H, 12242.0], [Element_N, 210.0], [Element_O, 222.0], [Element_S, 2.0]], name='Myoglobin', density=1.35 )
Material_DNA_double_helix = Material( [[Element_C, 20.0], [Element_H, 23.0], [Element_N, 210.0], [Element_O, 12.0], [Element_N, 7.0]], name='dsDNA', density=1.70 )

Material_Al2O3 = Material( [[Element_Al, 2.0], [Element_O, 3.0]], name='Al2O3', density=3.98 )
Material_Si = Material( [[Element_Si, 1.0]], name='Silicon' )
Material_SiO2 = Material( [[Element_Si, 1.0], [Element_O, 2.0]], name='SiO2', density=2.65 )
Material_SiN = Material( [[Element_Si, 3.0], [Element_N, 4.0]], name='Silicon nitride', density=3.44 )

Material_Polystyrene = Material( [[Element_C, 8.0],[Element_H, 8.0]], name='PS', density=1.06 )
Material_PMMA = Material( [[Element_C, 5.0],[Element_O, 2.0], [Element_H, 8.0]], name='PMMA', density=1.18 )

BCP_f = 0.01
Material_BCP = Blend( [ [Material_Polystyrene, BCP_f] , [Material_PMMA, (1.0-BCP_f)] ], name='Block copolymer', density=1.5 )




Material_Rubrene = Material( [[Element_C, 42.0],[Element_H, 28.0]], name='Rubrene', density=1.263 )
Rubrene = UnitCell( 26.86, 7.193, 14.433, 90.0, 90.0, 90.0 )
#Material_Rubrene.set_density_unitcell( Rubrene, repeats=4.0  )


Material_BN = Material( [[Element_B, 1.0],[Element_N, 1.0]], name='Boron Nitride', density=2.1 )
BoronNitride = UnitCell( 2.51, 2.51, 6.69, 90.0, 90.0, 120.0 )
#Material_BN.set_density_unitcell( BoronNitride, repeats=1.0 )



Material_magnetic = Material( [[Element_Co, 1.0],[Element_Cr, 18.0],[Element_Pt, 12.0]], name='maglayer', density=8.9 ) # Density just a guess
Material_TaOx = Material( [[Element_Ta, 2.0],[Element_O, 5.0]], name='TaOx', density=8.20 )
Material_Ru = Element_Ru
Material_NiW = Material( [[Element_Ni, 1.0],[Element_W, 1.0]], name='NiW', density=8.902 ) # Density just a guess
Material_TaNi = Material( [[Element_Ta, 1.0],[Element_Ni, 1.0]], name='TaNi', density=16.654 ) # Density just a guess
Material_CoFe = Material( [[Element_B, 1.0],[Element_N, 1.0]], name='CoFe', density=8.9 ) # Density just a guess


Material_Ag = Element_Ag
Material_Ni = Element_Ni
Material_ITO = Material( [[Element_In, 2.0*0.9],[Element_O, 3.0*0.9],[Element_Sn, 1.0*0.1],[Element_O, 2.0*0.1]], name='ITO', density=7.14 )
Material_ZnO = Material( [[Element_Zn, 1.0],[Element_O, 18.0]], name='ZnO', density=5.61 )
Material_CdS = Material( [[Element_Cd, 1.0],[Element_S, 18.0]], name='CdS', density=4.82 )
Material_CIGS = Material( [[Element_Cu, 0.9],[Element_In, 0.7],[Element_Ga, 0.3],[Element_Se, 2.0]], name='CIGS', density=5.69 )
Material_Mo = Element_Mo



if False:
    # Example: Calculate SLDs and expected contrast...
    
    
    
    energy = 13.5
    
    # Excellent contrast
    sld_Vac = 0.0
    sld_grating = Material_Si.get_xray_SLD(energy=energy)
    print( 'vacuum SLD = %.2f *10^(-6) A^(-2)' % (sld_Vac) )
    print( 'grating SLD = %.2f *10^(-6) A^(-2)' % (sld_grating) )
    print( 'contrast = %.2f *10^(-6) A^(-2)\n' % ( abs(sld_grating-sld_Vac) ) )
    
   
    # BCP contrast
    materials = [ Material_Polystyrene, Material_PMMA ]
    print( '%s SLD = %.2f *10^(-6) A^(-2)' % (materials[0].to_string(), materials[0].get_xray_SLD(energy=energy) ) )
    print( '%s SLD = %.2f *10^(-6) A^(-2)' % (materials[1].to_string(), materials[1].get_xray_SLD(energy=energy) ) )
    print( 'contrast = %.2f *10^(-6) A^(-2)\n' % ( abs(materials[0].get_xray_SLD(energy=energy)-materials[1].get_xray_SLD(energy=energy)) ) )

    
    # Weak contrast
    materials = [ Material_Water, Material_Myoglobin ]
    print( '%s SLD = %.2f *10^(-6) A^(-2)' % (materials[0].to_string(), materials[0].get_xray_SLD(energy=energy) ) )
    print( '%s SLD = %.2f *10^(-6) A^(-2)' % (materials[1].to_string(), materials[1].get_xray_SLD(energy=energy) ) )
    print( 'contrast = %.2f *10^(-6) A^(-2)\n' % ( abs(materials[0].get_xray_SLD(energy=energy)-materials[1].get_xray_SLD(energy=energy)) ) )

    materials = [ Material_Water, Material_DNA_double_helix ]
    print( '%s SLD = %.2f *10^(-6) A^(-2)' % (materials[0].to_string(), materials[0].get_xray_SLD(energy=energy) ) )
    print( '%s SLD = %.2f *10^(-6) A^(-2)' % (materials[1].to_string(), materials[1].get_xray_SLD(energy=energy) ) )
    print( 'contrast = %.2f *10^(-6) A^(-2)\n' % ( abs(materials[0].get_xray_SLD(energy=energy)-materials[1].get_xray_SLD(energy=energy)) ) )

    

    Material_P3HT = Material( [[Element_C, 10.0],[Element_H, 14.0], [Element_S, 1.0]], name='P3HT', density=1.33 )
    Material_P3OT = Material( [[Element_C, 12.0],[Element_H, 18.0], [Element_S, 1.0]], name='P3OT', density=1.33 )
    Material_CarbonDisulfide = Material( [[Element_C, 1.0],[Element_S, 2.0]], name='CS2', density=1.261 )
    
    sld_P3HT = Material_P3HT.get_xray_SLD(energy=energy)
    sld_P3OT = Material_P3OT.get_xray_SLD(energy=energy)
    sld_CS2 = Material_CarbonDisulfide.get_xray_SLD(energy=energy)
    print( 'P3HT SLD = %.2f *10^(-6) A^(-2)' % (sld_P3HT) )
    print( 'P3OT SLD = %.2f *10^(-6) A^(-2)' % (sld_P3OT) )
    print( 'CS2 SLD  = %.2f *10^(-6) A^(-2)' % (sld_CS2) )
    print( 'contrast = %.2f *10^(-6) A^(-2)\n' % ( abs(sld_CS2-sld_P3OT) ) )
    

if False:
    
    # Example: Consider critical angles
    energy = 13.5
    
    layers = [Material_Vacuum, Material_Rubrene, Material_BN, Material_SiO2, Material_Si]
    
    
    layers = [
                Material_Vacuum ,       # Ambient
                Material_SiN ,          # Top (protective) layer
                Material_magnetic ,     # Magnetic layer
                Material_TaOx ,         # Tantalum oxide (Ta2O5) amorphous epitaxial-control layer
                Material_Ru ,           # Ruthenium (epitaxy-directing layer)
                Material_NiW ,          # NiW
                Material_TaNi ,         # TaNi
                Material_CoFe ,         # SUL (soft magnetic underlayer)
                Material_SiO2 ,         # Glass substrate
            ]
    
    
    for i, layer in enumerate(layers):
        name = layer.to_string()
        layer_n = np.real(layer.get_xray_n(energy=energy))
        
        print( '%d. n = %.10f \t vac crit ang: %.3f\t\t%s' % (i, layer_n, layer.get_xray_critical_angle( energy=energy, ambient_n=1.0 ), name) )
    
    
    for i, layer in enumerate(layers[1:]):
        name = layer.to_string()
        
        prev_layer = layers[i]
        prev_layer_n = prev_layer.get_xray_n(energy=energy)
        
        layer_n = layer.get_xray_n(energy=energy)
        
        internal_crit_ang = layer.get_xray_critical_angle( energy=energy, ambient_n=prev_layer_n )
        internal_crit_ang_rad = np.radians(internal_crit_ang)
        external_crit_ang = np.degrees( np.real( np.arccos( np.cos(internal_crit_ang_rad)*prev_layer_n/1.0000 ) ) ) # Snell's law (cosine form)
        
        # Back-chain all the way up to vacuum
        ic = i
        angle_cur_rad = internal_crit_ang_rad
        angle_prev_layer_rad = internal_crit_ang_rad
        while ic>=1:
            cur_layer = layers[ic]
            cur_layer_n = cur_layer .get_xray_n(energy=energy)
            
            prev_layer = layers[ic-1]
            prev_layer_n = prev_layer.get_xray_n(energy=energy)
            
            angle_prev_layer_rad = np.real( np.arccos( np.cos(angle_cur_rad)*cur_layer_n/prev_layer_n ) ) # Snell's law (cosine form)
            
            angle_cur_rad = angle_prev_layer_rad 
            ic -= 1
            
        
        print( '%d. interface crit ang: %.3f \t (external: %.3f)\t\t%s' % (i+1, internal_crit_ang, external_crit_ang, name  ) )
        print( '\t\t top of stack: %.3f' % ( np.degrees(angle_prev_layer_rad) ) )
        
    
if False:
    
    # Example: Consider critical angles
    energy = 13.5
    
    layers = [
                Material_Vacuum ,       # Ambient
                Material_Ag,            # Contact
                Material_Ni,            # Contact
                Material_ITO,           # Electrode
                Material_ZnO,
                Material_CdS,
                Material_CIGS,
                Material_Mo,            # Electrode
                Material_SiO2 ,         # Glass substrate
            ]
    
    
    for i, layer in enumerate(layers):
        name = layer.to_string()
        layer_n = np.real(layer.get_xray_n(energy=energy))
        
        print( '%d. n = %.10f \t vac crit ang: %.3f\t\t%s' % (i, layer_n, layer.get_xray_critical_angle( energy=energy, ambient_n=1.0 ), name) )
    
    
    for i, layer in enumerate(layers[1:]):
        name = layer.to_string()
        
        prev_layer = layers[i]
        prev_layer_n = prev_layer.get_xray_n(energy=energy)
        
        layer_n = layer.get_xray_n(energy=energy)
        
        internal_crit_ang = layer.get_xray_critical_angle( energy=energy, ambient_n=prev_layer_n )
        internal_crit_ang_rad = np.radians(internal_crit_ang)
        external_crit_ang = np.degrees( np.real( np.arccos( np.cos(internal_crit_ang_rad)*prev_layer_n/1.0000 ) ) ) # Snell's law (cosine form)
        
        # Back-chain all the way up to vacuum
        ic = i
        angle_cur_rad = internal_crit_ang_rad
        angle_prev_layer_rad = internal_crit_ang_rad
        while ic>=1:
            cur_layer = layers[ic]
            cur_layer_n = cur_layer .get_xray_n(energy=energy)
            
            prev_layer = layers[ic-1]
            prev_layer_n = prev_layer.get_xray_n(energy=energy)
            
            angle_prev_layer_rad = np.real( np.arccos( np.cos(angle_cur_rad)*cur_layer_n/prev_layer_n ) ) # Snell's law (cosine form)
            
            angle_cur_rad = angle_prev_layer_rad 
            ic -= 1
            
        
        print( '%d. interface crit ang: %.3f \t (external: %.3f)\t\t%s' % (i+1, internal_crit_ang, external_crit_ang, name  ) )
        print( '\t\t top of stack: %.3f' % ( np.degrees(angle_prev_layer_rad) ) )
        
    
    for i, layer in enumerate(layers):
        name = layer.to_string()
        layer_n = np.real(layer.get_xray_n(energy=energy))
        crit_ang = layer.get_xray_critical_angle( energy=energy, ambient_n=1.0 )
        att_len = layer.get_xray_attenuation_length(energy=energy)
        
        print( '%d. n = %.10f \t vac crit ang = %.3f deg\t atten = %05.1f um\t%s' % (i, layer_n, crit_ang, att_len, name) )
        
    
    
    

    
if True:
    # Example: Plot GISAXS/GTSAXS peaks (two-beam approximation)
    
    

    # Strongest peaks
    peaks_present = None
    peaks_present = [ [1,0,0], [2,0,0], [3,0,0], [4,0,0] ] # Lamellae
    
    
    
    # Ia3d cubic gyroid
    #peaks_present = [ [2,1,1], [2,2,0], [3,2,1], [4,0,0], [4,2,0], [3,3,2], [4,2,2], [4,3,1] ] 
    
    
    #peaks_present = [ [-1,1,0], [0,1,-1], [1,0,-1], [1, -1,0], [-1,2,-1]] 
    
    if True:
        # Permute ordering
        peaks_present_base = peaks_present
        peaks_present = []
        for a, b, c in peaks_present_base:
            
            peaks_present.append([a,b,c])
            peaks_present.append([a,c,b])
            peaks_present.append([b,a,c])
            peaks_present.append([b,c,a])
            peaks_present.append([c,a,b])
            peaks_present.append([c,b,a])

    
    if True:
        # Permute signs
        peaks_present_base = peaks_present
        peaks_present = []
        for h, k, l in peaks_present_base:
            
            peaks_present.append([+h,+k,+l])
            peaks_present.append([+h,+k,-l])
            peaks_present.append([+h,-k,+l])
            peaks_present.append([+h,-k,-l])
            
            peaks_present.append([-h,+k,+l])
            peaks_present.append([-h,+k,-l])
            peaks_present.append([-h,-k,+l])
            peaks_present.append([-h,-k,-l])
        
    
    #peaks_present_abs = []
    #for peak in peaks_present:
        #h, k, l = peak
        #peaks_present_abs.append( [abs(h), abs(k), abs(l)] )
    
    theta_error = 0.035
    theta_target = 0.15
    e = EwaldSphere( wavelength=0.9184, theta_incident=theta_target+theta_error ) # 13.5 keV

    #plot_region = [-0.25, 0.02, 0.0, 0.30]
    #plot_buffers = [0.30,0.05,0.25,0.05]
    #plot_region = [-0.3, 0.3, 0, 0.4] # KY
    #plot_buffers = [0.20, 0.02,0.25,0.05] # KY

    plot_region = [-0.3, 0.3, -0.2, 0.4] 
    plot_buffers = [0,0,0,0] #[0.30, 0.05, 0.25, 0.05]
    
    dpi = 200
    
    
    # unit cell
    ###################################################################    
    print( '\n\nUnit Cell ----------------------------------------' )
    
    # Hex
    if 1:
        cyl_repeat_nm = 5.45
        cyl_repeat_A = cyl_repeat_nm*10.0
        Cell = UnitCell( cyl_repeat_A, cyl_repeat_A, 0.01, 90.0, 90.0, 120.0 )
    
    else:
        cyl_repeat_nm = 7.7
        cyl_repeat_A = cyl_repeat_nm*10.0
        
        Cell = UnitCell( cyl_repeat_A, cyl_repeat_A, cyl_repeat_A, 90.0, 90.0, 90.0 )
        
    #Cell.set_rotation_angles(eta=0.0, phi=0.0, theta=0.0) # Reset orientation
    
    #  a-axis is in x,y plane
    #  b-axis points along +y
    #  c-axis points along +z


    #Cell.apply_rotation_x( -90.0 )
    # Now:
    #  a-axis in x,z plane
    #  b-axis along z
    #  c-axis points along -y    

    # 111 vertical
    #Cell.apply_rotation_z(-45)
    #Cell.apply_rotation_y(-54.736)
    
    # 211 vertical
    #Cell.apply_rotation_z(-26.565)
    #Cell.apply_rotation_y(-65.905)
    
    
    #Cell.apply_rotation_y(45)
    #Cell.apply_rotation_x(45)
    
        
    
    #h, k, l = 1, 0, 0 # a
    #h, k, l = 0, 1, 0 # b
    h, k, l = 0, 0, 1 # c
    ##h, k, l = 2, 1, 1
    #h, k, l = 1, 1, 1
    
    qhkl, (qx, qy, qz), qxy, angle_wrt_x, angle_wrt_z = Cell.print_q_hkl_exp(h, k, l)
    
    #im_dir = './'
    #infile = im_dir + 'c91.png'
    im_dir = '/home/etsai/BNL/Users/CMS/LSita/2021C2/LSita/saxs/analysis/Bar3_offline/Index/'
    infile = im_dir + 'Bar3s1_thermal_CW03-34_190nm_vac_chp2_8265.6s_T100.673C_th0.150_x-0.371_y0.000_20.00s_009508_saxs.png'
    infile = im_dir + 'Run1_thermal_CW03-58_160nm_vac_chp1_6609.6s_T170.224C_th0.150_x-0.750_y0.000_20.00s_002161_saxs.png'
  
       
    #Cell.list_powder_peaks(filename='peaks.dat', max_hkl=10)
    
    #Cell.plot_exp_inplane_powder(filename='unitcell-powder.png', plot_region=plot_region, plot_buffers=plot_buffers, label_peaks=True, max_hkl=10, peaks_present=peaks_present)
    #im_dir = './q_images/'
    #save_dir = './exp_inplane/'


    #Cell.plot_ewald_inplane_powder(e, filename='ewald.png', plot_region=plot_region, plot_buffers=plot_buffers, label_peaks=True, blanked_figure=True, max_hkl=5, thresh=0.005, peaks_present=peaks_present, dpi=dpi)
    #im_dir = './qr_images/'
    #save_dir = './ewald_inplane/'


    #Cell.plot_ewald(e, filename='ewald.png', plot_region=plot_region, plot_buffers=plot_buffers, label_peaks=True, blanked_figure=True, max_hkl=5, thresh=0.5, peaks_present=peaks_present, dpi=dpi)
    #im_dir = './qr_images/'
    #save_dir = './ewald/'

    #Cell.plot_ewald_qxqz(e, filename='ewald.png', plot_region=plot_region, plot_buffers=plot_buffers, label_peaks=True, blanked_figure=True, max_hkl=5, thresh=0.25, peaks_present=peaks_present, dpi=dpi)
    #im_dir = './q_images/'
    #save_dir = './ewald_qxqz/'


    #infile = im_dir + 'run04_kinetics-120C_150nm-chip1_th0.150_85.0s_T86.340C_10.00s_73365_saxs.png'
    #outfile =  save_dir + infile[len(im_dir):-4] + '-overlay.png'
    #cmd = 'composite -gravity center ewald.png ' + infile + ' ' + outfile
    #os.system(cmd)
    
    

    #Cell.plot_ewald_two_beam(e, Material_Vacuum, Material_BCP, Material_Si, filename='ewald-two_beam.png', plot_region=plot_region, plot_buffers=plot_buffers, label_peaks=True, blanked_figure=False, max_hkl=8, thresh=0.1)

    #Cell.plot_ewald_DWBA(e, Material_Vacuum, Material_BCP, Material_Si, filename='ewald-DWBA.png', plot_region=plot_region, plot_buffers=plot_buffers, label_peaks=True, blanked_figure=True, max_hkl=8, thresh=0.1)

    Cell.plot_ewald_two_beam_qxqz(e, Material_Vacuum, Material_BCP, Material_Si, filename='ewald-two_beam.png', plot_region=plot_region, plot_buffers=plot_buffers, peaks_present=peaks_present, label_peaks=False, blanked_figure=True, max_hkl=8, thresh=0.1, dpi=dpi)
    #im_dir = './q_images/'
    #save_dir = './'

    save_dir = im_dir #'/home/etsai/BNL/Users/CMS/LSita/2021C2/LSita/saxs/analysis/Bar3_offline/'


    if 1:
        cyl_repeat_nm = 9
        cyl_repeat_A = cyl_repeat_nm*10.0    
        Cell = UnitCell( cyl_repeat_A, cyl_repeat_A, cyl_repeat_A, 90.0, 90.0, 90.0 )
        qhkl, (qx, qy, qz), qxy, angle_wrt_x, angle_wrt_z = Cell.print_q_hkl_exp(h, k, l)    
        Cell.plot_ewald_two_beam_qxqz(e, Material_Vacuum, Material_BCP, Material_Si, filename='ewald-two_beam2.png', plot_region=plot_region, plot_buffers=plot_buffers, peaks_present=peaks_present, label_peaks=False, blanked_figure=True, max_hkl=8, thresh=0.1, dpi=dpi)  
    
        
    outfile =  save_dir + infile[len(im_dir):-4] + '-overlay.png'
    cmd = 'composite -gravity center ewald-two_beam.png ' + infile + ' ' + outfile
    os.system(cmd)

    outfile2 =  save_dir + infile[len(im_dir):-4] + '-overlay2.png'
    cmd = 'composite -gravity center ewald-two_beam2.png ' + outfile + ' ' + outfile2
    os.system(cmd)    

    if 1:
        rot_array = np.linspace(0, 90, num=20)
        rot_array = [90]
        for i, angle in enumerate(rot_array):
            print('angle = {:.3f}'.format(angle))
            Cell.set_rotation_angles(eta=0.0, phi=0.0, theta=0.0) # Reset orientation
            
            # a points +x (horizontally)
            Cell.apply_rotation_y(90)
            # a points +z (vertically)
            
            Cell.apply_rotation_y(angle)

            Cell.plot_ewald_two_beam_qxqz(e, Material_Vacuum, Material_BCP, Material_Si, filename='frame.png'.format(int(angle*10)), plot_region=plot_region, plot_buffers=plot_buffers, peaks_present=peaks_present, label_peaks=False, blanked_figure=True, max_hkl=8, thresh=0.1, dpi=dpi)
            
            if i>0:
                cmd = 'composite -gravity center frame.png combinedlast.png combined.png'
                os.system(cmd)
            else:
                os.system('cp frame.png combined.png')
            os.system('cp combined.png combinedlast.png')
            
            
        #im_dir = './q_images/'
        #save_dir = './ewald_two_beam_qxqz/'

        #infile = im_dir + 'run04_kinetics-120C_150nm-chip1_th0.150_85.0s_T86.340C_10.00s_73365_saxs.png'
        outfile =  save_dir + infile[len(im_dir):-4] + '-overlay_isotropic.png'
        cmd = 'composite -gravity center combined.png ' + infile + ' ' + outfile
        os.system(cmd)
            

    
    