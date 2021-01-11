import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy, time
from PIL import Image
from math import *
from random import *
from scipy import linalg
import pandas as pds
import fabio
from math import *
import scipy.signal


#################################
# Some notes
#################################
def get_hint(spacegroup='BCC'):
    print('{}:'.format(spacegroup))
    if spacegroup=='HCP':
        print('c~sqrt(8/3)a=1.633a, alpha=90, beta=90, gamma=120')
    elif spacegroup=='BCC':
        print('shortest a*sqrt(3)/2=0.866a, density sqrt(3)pi/8~68%')
    elif spacegroup=='FCC':
        print('shortest a/sqrt(2)=0.707a, density sqrt(2)pi/6~74%')
        
    print('-----------------')
        
        
#################################
# Get list of indices
#################################
def get_index(a=10,b=10,c=10,alp_deg=90,beta_deg=90,gam_deg=90,**kwargs):
    alp = radians(alp_deg)
    beta = radians(beta_deg)
    gam = radians(gam_deg)
               
    spacegroup = kwargs['spacegroup']
    ori_hkl = kwargs['ori_hkl']
    if "range_hkl" in kwargs:
        range_hkl = kwargs['range_hkl']
    else: range_hkl = [[0,2], [0,2], [0,2]]
        
    geometry = kwargs['geometry']
    if geometry=='GI':
        GI = 1
    else:
        GI = 0
    lambda_A = kwargs['lambda_A']    
    Inci = radians(kwargs['inc_theta_deg'][0])
    Inci_c = radians(kwargs['inc_theta_deg'][1])
    Inci_s = radians(kwargs['inc_theta_deg'][2])
    verbose = kwargs['verbose']
    
    if verbose:
        print('ori_hkl = {}, spacegroup = {}'.format(ori_hkl,spacegroup))
        print('(a,b,c)=({:.2f}, {:.2f}, {:.2f}) A, (alp,beta,gam)=({:.2f}, {:.2f}, {:.2f}) deg'.format(a,b,c,alp_deg,beta_deg,gam_deg))
        #print('lambda={:.2f}A, Inci={}deg'.format(lambda_A, kwargs['inc_theta_deg'][0]))

    #Lattice calculation
    lattice=[a,b,c,alp,beta,gam]
    V=a*b*c*sqrt(1+2*cos(alp)*cos(beta)*cos(gam)-cos(alp)**2-cos(beta)**2-cos(gam)**2)

    #reciprocal lattice
    ar=2*pi*b*c*sin(alp)/V
    br=2*pi*a*c*sin(beta)/V
    cr=2*pi*a*b*sin(gam)/V

    alpr=acos((cos(gam)*cos(beta)-cos(alp))/abs(sin(gam)*sin(beta)))
    betar=acos((cos(alp)*cos(gam)-cos(beta))/abs(sin(alp)*sin(gam)))
    gamr=acos((cos(alp)*cos(beta)-cos(gam))/abs(sin(alp)*sin(beta)))

    #rint('reciprocal lattice:\n a = {}  b = {} c = {} \n {}  {}  {}'.format(ar,br,cr,degrees(alpr), degrees(betar), degrees(gamr)))

    #reciprocal space vector
    As=np.array([ar, 0, 0]).reshape(3,1)
    Bs=np.array([br*cos(gamr), br*sin(gamr), 0]).reshape(3,1)
    Cs=np.array([cr*cos(betar), (-1)*cr*sin(betar)*cos(alp), 2*pi/c]).reshape(3,1)
    #As=np.array([ar, br*cos(gamr), cr*cos(betar)]).reshape(3,1)
    #Bs=np.array([0, br*sin(gamr), (-1)*cr*sin(betar)*cos(alp)]).reshape(3,1)
    #Cs=np.array([0, 0, 2*pi/c]).reshape(3,1)
    
    #rint('reciprocal space vector :\n {} \n {} \n {} '.format(As, Bs, Cs))

    #preferential reflections //to surface normal as z
    H = ori_hkl[0]
    K = ori_hkl[1]
    L = ori_hkl[2]
    G=H*As+K*Bs+L*Cs
    tol=0.01
    for index, item in enumerate(G):
        if abs(item) < tol:
            G[index] = 0


    phi=atan2(G[1],G[0])
    chi=acos(G[2]/sqrt(G[0]**2+G[1]**2+G[2]**2))

    #R is the rotation matrix from reciprocal to surface normal
    #R=np.array([[cos(-chi),0,sin(-chi)], [0, 1, 0], [-sin(-chi),0,cos(-chi)]]).dot(np.array([[cos(-phi),sin(phi),0],[-sin(phi),cos(-phi),0],[0,0,1]]))

    #R=np.array([[cos(chi),0,sin(phi)], [0, 1, 0], [-sin(phi),0,cos(chi)]]).dot(np.array([[cos(phi),sin(phi),0],[-sin(phi),cos(phi),0],[0,0,1]]))

    R1 = np.array([[cos(-chi),0,sin(-chi)], [0, 1, 0], [-sin(-chi),0,cos(-chi)]])
    R2 = np.array([[cos(-phi),-sin(-phi),0],[sin(-phi),cos(-phi),0],[0,0,1]])
    R = np.matmul(R1, R2)
    #print(R)

    #rotated reciprocal lattice vectors
    AR=np.matmul(R, As)
    BR=np.matmul(R, Bs)
    CR=np.matmul(R, Cs)
        
    
    ##############
    check_ref(0,0,1,spacegroup=spacegroup)
    
    ##############

    Qxy_list=[]
    Qz_list=[]
    phi_list=[]
    nu_list=[]
    h_list=[]
    k_list=[]
    l_list=[]
    d_list = []
    
    q_data = pds.DataFrame()

    hkl_list=[]
    count = 0
    for h in range(range_hkl[0][0], range_hkl[0][1]+1):
        for k in range(range_hkl[1][0], range_hkl[1][1]+1):
            for l in range(range_hkl[2][0], range_hkl[2][1]+1):
                if GI == 1 and h+l<12:
                    temp = 1 
                else:
                    temp = (np.sum(np.array([h,k,l])*np.array(ori_hkl))==0)
           
                if check_ref(h, k, l, spacegroup) and temp:
                    
                    d = d_rule(lattice=spacegroup, a=a, h=h, k=k, l=l)
                    q = 2*np.pi/d
                    if count==0:
                        q0 = q
                        count = 1
                    q_ratio = q/q0
                        
                    hkl = str(str(h) + str(k) + str(l))                    
                    
                    data = [{'hkl': hkl,'d':d, 'q':q, 'q_ratio':q_ratio}]
                    df = pds.DataFrame(data)
                    q_data = q_data.append(df, ignore_index=True)
                    
                    #Q meet the reciprocal space
                    Q=h*AR+k*BR+l*CR

                    #plot Qxy and Qz position
                    if GI == 1:
                        Qxy=sqrt(Q[0]**2+Q[1]**2)
                        Qz=Q[2]
                    else: 
                        Qxy = Q[0]
                        Qz = Q[1]
                    
                    #save data
                    Qxy_list.append(Qxy)
                    Qz_list.append(Qz)
                    h_list.append(h)
                    k_list.append(k)
                    l_list.append(l)
                    hkl_list.append(hkl)
                    d_list.append(d)
                    
                    #transfer to space geometry
                    #try:
                    if False:
                        beta=asin(Qz/2/pi*lambda_A-sin(Inci))

                        beta_n=asin(sqrt(sin(beta)**2+sin(Inci_c)**2))
                        theta=asin(Qxy/4/pi*lambda_A)
                        if (cos(Inci)**2+cos(beta)**2-4*sin(theta)**2)>(2*cos(Inci)*cos(beta)):
                            phi = acos(1)
                        else:
                            phi=acos((cos(Inci)**2+cos(beta)**2-4*sin(theta)**2)/(2*cos(Inci)*cos(beta)))
                        phi_n=atan(tan(phi)/cos(Inci))
                        nu_n=beta_n+atan(tan(Inci)*cos(phi_n))
                        #save data
                        phi_list.append(phi_n)
                        nu_list.append(nu_n)

                    #except:
                    #    print('math error. ')
                    
    return hkl_list, Qxy_list, Qz_list, q_data    


#################################
# Plot index
#################################
def plot_index(data, Qxy_list, Qz_list, hkl_list, **param_plot):

    if data:
        img = data['img']
        x_axis = data['x_axis']
        y_axis = data['y_axis']
        X, Y = np.meshgrid(x_axis, y_axis)
        if 'img_use' in data:
            img_use = data['img_use']
            flag = data['plot_img_use']
            
    if 'cmap' in param_plot: cmap=param_plot['cmap']
    else: cmap='plasma'

    if 'log10' in param_plot and param_plot['log10']: log10=1
    else: log10=0
    
    if 'lim1' in param_plot: lim1 = param_plot['lim1']
    else: lim1 = [0, 2]
    if 'lim2' in param_plot: lim2 = param_plot['lim2']
    else: lim2 = [0, 2] 
        
    if 'textcolor' in param_plot:
        color = param_plot['textcolor']
    else:
        color = 'w'
    
    if 'FS' in param_plot:
        FS = param_plot['FS']
    else:
        FS = 12
    
    
    fig = plt.gcf()
    #fig.suptitle('GID in q-space', fontsize=15, fontweight='bold')

    ax1 = fig.add_subplot(111)    
    if data:
        if flag: img_plot = img_use
        else: img_plot = img
            
        if log10==1: img_plot = np.log10(img_plot)
        
        if 'vmin' in param_plot: vmin = param_plot['vmin']
        else: vmin = np.min(img_plot)
        if 'vmax' in param_plot: vmax = param_plot['vmax']
        else: vmax = np.max(img_plot)*0.98
    
        plt.pcolormesh(X,Y,(img_plot), vmin=vmin, vmax=vmax, cmap=cmap, alpha = 1); plt.colorbar()
        
    plt.plot(Qxy_list, Qz_list,'cx',markeredgecolor='c',markersize=6)
    plt.xlim(lim1[0], lim1[1])
    plt.ylim(lim2[0], lim2[1])
    plt.xlabel('q',fontsize=15, fontweight='bold')
    
    if not data: plt.grid()
    ax1.set_aspect('equal', 'box')
    for i, txt in enumerate(hkl_list):
        #plt.annotate(txt, (Qxy_list[i], Qz_list[i]*(1+random()/6)),color=color,fontsize=FS, fontweight='bold')
        plt.annotate(txt, (Qxy_list[i]+0.005*2, Qz_list[i]+0.005*2) ,color=color,fontsize=FS, fontweight='bold')

    if 0:
        ax2 = fig.add_subplot(122)
        if data:
            if flag: img_plot = img_use
            else: img_plot = img
            plt.pcolormesh(X,Y,np.log10(img_plot), vmin=vmin, vmax=vmax, cmap=mpl.cm.plasma, alpha = 1); plt.colorbar()
        plt.plot(Qxy_list,Qz_list,'ro',markeredgecolor='r',markersize=3)
        plt.xlim(lim1[0], lim1[1])
        plt.ylim(lim2[0], lim2[1])
        if not data: plt.grid()
        ax2.set_aspect('equal', 'box')
        for i, txt in enumerate(hkl_list):
            plt.annotate(txt, (Qxy_list[i], Qz_list[i]) ,color=color,fontsize=FS, fontweight='bold')


    plt.show()
    
#################################
# See: http://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-hkl?gnum=225
# https://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-table
#
# TO-DO: combine this with function index_rule
#################################
def check_ref(h, k, l, spacegroup):
    if spacegroup == 14:
        if h!=0 and k==0 and l!=0:
            if l%2 ==0:
                return True
        if h==0 and k!=0 and l==0:        
            if k%2 ==0:
                return True
        if h==0 and k==0 and l!=0:        
            if l%2 ==0:
                return True
        if (k+l)%2 ==0:
            return True
        if (k+h)%2 ==0:
            return True
        

    elif spacegroup == 139: 
        if (h+k+l)%2==0:
            return True
        elif (h+k)%2==0 and l==0:
            return True
        elif (k+l)%2==0 and h==0:
            return True        
        elif h==k and l%2==0:
            return True    
        elif h==0 and k==0 and l%2==0:
            return True    
        elif h%2==0 and k==0 and l==0:
            return True    
        
    elif spacegroup == 194 or spacegroup=='HCP': 
        if h==k and l%2==0:
            return True
        if l%2==0 or (h-k-1)%3==0 or (h-k-2)%3==0:
            return True        

    elif spacegroup == 198 or spacegroup=='P213': #P213 
        if h%2==0 and k==0 and l==0:
            return True

    elif spacegroup == 4 or spacegroup=='P21': 
        if h==0 and k!=0 and l==0:
            if k%2==0:
                return True
        else:
            return True
                
        
    elif spacegroup == 225 or spacegroup=='FCC': 
       
        if check_odd(h) and check_odd(k) and check_odd(l):
            return True
        elif check_even(h) and check_even(k) and check_even(l):
            return True
        
    elif spacegroup == 229 or spacegroup=='BCC': 
        if (h+k+l)%2 == 0:
                return True
        if h==0 and (k+l)%2 == 0:
                return True
        if h==k and (l%2)==0:
                return True
        if k==0 and l==0 and (h%2) == 0:
                return True
        if h==k and k==l and k%2==0: # and l%2==0:
                return True
            
    elif spacegroup == 62 or spacegroup=='pnma': #pnma  http://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-hkl?gnum=62
        
        #for Kaichen's case, k and l has to be exchanged. Pnma to Pnam
        #tt=k
        #k=l
        #l=tt
        
        if h==0:
            if k*l !=0:
                if (k+l)%2 ==0:
                    return True
            elif k%2==0 and l%2==0:
                return True
        if l==0:
            if h%2==0:
                return True
        if h!=0 and k!=0 and l!=0:
            if (h+l)%2==0 and k%2==0:
                return True
        
        
    else:
        print('Space group not yet supported, contact authors!')
        return True


#################################
# Calculate q for some lattice and a
#################################
def cal_q(lattice='FCC', a=1, max_index=5):
    q_data = pds.DataFrame()
    
    for h in range(max_index):
        for k in range(h,max_index):
            if lattice == 'HCP':
                for l in range(max_index):
                    if index_rule(lattice=lattice, h=h, k=k, l=l):
                        d = d_rule(lattice=lattice, a=a, h=h, k=k, l=l)
                        q = 2*np.pi/d
                        if h==0 and k==0 and l==2:
                            q0 = q
                        q_ratio = q/q0
                        #print(h, k, l, d, q)
                        hkl = '{}{}{}'.format(h, k, l)
                        data = [{'hkl': hkl,'d':d, 'q':q, 'q_ratio':q_ratio}]
                        df = pds.DataFrame(data)
                        q_data = q_data.append(df, ignore_index=True)
                    
            else:  #lattice is not HCP
                for l in range(k,max_index):
                    if index_rule(lattice=lattice, h=h, k=k, l=l):
                        d = d_rule(lattice=lattice, a=a, h=h, k=k, l=l)
                        q = 2*np.pi/d
                        if h==0 and k==0 and l==0:
                            q0 = q
                        q_ratio = q/q0
                        #print(h, k, l, d, q)
                        hkl = '{}{}{}'.format(h, k, l)
                        data = [{'hkl': hkl,'d':d, 'q':q, 'q_ratio':q_ratio}]
                        df = pds.DataFrame(data)
                        q_data = q_data.append(df, ignore_index=True)

    q_data.sort_values(by=['q'])
                    
    return q_data        
                        
                
def check_odd(x):
    if x%2 == 1:
        return True
def check_even(x):
    if x%2 == 0:
        return True

#################################
# Index rule for some lattice
#################################
def index_rule(lattice, h, k, l):
    if lattice == 'SC':
        return True
    if lattice == 'FCC':
        if check_odd(h) and check_odd(k) and check_odd(l):
            return True
        elif check_even(h) and check_even(k) and check_even(l):
            return True
        else:
            return False
    if lattice == 'Diamond':
        if check_odd(h) and check_odd(k) and check_odd(l):
            return True
        elif check_even(h) and check_even(k) and check_even(l) and (h+k+l)%4==0:
            return True
        else:
            return False
    if lattice == 'BCC':
        if (h+k+l)%2==0:
            return True
        else:
            return False
    if lattice == 'HCP':
        if check_even(l) and (2*h+l)%3!=0:
            return True
        else:
            return False

#################################
# Calculate d from a
#################################
def d_rule(lattice, a, h, k, l):
    if h==0 and k==0 and l==0:
        return a
    if lattice == 'FCC' or lattice == 'BCC'or lattice == 'SC' or lattice == 'Diamond':
        #b = a
        #c = a
        #gamma = np.deg2rad(90)
        return a/(h**2+k**2+l**2)**0.5
    if lattice == 'HCP':
        c = a*(8/3)**0.5
        #print(c)
        return 1/(4/3*(h**2+h*k+k**2)/a**2+(l/c)**2)**0.5
    else:
        #print('WARNING:::Lattice is NOT a correct input')
        return a
        

#################################
# load .npy file (from SciAnalysis)
#################################
def load_data(filename, n_svd=0, verbose=0):

    print(filename)
    if filename.find('.npz')>0:
        temp = np.load(filename)
        data = {  
            "img": temp['image'],
            "x_scale": temp['x_scale'], #unused
            "y_scale": temp['y_scale'], #unused
            "x_axis": temp['x_axis'],
            "y_axis": temp['y_axis'],   
        }
        img = data['img']
    elif filename.find('.tiff')>0:
        temp = fabio.open(filename)
        img = np.flip(temp.data,axis=0)
        fn_q = filename.replace('q_map', 'qpar')
        fn_q = fn_q.replace('.tiff','.txt')
        fn_q = fn_q.replace('/qr/','/txt/')
        temp = open(fn_q,'r').read().splitlines()
        x_axis = np.asarray(temp, dtype='float64') #*10.0
        fn_q = fn_q.replace('qpar', 'qver')
        temp = open(fn_q,'r').read().splitlines()
        y_axis = np.asarray(temp, dtype='float64') #*10.0
        data = {  
            "img": img,
            "x_axis": x_axis,
            "y_axis": y_axis,   
        }
    else:
        print('File not in .npy nor .tiff')

    ### Remove low freq # testing
    U,s,V = linalg.svd(img)

    if verbose: plt.figure(99, figsize=[7,10]); 
    count = 0
    for n in np.arange(1,n_svd+1,1):
        L=np.zeros((np.shape(img)))
        for i in range(n):
            L = L + np.outer(U[:,i],V[i,:])*(np.diag(s)[np.diag(s)!=0])[i]
        L = np.abs(L)
        if verbose:
            ax = plt.subplot2grid([3,2],[count,0]); #count += 1
            plt.imshow((L)); plt.colorbar()
            ax.set_aspect('equal', 'box')

            ax = plt.subplot2grid([3,2],[count,1]);  count += 1
            plt.imshow(np.log10(img-L)); plt.colorbar()
            ax.set_aspect('equal', 'box')
            plt.show()

    if n_svd>0:
        img_use = img-L
        img_use[img_use<0]=0
    else:
        #img_use = thr_img(img)
        img_use = np.copy(img)
        
    data['img_use'] = img_use
    data['plot_img_use'] = 1
        
    return data


#################################
# Calculate in-plane angle
#################################
def d_spacing(h,k,l, lattice, verbose=0):
    #should be in gradien
    [a,b,c,alp,beta,gam] = lattice
    V=a*b*c*sqrt(1+2*cos(alp)*cos(beta)*cos(gam)-cos(alp)**2-cos(beta)**2-cos(gam)**2)

    #reciprocal lattice
    ar=b*c*sin(alp)/V
    br=a*c*sin(beta)/V
    cr=a*b*sin(gam)/V

    alpr=acos((cos(gam)*cos(beta)-cos(alp))/abs(sin(gam)*sin(beta)))
    betar=acos((cos(alp)*cos(gam)-cos(beta))/abs(sin(alp)*sin(gam)))
    gamr=acos((cos(alp)*cos(beta)-cos(gam))/abs(sin(alp)*sin(beta)))
    if verbose>0:
        print(h, k, l)
    ss = (h**2*ar**2+k**2*br**2+l**2*cr**2+2*k*l*br*cr*cos(alpr)+2*l*h*cr*ar*cos(betar)+2*h*k*ar*br*cos(gamr))**0.5
    d=1/ss
    return d

################################
# Calc angle between planes
#
# lattice=(5.83, 7.88, 29.18, 90/180*pi, 99.4/180*pi, 90/180*pi)
# angle_interplane(0,0,1,1,1,0,lattice)
##########################
def angle_interplane(h1,k1,l1, h2, k2, l2, lattice, verbose=0):
    [a,b,c,alp,beta,gam] = lattice
    V=a*b*c*sqrt(1+2*cos(alp)*cos(beta)*cos(gam)-cos(alp)**2-cos(beta)**2-cos(gam)**2)

    #reciprocal lattice
    ar=b*c*sin(alp)/V
    br=a*c*sin(beta)/V
    cr=a*b*sin(gam)/V

    alpr=acos((cos(gam)*cos(beta)-cos(alp))/abs(sin(gam)*sin(beta)))
    betar=acos((cos(alp)*cos(gam)-cos(beta))/abs(sin(alp)*sin(gam)))
    gamr=acos((cos(alp)*cos(beta)-cos(gam))/abs(sin(alp)*sin(beta)))
    
    cosangle = d_spacing(h1,k1,l1,lattice,verbose=verbose)*d_spacing(h2,k2,l2,lattice,verbose=verbose)*(h1*h2*ar**2+k1*k2*br**2+l1*l2*cr**2+(k1*l2+k2*l1)*br*cr*cos(alpr)+(h1*l2+h2*l1)*ar*cr*cos(betar)+(h1*k2+h2*k1)*ar*br*cos(gamr))
    angle_deg = acos(cosangle)/np.pi*180
    if verbose>0:
        print('{:.3f} deg\n'.format(angle_deg))
    return angle_deg


def QtoTheta(Q=3, wavelength=12.4/13.576):
    Theta = np.arcsin(Q*wavelength/4/np.pi)*180/np.pi
    return Theta

def TwoThetatoQ(TwoTheta=33, wavelength=12.4/13.576):
    Q = 4*np.pi/wavelength*np.sin(np.deg2rad(TwoTheta)/2)
    return Q
    



############################## Following functions only for testing ####################################
    
#################################
# 
#################################
def get_meshgrid(dim, scaled):
    X, Y = np.meshgrid(np.arange(dim[0]+1)*scaled[0], np.arange(dim[1]+1)*scaled[1])
    return X, Y


#################################
# (Unused)
#################################
def thr_img(img):
    thr = np.mean(img)
    img_thr = copy.deepcopy(img)
    img_thr[img<=thr] = 0 
    return img_thr


#################################
# (Unused)
#################################
def calc_cost(data, Qxy_list, Qz_list, hkl_list, n_peaks, pixels, verbose): 
    
    if data:
        img = data['img']
        x_axis = data['x_axis']
        y_axis = data['y_axis']
        X, Y = np.meshgrid(x_axis, y_axis)
        if 'img_use' in data:
            img = data['img_use']
            if verbose>1:
                print('Using img_use (low freq removed)')
    
    I_array = []
    for ii, x in enumerate(Qxy_list):
        z = Qz_list[ii]
        I = get_int(img, x_axis, y_axis, x, z, pixels)
        if isnan(I)==False:
            I_array.append(I)
        if verbose>1:
            print('({}) {:.2f},{:.2f}, val={:.2f}'.format(hkl_list[ii], float(x),float(z),I))           
    
    if n_peaks=='all':
        cost = np.sum(I_array)
    else:
        cost = np.sum(np.sort(I_array)[-n_peaks::])  
    
    if verbose>1:                    
        fig = plt.figure(50); fig.clf()
        plt.plot(I_array)
        plt.ylabel('intensity')
    if verbose: print('With n_peaks = {}, match = {:.2f}'.format(n_peaks,cost))    
    
    return cost


#################################
# (Unused)
#################################
def get_int(img, x_axis, y_axis, x, y, pixels):
    m, n = img.shape
    idx_x = np.argmin(np.abs(x_axis-x))
    idx_y = np.argmin(np.abs(y_axis-y))
    I = img[idx_y,idx_x]
    I = np.mean(img[np.max([0,idx_y-pixels]) : np.min([n,idx_y+pixels+1]), \
                    np.max([0,idx_x-pixels]) : np.min([m,idx_x+pixels+1])])
    
    return I
            
        
