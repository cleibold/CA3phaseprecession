import numpy as np
from numpy import pi

import mat73
import pickle
import scipy
import os
import itertools
from multiprocessing import Manager, Pool

import matplotlib.pyplot as plt

from matplotlib.colors import to_rgb
from matplotlib.colors import LinearSegmentedColormap as owncmap
from matplotlib.colors import hsv_to_rgb as hsv2rgb

from utils import rootpth, load_regression_results,  load_simulation_results, plot_parameters_with_mask
##
runstr='output/'
figstr='figures/'

#
#
# analysis parameters
#
DC_values=np.array([0])
Psi_values=np.array([0])
ph_shift=np.array([np.pi])
pct_around_median = 85;
###


#
#load empirical data
#
path_simulation_results = rootpth+runstr
emp_pth=rootpth+'empirical_results.mat'

emp_all=mat73.loadmat(emp_pth)
empirical_results=emp_all['empirical_results']

slopes = empirical_results['slopes']
phi_0_pass = empirical_results['phi_0_pass']
phi_f_pass = empirical_results['phi_f_pass']
r2_pass = empirical_results['r2_pass']

data={0:slopes,1:r2_pass,2:phi_0_pass,3:phi_f_pass}


res_pth=rootpth+'labels.mat'
results=mat73.loadmat(res_pth)

labels=results['labels']['str']
#
# set path
#
regression_results_dir_npz = rootpth+runstr
regression_results_dir_mat = rootpth+runstr
#

####
#
# definitions
#
blue_white=np.array([list(hsv2rgb([.6,1-v,1])) for v in [0,.25,.5,.75,1]])
blue_white_binary = np.array([list(hsv2rgb([.6,1,1])),list(np.ones(3))]);
cmap_blue_white =owncmap.from_list('bluewhite1', blue_white, N=5)
cmap_blue_white_binary =owncmap.from_list('bluewhite2', blue_white_binary, N=2)

IDCS=['MECX','MECC','DGX','DGC']
TITLE_ORDER = ['$\phi_0$', '$\phi_f$', 'Slope', '$R^2$']
Simulations = ['control', 'dglesion', 'meclesion']

def lmask(matin,data,pct):
    lower=np.nanquantile(data,pct/2)
    upper=np.nanquantile(data,1-pct/2)
    print(lower,upper)
    mat_shift=np.zeros_like(matin)
    nhalf=int(matin.shape[1]/2)
    nr=matin.shape[1]-nhalf
    mat_shift[:,0:nhalf]=matin[:,nr:]
    mat_shift[:,nhalf:]=matin[:,0:nr]
    mat1=scipy.signal.convolve2d(matin,np.ones((5,5))/25,'same')
    mat2=scipy.signal.convolve2d(mat_shift,np.ones((5,5))/25,'same')

    
    matr = mat1
    matr[:,0:10] = mat2[:,nr:nr+10]
    matr[:,-10:] = mat2[:,nr-10:nr]
    return (matr<upper)*(matr>lower)

def mask(matr,d_emp,ind, pct):
    sl=[]
    for ni in ind:
        arr=d_emp[ni][0]
        arr=arr.reshape(-1)
        sl = sl+list(arr)
  
    return lmask(matr,sl,pct),sl



###
#
# plot functions
#
#
#
#
def do_all_plots(in_args):
    
    (psi, dc_comp, experiment, phase_shift) = in_args

    saveFigsTo = rootpth+figstr;
    savePklTo = rootpth+runstr

    file_name = 'new-slope_rho_r2_ph0_ph_f-'+experiment+'-psi='+str(psi)+'-dc_comp='+f"{dc_comp:.2f}"
    load_name = os.path.join(regression_results_dir_npz, file_name + '.npz')

    in_sim = load_simulation_results(psi, dc_comp,experiment, path=path_simulation_results)         
      
    PHI_INH_VALUES = in_sim[-2]
    A_VALUES = in_sim[-1]
    MIN_INH_AMPL = A_VALUES[0];
    MAX_INH_AMPL = A_VALUES[-1];
    N_BINS_INH_AMPL = len(A_VALUES);
    N_BINS_INH_PHASEOFFSET = len(PHI_INH_VALUES);
      
    pha_array = np.arange(0, 720, 2*N_BINS_INH_PHASEOFFSET);
    inh_str_array = np.arange(MIN_INH_AMPL, MAX_INH_AMPL, N_BINS_INH_AMPL);
    #
    # load simulation result
    #
    file_name = 'new-slope_rho_r2_ph0_ph_f-'+experiment+'-psi='+str(psi)+'-dc_comp='+f"{dc_comp:.2f}"
    load_name = os.path.join(regression_results_dir_mat, file_name + '.mat')

    model_param_space=scipy.io.loadmat(load_name)  
 
    slopes_model = model_param_space['Slopes'] * 2*pi
    onset_phases = np.mod(model_param_space['Phi_0']+phase_shift,2*np.pi)
    offset_phases = np.mod(model_param_space['Phi_f']+phase_shift,2*np.pi)
    var_explained = model_param_space['r2']

    model={0:slopes_model,1:var_explained,2:onset_phases,3:offset_phases}
    #
    #
    if experiment == 'control':
        LIDCS=[IDCS[1], IDCS[3]]
    elif experiment == 'dglesion':
        LIDCS=[IDCS[2]]
    elif experiment == 'meclesion':
        LIDCS=[IDCS[0]]

    masked={}
    for REGION_IDX in LIDCS:
        
        FIG_TITLE = REGION_IDX
        ind=np.where([l==REGION_IDX for l in labels])[0]

        fig=plt.figure(figsize=(8,9))
        axs=[]
        mult_mask=np.ones_like(model[0])
        summ_mask=np.zeros_like(model[0])

        displays=[]
        summarydata={}
        for feature_id in range(4):
            ax=fig.add_subplot(3,2,feature_id+1)
            _mask,summarydata[feature_id]=mask(model[feature_id],data[feature_id],ind,1-pct_around_median/100)
            summ_mask += _mask
            mult_mask *= _mask
            extent=[0,2*pi , MAX_INH_AMPL, MIN_INH_AMPL]
            displays.append(ax.imshow(_mask, extent=extent, cmap=cmap_blue_white_binary, vmin=0, vmax=1, interpolation='none', aspect='auto'))
            #ax.contour(np.array(PHI_INH_VALUES),np.array(A_VALUES), _mask, levels=[0.5], colors='b')
            ax.invert_yaxis()
            ax.set_xlabel('$\Phi$')
            ax.set_ylabel('A')
            axs.append(ax)

        ax=fig.add_subplot(3,2,5)
        displays.append(ax.imshow(mult_mask, extent=extent, cmap=cmap_blue_white_binary, vmin=0,vmax=1,interpolation='none', aspect='auto'))
        ax.contour(np.array(PHI_INH_VALUES),np.array(A_VALUES), mult_mask, levels=[0.5], colors='b')
        ax.invert_yaxis()
        ax.set_xlabel('$\Phi$')
        ax.set_ylabel('A')
        axs.append(ax)

        ax=fig.add_subplot(3,2,6)
        displays.append(ax.imshow(summ_mask, extent=extent, cmap=cmap_blue_white, interpolation='none', aspect='auto'))
        ax.contour(np.array(PHI_INH_VALUES),np.array(A_VALUES), summ_mask, levels=[3.5], colors='b')
        ax.invert_yaxis()
        ax.set_xlabel('$\Phi$')
        ax.set_ylabel('A')
        axs.append(ax)

        [fig.colorbar(disp, ax=ax) for disp, ax in zip(displays, axs)];

        fig.suptitle(FIG_TITLE, fontsize=12)
        fig.subplots_adjust(wspace=0.4,hspace=0.3)
        
        file_name = 'figmasks_'+REGION_IDX+'-PCT='+str(pct_around_median)+'-psi='+str(psi)+'-dc_comp='+f"{dc_comp:.2f}"+'-phase_shift'+str(int(phase_shift/np.pi*180))
        save_name = os.path.join(saveFigsTo, file_name + '.eps')
        fig.savefig(save_name)
        
        
        figsim=plt.figure()
        plot_parameters_with_mask(model[0]/2/pi, model[1], model[2], model[3], A_VALUES, PHI_INH_VALUES, experiment, mult_mask, figsim);
        figsim.suptitle(FIG_TITLE, fontsize=12)

                
        file_name = 'figsim_'+REGION_IDX+'-PCT='+str(pct_around_median)+'-psi='+str(psi)+'-dc_comp='+f"{dc_comp:.2f}"+'-phase_shift'+str(int(phase_shift/np.pi*180))
        save_name = os.path.join(saveFigsTo, file_name + '.eps')
        figsim.savefig(save_name)

        nan_mask=mult_mask
        nan_mask[mult_mask==0]=np.nan
        
        masked[REGION_IDX]={'slope': nan_mask*model[0],
                            'r2': nan_mask*model[1],
                            'phi0': nan_mask*model[2],
                            'phif': nan_mask*model[3],
                            'p(A)': (np.array(A_VALUES),np.nansum(mult_mask,axis=1)),
                            'p(phi)': (np.array(PHI_INH_VALUES), np.nansum(mult_mask,axis=0)),
                            'data': summarydata,
                            'mult_mask':nan_mask,
                            'p0raw': model[2],
                            'pfraw': model[3]}
                            
                            
        file_name = 'Masked_'+REGION_IDX+'-PCT='+str(pct_around_median)+'-psi='+str(psi)+'-dc_comp='+f"{dc_comp:.2f}"+'-phase_shift'+str(int(ph_shift/np.pi*180))
        save_name = os.path.join(savePklTo, file_name + '.pkl')

        with open(save_name,'wb') as fd:
            pickle.dump(masked,fd)

####
#
# loop
#



if __name__ == "__main__":
    
    manager = Manager()


    n_workers = len(Psi_values) * len(ph_shift) * len(DC_values)* len(Simulations)


    with Pool(processes=min(4, n_workers)) as pool:
            pool.map(do_all_plots, itertools.product(Psi_values, DC_values, Simulations, ph_shift) )

