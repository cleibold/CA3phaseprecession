#
import numpy as np
from numpy import pi
import torch
import itertools
import os
import scipy
from multiprocessing import Manager, Pool

from phase_precession_model import PhasePrecessionModel, \
                                   PhasePrecessionModelSimulator

from utils import write_simulation_results, gridded_simulation, rootpth, load_simulation_results, regress_in_chunks, save_regression_results, load_regression_results
#######


#gobal flags
runstr='output/'
sim_flag=True
reg_flag=True
phisteps=90# in the manuscript: 180
asteps=50
#
#
# model parameters
noise_scale=2
DC_values=np.array([0])
Psi_values=np.array([0])

# trains per trial
N_trains = 20
mean_firing_rate = 20


def perform_one_psi_and_dc_combo(in_args):
    global N_trains, mean_firing_rate
    
    (psi, dc_comp, experiment), inputs_dict = in_args

    A_VALUES, PHI_INH_VALUES = inputs_dict['A_VALUES'], inputs_dict['PHI_INH_VALUES']
    print(os.getpid(), f"is doing psi={psi}, dc_comp={dc_comp:.2f} for {experiment}")
    
    device = 'cpu'

    all_experiments = ['control', 'dglesion', 'meclesion']

    pp_model = {exp: PhasePrecessionModel(device=device, noise_scale=noise_scale) for exp in all_experiments}

    pp_model['dglesion']._DG.amplitude = 0
    pp_model['meclesion']._MEC.amplitude = 0

    simulator = {exp: PhasePrecessionModelSimulator(pp_model[exp], n_points=1000, device=device) for exp in all_experiments}

    try:
        X, Ph, Phi_0, Phi_f = gridded_simulation(pp_model[experiment], simulator[experiment], psi, dc_comp, N_trains, mean_firing_rate, A_VALUES, PHI_INH_VALUES)

        write_simulation_results(psi,
                                 dc_comp,
                                 append_name=experiment,
                                 N_trains=N_trains,
                                 mean_firing_rate=mean_firing_rate,
                                 X=X,
                                 path=rootpth+runstr,
                                 Ph=Ph,
                                 Phi_0=Phi_0,
                                 Phi_f=Phi_f,
                                 PHI_INH_VALUES=PHI_INH_VALUES,
                                 A_VALUES=A_VALUES)
    except:
        print(f"Unable to complete {dc_comp:.2f} for {experiment}")
    
    return



def regress_all(in_args):
    
    (psi, dc_comp, experiment) = in_args

    n_optimization_steps = 200
    optimizer_step_size = 3e-2
    chunk_size = 512

    path_simulation_results = rootpth+runstr
    regression_results_dir_npz = rootpth+runstr


    print(f"\n\n\nStarting psi = {psi}, dc_comp={dc_comp:.2f}, (experiment: {experiment})")
        
    psi, N_trains, mean_firing_rate, X, Ph, Phi_0, Phi_f, PHI_INH_VALUES, A_VALUES = load_simulation_results(psi, dc_comp, experiment, path=path_simulation_results)
        
    Slopes, rho, r2, Pvals = regress_in_chunks(X, Ph, chunk_size, n_optimization_steps, optimizer_step_size, A_VALUES, PHI_INH_VALUES)
        
    file_name='new-slope_rho_r2_ph0_ph_f-'+experiment+'-psi='+str(psi)+f"-dc_comp={dc_comp:.2f}"

    jpth=os.path.join(regression_results_dir_npz, file_name)
    
    save_regression_results(jpth, Slopes, rho, r2, Phi_0, Phi_f, n_optimization_steps)

    
    regression_results_dir_mat = rootpth+runstr
    load_name = os.path.join(regression_results_dir_npz, file_name + '.npz')
    save_name = os.path.join(regression_results_dir_mat, file_name + '.mat')
        
    Slopes, rho, r2, Phi_0, Phi_f, n_optimization_steps = load_regression_results(load_name)
        
    mdict = dict(Slopes=Slopes, 
                 rho=rho, 
                 r2=r2, 
                 Phi_0=Phi_0, 
                 Phi_f=Phi_f, 
                 A_VALUES=A_VALUES,
                 PHI_INH_VALUES=PHI_INH_VALUES,
                 n_optimization_steps=n_optimization_steps)

    scipy.io.savemat(save_name, mdict)


    


if __name__ == "__main__":
    
    manager = Manager()

    
    PHI_INH_VALUES = torch.linspace(0, 2*pi, 1+phisteps, device='cpu')[:-1]
    A_VALUES = torch.linspace(0, 5, 1+asteps, device='cpu')

    shared_inputs = manager.dict(A_VALUES=A_VALUES, PHI_INH_VALUES=PHI_INH_VALUES)

    Experiments = ['control', 'dglesion', 'meclesion']

    
    n_workers = len(Psi_values) * len(DC_values) * len(Experiments)

    if sim_flag==True:
        with Pool(processes=min(4, n_workers)) as pool:
            pool.map(perform_one_psi_and_dc_combo, zip(itertools.product(Psi_values, DC_values, Experiments), [shared_inputs]*n_workers))

    if reg_flag==True:
        with Pool(processes=min(4, n_workers)) as pool:
            pool.map(regress_all, itertools.product(Psi_values, DC_values, Experiments))

    



