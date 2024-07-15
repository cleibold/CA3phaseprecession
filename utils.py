import time
import torch
import scipy
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from phase_precession_model import CircularLinearRegressor

#rootpth='/home/leibold/Desktop/research/phase_precession_Sia/output'
rootpth='/home/leibold/homeBWSFS/ongoing_projects/phase_precession_Sia/output'

def gridded_simulation(pp_model, simulator, psi, dc_comp, N_trains, mean_firing_rate, A_VALUES, PHI_INH_VALUES):
    """Simulate spikes from a phase precession model with given parameters on a grid

    Return spike time/location (X) and theta phase of those spikes (Ph) together with an estimate of
    the onset (Phi_0) and offset (Phi_f) phase of each set of simulated passes.
    
    The excitatory phase difference (psi) and the baseline (constant) inhibition (dc_comp) are given as single floats.
    The values of inhibitory amplitude (A) and excitatory-inhibitory phase difference (phi_inh) are given as arrays.
    The spikes are simulated on a grid of the A-phi values so there will N_A * N_phi simulations where N_A is the number
    of values specified for A and N_phi is the number of values specified for phi_inh.

    The returned variables will be N_A-by-N_phi in the case of Phi_0 and Phi_f
    and lists of length N_A * N_phi in the case of X and Ph"""

    pp_model.psi_(torch.deg2rad(torch.tensor(psi)))
    pp_model.dc_comp = dc_comp

    X, Ph, Phi_0, Phi_f = [], [], [], []

    tic = time.time()
    for A in A_VALUES:
        for phi_inh in PHI_INH_VALUES:
            pp_model.A_(A)
            pp_model.phi_(phi_inh)
            spike_times, spike_phases, valid = simulator.simulate(N_trains, mean_firing_rate)

            x, ph = simulator.cell_pooled_spike_x_ph(spike_times, spike_phases, valid)
            phi0, phif = simulator.cell_pooled_onset_offset_phase(spike_times, spike_phases, valid)
            X.append(x)
            Ph.append(ph)
            Phi_0.append(phi0)
            Phi_f.append(phif)
        if np.round(100*A) % 25 == 0:
            print(f"(psi = {psi}) Finished up to A = {A:.3f} in {time.time() - tic:.3f} s.")

    
    print(f"Elasped {time.time() - tic:.3f} s.")
    if not Phi_0: # if no spikes were generated
        Phi_0 = torch.full((A_VALUES.shape[0], PHI_INH_VALUES.shape[0]), torch.nan)
        Phi_f = torch.full((A_VALUES.shape[0], PHI_INH_VALUES.shape[0]), torch.nan)
    else:
        Phi_0 = torch.stack(Phi_0).reshape(A_VALUES.shape[0], PHI_INH_VALUES.shape[0])
        Phi_f = torch.stack(Phi_f).reshape(A_VALUES.shape[0], PHI_INH_VALUES.shape[0])
    
    return X, Ph, Phi_0, Phi_f


def regress_in_chunks(X, Ph, chunk_size, n_optimization_steps, optimizer_step_size, A_VALUES, PHI_INH_VALUES):
    """Perform circular linear regression over the simulated spikes contained in X and Ph in parallel (on GPU)
    but with memory considerations
    
    X and Ph can contain many sets of simulated spikes that together might not fit on the GPU memory.
    In that case, `chunk_size` determines how many sets of values to simultaneously process on GPU for better performance."""

    Slopes = []
    Rho = []
    Pvals = []
    tic = time.time()
    for chunk in range(0,len(X)-1, chunk_size):
        tic_chunk = time.time()
        print(f"\nProcessing chunk {chunk:,} to {min(chunk+chunk_size, len(X)):,} of {len(X):,} total.")

        clr = CircularLinearRegressor(X[chunk:chunk+chunk_size], Ph[chunk:chunk+chunk_size], [-2, 2], n_optimization_steps=n_optimization_steps, optimizer_step_size=optimizer_step_size)
        sizeof_float = 4 if clr.x.dtype==torch.float32 else 8
        slopes, _, rho, _, pvals, _ = clr.regresscl()
        del clr.x
        del clr.phi
        torch.cuda.empty_cache()

        Slopes.append(slopes)
        Rho.append(rho)
        Pvals.append(pvals)
        print(f"Chunk time: {time.time() - tic_chunk:.3f} s", end=' ')
        print(f"(Total time: {time.time() - tic:.3f} s.)")

    Slopes = np.concatenate(Slopes).reshape(A_VALUES.shape[0], PHI_INH_VALUES.shape[0])
    Rho = np.concatenate(Rho).reshape(A_VALUES.shape[0], PHI_INH_VALUES.shape[0])
    Pvals = np.concatenate(Pvals)
    r2 = Rho**2
    
    return Slopes, Rho, r2, Pvals

def plot_parameters(Slopes, r2, Phi_0, Phi_f, A_VALUES, PHI_INH_VALUES, experiment):
    """Plot the matrices of the regression results as displayed in Figure 7d,f"""
    
    extent = [PHI_INH_VALUES[0], PHI_INH_VALUES[-1], A_VALUES[0], A_VALUES[-1]]

    fig, ax = plt.subplots(2,2,figsize=(8,6))
    displays = []

    displays.append(ax[0,0].imshow(np.flipud(Slopes), extent=extent, interpolation='none', aspect='auto', cmap='coolwarm', vmin=-1, vmax=1))
    displays.append(ax[0,1].imshow(np.flipud(r2), extent=extent, interpolation='none', aspect='auto'))
    displays.append(ax[1,0].imshow(np.flipud(Phi_0), extent=extent, interpolation='none', aspect='auto', cmap='hsv', vmin=0, vmax=2*np.pi))
    displays.append(ax[1,1].imshow(np.flipud(Phi_f), extent=extent, interpolation='none', aspect='auto', cmap='hsv', vmin=0, vmax=2*np.pi))

    [fig.colorbar(disp, ax=ax) for disp, ax in zip(displays, ax.flatten())];
    
    ax[0, 0].set_title("Slopes")
    ax[0, 1].set_title("R$^2$")
    ax[1, 0].set_title("Phi$_{on}$")
    ax[1, 1].set_title("Phi$_{off}$")
    
    fig.suptitle(experiment)
    
    return fig, ax


def plot_parameters_with_mask(Slopes, r2, Phi_0, Phi_f, A_VALUES, PHI_INH_VALUES, experiment, mask ,figsim):
    """Plot the matrices of the regression results as displayed in Figure 7d,f"""
    
    extent = [PHI_INH_VALUES[0], PHI_INH_VALUES[-1], A_VALUES[0], A_VALUES[-1]]

    ax = figsim.subplots(2,2)
    displays = []

    displays.append(ax[0,0].imshow(np.flipud(Slopes), extent=extent, interpolation='none', aspect='auto', cmap='coolwarm', vmin=-1, vmax=1))
    displays.append(ax[0,1].imshow(np.flipud(r2), extent=extent, interpolation='none', aspect='auto'))
    displays.append(ax[1,0].imshow(np.flipud(Phi_0), extent=extent, interpolation='none', aspect='auto', cmap='hsv', vmin=0, vmax=2*np.pi))
    displays.append(ax[1,1].imshow(np.flipud(Phi_f), extent=extent, interpolation='none', aspect='auto', cmap='hsv', vmin=0, vmax=2*np.pi))

    [figsim.colorbar(disp, ax=ax) for disp, ax in zip(displays, ax.flatten())];
    
    for _ax in [ax[0,0],ax[0,1],ax[1,0],ax[1,1]]:
        _ax.contour(np.array(PHI_INH_VALUES),np.array(A_VALUES), mask, levels=[0.5], colors='k')
    
    ax[0, 0].set_title("Slopes")
    ax[0, 1].set_title("R$^2$")
    ax[1, 0].set_title("Phi$_{on}$")
    ax[1, 1].set_title("Phi$_{off}$")
    
    figsim.suptitle(experiment)
    
    return ax



def heaviside(x):
    """Heaviside function"""

    x[x<0] = 0
    return x

def hex2rgb(hexcode,normalize=True):
    """Convert a hex color code to a 1-d numpy array
    with three entries representing red, green, and blue"""

    r = (int(hexcode,16)>>16 & (1<<8)-1)
    g = (int(hexcode,16)>>8  & (1<<8)-1)
    b = (int(hexcode,16)     & (1<<8)-1)

    return np.array([r,g,b]) / 255 if normalize else np.array([r,g,b])


def place_regression_line(slope, phi_on, m, M, ax=None):
    """Easy plotting of linear regression on existing axes"""

    if ax is None:
        ax = plt.gca()
    x = [m, M]
    X = np.vstack([x, np.ones(len(x))])
    A = np.c_[slope*2*pi, phi_on]
    Y = A @ X
    
    while np.any(np.mean(Y, axis=1) < np.pi):
        Y += 2*pi
    ax.plot(x, Y.T, 'k')
    ax.set_ylim([0, 4*pi])
    ax.set_yticks([0, 2*pi, 4*pi])
    ax.set_yticklabels([0, 1, 2])   
    ax.set_ylabel("Theta Phase (cycles)")
    ax.set_xlabel("Normalized Distance")
    return Y

def sigstars(pval):
    """Significance stars"""
    
    if pval > 0.05:
        return 'n.s.'
    if pval == 0:
        return '****'
    return '*' * min(4, -int(np.ceil(np.log10(pval))))


def save_regression_results(filename, Slopes, rho, r2, Phi_0, Phi_f, n_optimization_steps):
    np.savez(filename, Slopes=Slopes, rho=rho, r2=r2, Phi_0=Phi_0, Phi_f=Phi_f, n_optimization_steps=n_optimization_steps)

def load_regression_results(filename):
    if not filename.endswith('.npz'):
        filename += '.npz'
    npz = np.load(filename)

    Slopes = npz['Slopes']
    rho = npz['rho']
    r2 = npz['r2']
    Phi_0 = npz['Phi_0']
    Phi_f = npz['Phi_f']
    n_optimization_steps = npz['n_optimization_steps']
    
    return Slopes, rho, r2, Phi_0, Phi_f, n_optimization_steps


def write_simulation_results(psi, dc_comp=0, append_name='', path=rootpth, **simulation_results):
    if append_name:
        append_name = '-' + append_name
    simulation_results['psi'] = psi

    fname='simulation_results'+append_name+'-psi='+str(psi)+'-dc_comp='+f"{dc_comp:.2f}"+'_.pkl'
    print(fname)
    
    with open(os.path.join(path, fname), 'wb') as f:
        pickle.dump(simulation_results, f)
    return



def load_simulation_results(psi, dc_comp=0, append_name='', path=rootpth):
    def extract(psi, N_trains, mean_firing_rate, X, Ph, Phi_0, Phi_f, PHI_INH_VALUES, A_VALUES):
        return psi, N_trains, mean_firing_rate, X, Ph, Phi_0, Phi_f, PHI_INH_VALUES, A_VALUES
    tic = time.time()
    if append_name:
        append_name = '-' + append_name
        fname='simulation_results'+append_name+'-psi='+str(psi)+'-dc_comp='+f"{dc_comp:.2f}"+'_.pkl' 
        print(fname)
    with open(os.path.join(path, fname), 'rb') as f:
        results = extract(**pickle.load(f))
        print(f"Read in {time.time() - tic:.3f} s")
        return results

def write_simulation_results_matlab(psi, dc_comp=0, append_name='', path=rootpth, **simulation_results):
    if append_name:
        append_name = '-' + append_name

    simulation_results['psi'] = psi

    print(simulation_results['X'])

    #dict_keys(['N_trains', 'mean_firing_rate', 'X', 'Ph', 'Phi_0', 'Phi_f', 'PHI_INH_VALUES', 'A_VALUES', 'psi'])
    fname='simulation_results'+append_name+'-psi='+str(psi)+'.mat'
    jpth=os.path.join(path,fname)
    print(jpth)
    
    scipy.io.savemat(jpth, simulation_results)

def load_simulation_results_matlab(psi, dc_comp=0, append_name='', path=rootpth):
    def extract(psi, N_trains, mean_firing_rate, X, Ph, Phi_0, Phi_f, PHI_INH_VALUES, A_VALUES):
        return psi, N_trains, mean_firing_rate, X, Ph, Phi_0, Phi_f, PHI_INH_VALUES, A_VALUES
    if append_name:
        append_name = '-' + append_name
    tic = time.time()
    #fname='simulation_results'+append_name+'-psi='+str(psi)+'-dc_comp='+str(dc_comp)+'.mat'
    fname='simulation_results'+append_name+'-psi='+str(psi)+'.mat'
    print(fname)
    mdict = scipy.io.loadmat(os.path.join(path, fname))
    variables = {}
    for k in mdict:
        if not k.startswith('_'):
            if k == "X" or k == "Ph":
                variables[k] = [torch.tensor(x.squeeze()) for x in mdict[k][0]]
            else:
                variables[k] = torch.tensor(mdict[k].squeeze())
    results = extract(**variables)
    print("Read in {time.time() - tic:.3f} s")
    return results
