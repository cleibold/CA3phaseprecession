# CA3phaseprecession

1.  System requirements

-All software dependencies and operating systems (including version numbers)
The code was developed and run on a Linux Anaconda3 environment. 
Software dependencies (including version numbers) are summarized in the conda CA3pp.yml file

-Versions the software has been tested on
The software has only been tested on Linux using the libary versions in the CA3pp.yml file

-Any required non-standard hardware
None

2.  Installation guide

-Instructions

a) We recommend to use Anaconda3 to create a python3 environment CA3pp from the yml file by
>> conda env create -f CA3pp.yml

b) create a working directory a dirictory (output) for simulation results and a directory (figures) for plots
>> mkdir <ROOTDIR>

>> mkdir <ROOTDIR>/output

>> mkdir <ROOTDIR>/figures

c) copy the code into <ROOTDIR>
d) unzip the simluatin results file
>> tar xfvz empirical_results.tgz

-Typical install time on a "normal" desktop computer
less than 15 min

3.  Instructions for use & Demo
Instructions to run on data

Start gerating the Simulation results
>> python3 multip.py

Expected Duration: 25 Minutes [with reduced resolution as compared to the manuscript]

Expected outputs
1) spike times of the three model variants:

<ROOTDIR>/output/simulation_results-control-psi=0-dc_comp=0.00_.pkl

<ROOTDIR>/output/simulation_results-dglesion-psi=0-dc_comp=0.00_.pkl

<ROOTDIR>/output/simulation_results-meclesion-psi=0-dc_comp=0.00_.pkl

2) output of circular linear regression on the model spikes   

<ROOTDIR>/output/new-slope_rho_r2_ph0_ph_f-control-psi=0-dc_comp=0.00.mat

<ROOTDIR>/output/new-slope_rho_r2_ph0_ph_f-control-psi=0-dc_comp=0.00.npz

<ROOTDIR>/output/new-slope_rho_r2_ph0_ph_f-dglesion-psi=0-dc_comp=0.00.mat

<ROOTDIR>/output/new-slope_rho_r2_ph0_ph_f-dglesion-psi=0-dc_comp=0.00.npz

<ROOTDIR>/output/new-slope_rho_r2_ph0_ph_f-meclesion-psi=0-dc_comp=0.00.mat

<ROOTDIR>/output/new-slope_rho_r2_ph0_ph_f-meclesion-psi=0-dc_comp=0.00.npz

Start the plotting scripts
>> python3 mid_quartiles.py; python3 do_summary_plots.py

Expected Duration: 2 Minutes
Expected outputs

1) Plotting data

<ROOTDIR>/output/Masked_DGC-PCT=85-psi=0-dc_comp=0.00-phase_shift180.pkl

<ROOTDIR>/output/Masked_DGX-PCT=85-psi=0-dc_comp=0.00-phase_shift180.pkl

<ROOTDIR>/output/Masked_MECC-PCT=85-psi=0-dc_comp=0.00-phase_shift180.pkl

<ROOTDIR>/output/Masked_MECX-PCT=85-psi=0-dc_comp=0.00-phase_shift180.pkl


2) Figures (panels in figure 7)

<ROOTDIR>/figures/figmasks_DGC-PCT=85-psi=0-dc_comp=0.00-phase_shift180.eps

<ROOTDIR>/figures/figmasks_DGX-PCT=85-psi=0-dc_comp=0.00-phase_shift180.eps

<ROOTDIR>/figures/figmasks_MECC-PCT=85-psi=0-dc_comp=0.00-phase_shift180.eps

<ROOTDIR>/figures/figmasks_MECX-PCT=85-psi=0-dc_comp=0.00-phase_shift180.eps

<ROOTDIR>/figures/figsim_DGC-PCT=85-psi=0-dc_comp=0.00-phase_shift180.eps

<ROOTDIR>/figures/figsim_DGX-PCT=85-psi=0-dc_comp=0.00-phase_shift180.eps

<ROOTDIR>/figures/figsim_MECC-PCT=85-psi=0-dc_comp=0.00-phase_shift180.eps

<ROOTDIR>/figures/figsim_MECX-PCT=85-psi=0-dc_comp=0.00-phase_shift180.eps

<ROOTDIR>/figures/DG_masked-PCT=85-psi=0-dc_comp=0.00-phase_shift180.eps

<ROOTDIR>/figures/MEC_masked-PCT=85-psi=0-dc_comp=0.00-phase_shift180.eps




