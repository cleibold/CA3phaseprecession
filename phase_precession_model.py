import numpy as np
import torch
import scipy
import time

class PhasePrecessionModel():
    """Define the paper's phase precession model"""

    class RhythmicInput():
        """Generic oscillatory input class with a given
        frequency, amplitude, offset, and envelope shape
        
        can be instantiated to model DG, MEC or the inhibitory inputs
        to the model
        
        The envelope is a gaussian PDF and its shape is determined by
        two parameters `peak_loc` and `spatial_broadness`
        
        peak_loc is the mean of the gaussian PDF
        spatial_broadness is the standard deviation
        
        the gaussian is normalized such that the peak of
        the PDF is always at 1. This is done by dividing
        the gaussian by the normalization factor behind
        the Gaussian PDF formula:
            1 / (sigma * sqrt(2*pi))
        
        This division, however, is carried out by subtracting 
        the log value of the normalization factor from the log
        of the probability for numerical stability.
        """

        def __init__(self, amplitude=None, frequency=None, offset=None, peak_loc=None, spatial_broadness=None, device=None):
            self.amplitude = amplitude if amplitude is None else torch.tensor(amplitude, device=device)
            self.frequency = frequency if frequency is None else torch.tensor(frequency, device=device)
            self.offset = offset if offset is None else torch.tensor(offset, device=device)
            self._peak_loc = peak_loc if peak_loc is None else torch.tensor(peak_loc, device=device)
            self._spatial_broadness = spatial_broadness if spatial_broadness is None else torch.tensor(spatial_broadness, device=device)
            self.gaussian = None
            self.normalization_factor = None
            self.device = device
            self._two_pi =  torch.tensor(2.*np.pi, device=self.device)      
            
            self.to(device)

            if peak_loc is not None and spatial_broadness is not None:
                self.gaussian = torch.distributions.normal.Normal(peak_loc, spatial_broadness)

        def spatial_modul(self, x):
            """A gaussian, if the relevant parameters have been furnished---
            the identity function otherwise"""

            return (self.gaussian.log_prob(x) + self._normalization_factor()).exp() if self.gaussian is not None else x

        def total_spiking_drive(self, x):
            """Up shifted cosine as model output"""
            
            return self.amplitude * 0.5 * (1. + torch.cos(2.*np.pi * self.frequency * x - self.offset)) # up-shifted cosine

        def peak_loc_(self, loc=None):
            """Set spatial modulation location, return the current value if none supplied"""
            
            if loc is None:
                return self._peak_loc
            self._peak_loc = torch.tensor(loc, device=self.device) if not isinstance(loc, torch.Tensor) else loc.to(self.device)
            if self._spatial_broadness is None:
                self._spatial_broadness = torch.tensor(1., device=self.device)
            if self.gaussian is None:
                self.gaussian = torch.distributions.normal.Normal(self._peak_loc, self._spatial_broadness)
            else:
                self.gaussian.loc = self._peak_loc

        def spatial_broadness_(self, spatial_broadness=None):
            """Set spatial modulation broadness, return the current value if none supplied"""

            if spatial_broadness is None:
                return self._spatial_broadness
            self._spatial_broadness = torch.tensor(spatial_broadness, device=self.device) if not isinstance(spatial_broadness, torch.Tensor) else spatial_broadness.to(self.device)
            if self._peak_loc is None:
                self._peak_loc = torch.tensor(0.5, device=self.device)
            if self.gaussian is None:
                self.gaussian = torch.distributions.normal.Normal(self._peak_loc, self._spatial_broadness)
            else:
                self.gaussian.scale = self._spatial_broadness

        def _normalization_factor(self):
            if self._spatial_broadness is not None:
                return torch.log(self._spatial_broadness) + 0.5 * torch.log(self._two_pi)
        
        def to(self, device):
            """Move model attributes to device, leave untouched if device is None"""

            if device is None:
                return

            self.amplitude = self.amplitude \
                            if self.amplitude is None \
                            else self.amplitude.to(device)
            
            self.frequency = self.frequency \
                            if self.frequency is None \
                            else self.frequency.to(device)
            
            self.offset = self.offset \
                            if self.offset is None \
                            else self.offset.to(device)
            
            self._peak_loc = self._peak_loc \
                            if self._peak_loc is None \
                            else self._peak_loc.to(device)
            
            self._spatial_broadness = self._spatial_broadness \
                            if self._spatial_broadness is None \
                            else self._spatial_broadness.to(device)


    class DGInput(RhythmicInput):
        """DG input inheriting from generic RhythmicInput template"""
        def __init__(self,
                    peak_loc=0.3,
                    gaussian_std=0.45,
                    device=None) -> None:

            frequency = 8.6 # Mizuseki et al. (2009) Suppl Fig 15;
                            # DG Precession Range is ~210 degrees;
                            # Also Geisler et al. (2010) determine this value to be 8.61 Hz from data
            
            psi = np.deg2rad(-40) # Mizuseki et al. (2009) Fig. 3A

            super().__init__(1., frequency, psi, peak_loc, gaussian_std, device=device)

    class MECInput(RhythmicInput):
        """MEC input inheriting from generic RhythmicInput template"""
        def __init__(self,
                    peak_loc=0.7,
                    gaussian_std=0.75,
                    device=None) -> None:

            frequency = 8.5 # Mizuseki et al. (2009) Suppl Fig 15;
                            # MEC Precession Range is ~180 degrees
            
            psi = np.deg2rad(160) # From Mizuseki (2009) Fig 3A

            super().__init__(1., frequency, psi, peak_loc, gaussian_std, device=device)

    class InhibitoryInput(RhythmicInput):
        """Inhibitory input inheriting from generic RhythmicInput template"""
        def __init__(self,
                        A=1,
                        phi_inh=np.deg2rad(290),
                        device=None) -> None:

            super().__init__(A, 8, phi_inh, device=device)


    def __init__(self, A=None, phi_inh=None, psi=np.deg2rad(-200), dc_comp=0, noise_scale=0, device=None):
        self._DG  = PhasePrecessionModel.DGInput(device=device)
        self._MEC = PhasePrecessionModel.MECInput(device=device)
        self._INH = PhasePrecessionModel.InhibitoryInput(A, phi_inh, device=device)
        self.dc_comp = dc_comp

        self.psi_(psi)
        
        self._membrane_noise = None
        if noise_scale>0:
            self._membrane_noise = torch.distributions.normal.Normal(0, noise_scale)          

        self._lfp_theta_frequency = torch.tensor(8, device=device)
        self._lfp_phase_shift = torch.tensor(0, device=device)
        self.device = device

    def LFP_theta(self, x):
        """Simulate extracellular theta oscillation"""

        return torch.cos(2*np.pi*self._lfp_theta_frequency * x - self._lfp_phase_shift)

    def A_(self, A):
        """Set model's inhibitory input amplitude"""

        self._INH.amplitude = A.to(self.device)

    def phi_(self, phi_inh):
        """Set model's inhibitory offset phase"""

        self._INH.offset = phi_inh.to(self.device)
    
    def psi_(self, psi=None):
        """Set model's DG-MEC excitatory offset phase (psi)
        
        return the current value of the offset (psi) if none provided"""

        if psi is not None:
            self._DG.offset = self._MEC.offset - psi
        else:
            return self._MEC.offset - self._DG.offset

    def TotalDrive(self, x, return_components=False):
        """Model output simultating total spiking probabilities"""

        DG  = self._DG.spatial_modul(x)  * self._DG.total_spiking_drive(x)
        MEC = self._MEC.spatial_modul(x) * self._MEC.total_spiking_drive(x)
        INH = self._INH.total_spiking_drive(x)

        if self._membrane_noise is not None:
            membrane_noise = self._membrane_noise.sample(x.shape).to(self.device)
        else:
            membrane_noise = torch.zeros(x.shape, device=self.device)

        return heaviside(DG + MEC + membrane_noise - INH - self.dc_comp) ** 2 if not return_components \
             else (heaviside(DG + MEC + membrane_noise - INH - self.dc_comp) ** 2, DG, MEC, INH)

class PhasePrecessionModelSimulator():
    """Simulate spikes based on a barebones PhasePrecession model instance"""
    
    def __init__(self, pp_model: PhasePrecessionModel, device='cpu', n_points=100, seed=0):

        self.model = pp_model
        self.device = device
        self.T = 1 # max time for simulation

        self.x = torch.linspace(0, 1, n_points, device=device)

        self.theta = self.model.LFP_theta(self.x)
        self.theta_phase = self._phase_of_oscillation(self.theta)

        self.seed = seed # for reproducibility


    def _phase_of_oscillation(self, theta):
        """Phase of oscillation using the Hilbert transform"""

        hilbert = torch.tensor(scipy.signal.hilbert(theta.cpu()), device=self.device)
        return torch.angle(hilbert) % (2*np.pi)

    def simulate(self, N_trains, mean_firing_rate, seed=None):
        """Simulate phase precession based on total excitatory drive

        Args:
            N_trains (int):
                How many passes to simulate
            mean_firing_rate (float):
                Mean firing rate across passes
        Returns:
            spike_times (float):
                M-by-N numpy array of floats where N is the number of
                trains and M is the number of spikes in the train with
                the largest number of spikes. Each value is considered
                the time/location of a spike within the simulated place
                field (values between 0 and 1).
            spike_phases (float):
                Same shape as `spike_times` but contains the theta phase
                of the corresponding spike.
            valid (bool):
                Same shape as `spike_times` indicating whether the
                corresponding values in `spike_times` and `spike_phases`
                are actual spikes. This is important to keep the outputs
                nicely shaped for compute efficiency (so that computations
                can be carried out on GPU in a vectorized form).

                IMPORTANT: ONLY THE VALID ENTRIES OF THE TIME AND PHASE
                VARIABLES SHOULD BE USED. THE INVALID VALUES ARE GARBAGE.
        """


        spike_times, valid = inhomopp(self.model.TotalDrive,
                                      T=self.T,
                                      expected_n_particles=mean_firing_rate,
                                      n_samples=N_trains,
                                      device=self.device,
                                      seed=seed if seed is not None else self.seed)
        spike_phases = self._phase_of_spikes(spike_times)

        return spike_times, spike_phases, valid

    def  _phase_of_spikes(self, spike_times):
        return torch.tensor(np.interp(spike_times.cpu(), self.x.cpu(), self.theta_phase.cpu()), device=self.device)

    def cell_pooled_spike_x_ph(self, spike_times, spike_phases, valid):
        """Pool simualted spike times (locations) and phases from all trains
        ready for circular-linear regression"""

        cell_pooled_spike_x = spike_times[valid]
        cell_pooled_spike_ph = spike_phases[valid]

        return cell_pooled_spike_x, cell_pooled_spike_ph

    def per_train_onset_offset_phase(self, spike_times, spike_phases, valid):
        """Train-mean onset and offset phases of simulated spikes

        train_empirical_phi_0 and train_empirical_phi_f will be
            the empirical Phi_0 and Phi_f of the model"""

        first_cycle_spikes = self._mark_first_cycle_spikes(spike_times, valid)
        last_cycle_spikes  = self._mark_last_cycle_spikes(spike_times, valid)


        train_empirical_phi_0 = circ_mean(spike_phases, w=first_cycle_spikes, device=self.device)
        train_empirical_phi_f = circ_mean(spike_phases,  w=last_cycle_spikes, device=self.device)

        return train_empirical_phi_0, train_empirical_phi_f

    def onset_offset_spikes(self, spike_times, valid):
        """Spike times of onset and offset spikes

        Returns lists of tensors"""

        first_cycle_spikes = self._mark_first_cycle_spikes(spike_times, valid)
        last_cycle_spikes  = self._mark_last_cycle_spikes(spike_times, valid)

        first_cycle_spikes_phase = [sp[v] for sp, v in zip(spike_times, first_cycle_spikes)]
        last_cycle_spikes_phase  = [sp[v] for sp, v in zip(spike_times,  last_cycle_spikes)]

        return first_cycle_spikes_phase, last_cycle_spikes_phase

    def cell_pooled_onset_offset_phase(self, spike_times, spike_phases, valid):
        """Onset and offset phases of simulated spikes with all trains pooled"""

        train_empirical_phi_0, train_empirical_phi_f = self.per_train_onset_offset_phase(spike_times, spike_phases, valid)

        cell_empirical_phi_0 = circ_mean(train_empirical_phi_0)
        cell_empirical_phi_f = circ_mean(train_empirical_phi_f)

        return cell_empirical_phi_0, cell_empirical_phi_f

    def _mark_first_cycle_spikes(self, spike_times, valid):
        """Return binary mask indicating the first valid spike and all the other spikes in the same theta cycle as the first spike
        
        has the same shape as spike_times or valid"""

        spike_bins = spike_times // (1/self.model._lfp_theta_frequency)
        first_nonzero_list = []
        for bins in (spike_bins+1) * valid:
            if len(bins)>0 and bins.sum()>0:
                first_nonzero_list.append(torch.nonzero(bins)[0])
            else:
                first_nonzero_list.append(torch.tensor([-1]))
        if not first_nonzero_list:
            return torch.zeros(spike_times.shape) == 1 # all False
        first_nonzero = torch.concat(first_nonzero_list)
        first_spike_bin = pick_element_tensor(spike_bins, first_nonzero).unsqueeze(1).to(self.device)
        first_cycle_spikes = (spike_times < (1+first_spike_bin) / self.model._lfp_theta_frequency) & valid
        return first_cycle_spikes

    def _mark_last_cycle_spikes(self, spike_times, valid):
        """Return binary mask indicating the last valid spike and all the other spikes in the same theta cycle as the last spike
        
        has the same shape as spike_times or valid"""
        
        spike_bins = spike_times // (1/self.model._lfp_theta_frequency)
        last_nonzero_list = []
        for bins in (spike_bins+1) * valid:
            if len(bins)>0 and bins.sum()>0:
                last_nonzero_list.append(torch.nonzero(bins)[-1])
            else:
                last_nonzero_list.append(torch.tensor([-1]))
        if not last_nonzero_list:
            return torch.zeros(spike_times.shape) == 1 # all False
        last_nonzero = torch.concat(last_nonzero_list)
        last_spike_bin = pick_element_tensor(spike_bins, last_nonzero).unsqueeze(1).to(self.device)
        last_cycle_spikes = (spike_times >= last_spike_bin/self.model._lfp_theta_frequency) & valid
        return last_cycle_spikes


class CircularLinearRegressor():
    def __init__(self, x, phi, sInt, ds=0.1, epsilon=0.01, optimizer_step_size=1e-5, n_optimization_steps=1000, device=None):

        self.device = device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # validate finiteness
        finite = [~torch.isnan(x) & ~torch.isnan(phi) for x, phi in zip(x, phi)]
        x   = [x[fin] for x, fin in zip(x, finite)]
        phi = [phi[fin] for phi, fin in zip(phi, finite)]

        # convert to tensor with the first dimension denoting batches
        X, Ns, mask = self._make_batch(x)
        PHI = self._make_batch(phi)[0]

        # convert numpy arrays to tensors
        x         = torch.tensor(X).to(self.device)
        phi       = torch.tensor(PHI).to(self.device)
        self.Ns   = torch.tensor(Ns).to(self.device)
        self.mask = torch.tensor(mask).to(self.device)

        # push phi to [0, 2*pi)
        phi %= 2*np.pi # push to [-2*pi, 2*pi]
        phi += 2*np.pi # push to [0, 4*pi]
        phi %= 2*np.pi # modulo to [0, 2*pi]

        self.x = x
        self.phi = phi
        self.sInt = sInt
        self.ds = ds
        self.optimizer_step_size  = optimizer_step_size
        self.n_optimization_steps = n_optimization_steps

    def regresscl(self):
        """Circular-linear regression and correlation using PyTorch

        Can simultaneously fits many circular-linear regression models to data in parallel
        thereby significantly speeding up computations

        Returns (all numpy.ndarray's):
            Slopes (float):
                Slope of circular-linear regression optimized using SGD
            Phi_0 (float):
                Phase offset of circular-linear regression
            rho (float):
                Circular-linear correlation coefficient
            tval (float):
                t-value for significance testing of the circular-linear regression
            p (float):
                p-value for significance testing of the circular-linear regression
            Rmax:
                Maximum resultant vector length obtained by the optimization process

        Originally developed by Kempter et al., (2012) in Matlab,
        Translated by Sia Ahmadi to PyTorch (11/8/2023)
        """

        # find best circular-linear fit with SGD
        s_min, rbar_min = self._optimize_regresscl(self.x, self.phi, self.Ns)
        max_mean_resultant_vector = -rbar_min

        s_min_tensor = torch.tensor(s_min).unsqueeze(1).unsqueeze(2).to(self.device)
        _, nu = self.RBar(s_min_tensor, self.x, self.phi, self.Ns)

        # compute statistics
        phi0  = torch.angle(nu)

        theta  = torch.angle(torch.exp(2*np.pi*1j*torch.abs(s_min_tensor)*self.x))
        thmean = torch.angle(self._sum_minus_dummy(torch.exp(1j*theta), keepdim=True))
        phmean = torch.angle(self._sum_minus_dummy(torch.exp(1j*self.phi)  , keepdim=True))

        sthe = torch.sin(theta - thmean)
        sphi = torch.sin(  self.phi - phmean)

        c12 = self._sum_minus_dummy(sthe*sphi)
        c11 = self._sum_minus_dummy(sthe*sthe)
        c22 = self._sum_minus_dummy(sphi*sphi)
        # rho = c12/torch.sqrt(c11*c22) # as implemented in the original, but can be rewritten as follows for numerical reasons
        rho = torch.sign(c12) * torch.exp(torch.log(c12.abs()) - .5 * (torch.log(c11.abs()) + torch.log(c22.abs())))

        lam22 = self._sum_minus_dummy(sthe**2 * sphi**2) / self.Ns
        lam20 = self._sum_minus_dummy(          sphi**2) / self.Ns
        lam02 = self._sum_minus_dummy(sthe**2          ) / self.Ns
        tval  = rho*torch.sqrt(self.Ns*lam20*lam02/lam22)

        p = 1. - scipy.special.erf(np.abs(tval.cpu().numpy())/np.sqrt(2))
        
        return (s_min,
                phi0.cpu().numpy().squeeze(),
                rho.cpu().numpy().squeeze(),
                tval.cpu().numpy().squeeze(),
                p.squeeze(),
                max_mean_resultant_vector.squeeze())

    def _optimize_regresscl(self, x, phi, Ns):
        """Find optimal slope values for circular-linear regression with SGD"""

        # initialize points at the center of length-ds bins in the sInt interval
        s = torch.stack([torch.arange(self.sInt[0], self.sInt[1], self.ds, device=self.device) + self.ds/2 for _ in range(Ns.numel())]).unsqueeze(2).requires_grad_()

        # optimizer
        optim = torch.optim.SGD([s], lr=self.optimizer_step_size)

        # multi-variate gradient of RBar with respect to RBar (which is an array of all ones)
        ones = torch.ones(Ns.numel(), s.size(1), device=self.device)

        # loop
        toc_forward = toc_backward = toc_update = 0
        tic_total = time.time()
        for step in range(self.n_optimization_steps):
            tic = time.time()
            y, _ = self.RBar(s, x, phi, Ns)
            toc_forward += time.time() - tic

            tic = time.time()
            y.backward(gradient=ones)
            toc_backward += time.time() - tic

            tic = time.time()
            optim.step()
            toc_update += time.time() - tic

            optim.zero_grad()

            if (1+step) % 100 == 0:
                print(f"Optimization: done {1+step} steps in {time.time() - tic_total:.3f} s.")
                print(f"\tForward pass: {toc_forward:.3f}s.")
                print(f"\tBackward pass: {toc_backward:.3f}s.")
                print(f"\tParam updates: {toc_update:.3f}s.")

                toc_forward = toc_backward = toc_update = 0

        # found it
        rbar, _ = self.RBar(s, x, phi, Ns)
        best = torch.argmin(rbar, dim=1).detach().cpu().numpy()

        # extract optimum from array of inputs
        rbar_min = pick_element_array(rbar.detach().cpu().numpy(), best)
        s_min    = pick_element_array(   s.detach().cpu().numpy().squeeze(), best)

        return s_min, rbar_min

    def _sum_minus_dummy(self, x, dim=2, keepdim=False):
        """Sum tensor along dim but subtract the sum of the invalid values"""

        sums  = torch.sum(x*self.mask, dim=dim, keepdim=keepdim)
        return sums

    def _make_batch(self, X, eps=1e-10):
        """Make 3-d tensor of inputs, padding the last dimension with `eps`

        Args:
            X
                list of lists of numbers
            eps
                padding value (default=1e-10)

        Returns
            X
                n-by-1-by-N tensor where n==len(x) and N is the
                length of the longest list in X
            mask
                boolean mask of valid entries in X
        """

        lengths = list(map(len, X))
        maxlen = max(lengths)
        Ns = np.array(lengths)[:, np.newaxis]


        mask = [np.r_[np.ones((len(x),))==1,np.ones((maxlen-len(x),))==0][np.newaxis, np.newaxis, :] for x in X]
        mask = np.concatenate(mask)

        X = [np.r_[x,eps*np.ones(maxlen-len(x))][np.newaxis, np.newaxis, :] for x in X]
        X = np.concatenate(X)

        return X, Ns, mask

    def RBar(self, a, x, phi, Ns):
        """RBar as defined in regresscl by Kempter et al., (2012)

        Vectorized, so can run on GPU with PyTorch"""

        y0 = 2*np.pi*a*x
        y1 = 1j*(phi - y0)
        exp = torch.exp(y1)
        sums = self._sum_minus_dummy(exp, dim=2)
        mean = sums / Ns

        return -torch.abs(mean), mean


def pick_element_tensor(x, i):
    """Pick a single element from each row of x

    x and i must be the same length

    returns y where y[j] == x[j, i[j]]"""

    x = torch.atleast_2d(x)
    
    return torch.tensor([r[b] if b!=-1 else torch.nan for r, b in zip(x, i)])

def pick_element_array(x, i):
    """Pick a single element from each row of x

    x and i must be the same length

    returns y where y[j] == x[j, i[j]]"""

    x = np.atleast_2d(x)

    return np.array([r[b] for r, b in zip(x, i)])

def heaviside(x):
    """Heaviside function"""

    x[x<0] = 0
    return x

def circ_mean(alpha, w=None, dim=1, device=None):
    """Mean direction for circular data
    Args:
        alpha
            sample of angles in radians
        w
            weightings in case of binned angle data
        dim (ignored for now)
            dimension of alpha along which to calculate the circular mean

    Returns:
        mu
            mean direction
        ul (not translated)
            upper 95% confidence limit
        ll (not translated)
            lower 95% confidence limit

    PHB 7/6/2008

    References:
        Statistical analysis of circular data, N. I. Fisher
        Topics in circular statistics, S. R. Jammalamadaka et al.
        Biostatistical Analysis, J. H. Zar

    Circular Statistics Toolbox for Matlab

    By Philipp Berens, 2009

    berens@tuebingen.mpg.de - www.kyb.mpg.de/~berens/circStat.html

    Translated to Python by Sia Ahmadi"""

    was_tensor = isinstance(alpha, torch.Tensor)

    if not was_tensor:
        alpha = torch.tensor(alpha,device=device)

    # check vector size
    alpha = torch.atleast_2d(alpha)

    if w is None:
        w = torch.ones(alpha.shape,device=alpha.device)
    else:
        if not isinstance(w, torch.Tensor):
            w = torch.tensor(w,device=alpha.device)
        w = torch.atleast_2d(w)

    # compute weighted sum of cos and sin of angles
    r = torch.sum(w * torch.exp(1j*alpha), dim=dim)

    # obtain mean by
    mu = torch.angle(r)

    # push mu to [0, 2*pi)
    mu %= 2*np.pi

    return mu if was_tensor else mu.cpu().numpy()


def inhomopp(intens, T=1, expected_n_particles=1, n_samples=None, device='cpu', seed=None):
    """Generate a inhomogeneous poisson process on [0,T]
    with intensity function intens

    Modified from http://sas.uwaterloo.ca/~dlmcleis/s906/slides151-200
    Slide 11, by DON L. McLEISH

    Args:
        intense (function handle):
            function handle with a single input of floats and output of floats
        T (float) (default=1):
            upper bound of the interval over which to sample the function
            (default=1)
        expected_n_particles (integer) (default=1):
            expected number of points per sample (if T=1 (default),
            this can be thought of average firing rate); can be a scalar or a
            length n 1-d array (of firing rates) for n independent samples
        n_sample (integer) (default=None):
            number of samples to genrate if expected_n_particles is scalar, ignored otherwise
        device (string) (default='cpu'):
            PyTorch device to execute the code on
    Returns:
        points:
            2-d tensor of n_particles-by-n_samples generated
            from inhomogeneous Poisson process generated by
            `intens` and expected number of valid particles
            given by `expected_n_particles`.
        admissible:
            mask for valid samples in `points`

    Translated to PyTorch by Sia Ahmadi 2023/11/07
    """


    rng = torch.Generator(device)
    if seed is not None:
        rng.manual_seed(seed)

    expected_n_particles = torch.tensor(expected_n_particles, device=device)
    assert expected_n_particles.dim() in [0, 1]

    if expected_n_particles.dim() == 1 or n_samples is None:
        n_samples = 1 if expected_n_particles.dim()==0 else expected_n_particles.size(0)
    expected_n_particles = torch.ones((1, n_samples), device=device) * expected_n_particles

    x = torch.linspace(0, T, 1000, device=device)
    l_unnormalized = intens(x)
    if torch.all(l_unnormalized==0):
        return torch.ones((0,0)), torch.ones((0,0))==1
    auc = torch.trapz(l_unnormalized, x)
    l = l_unnormalized/auc*expected_n_particles.T
    lam0, _ = torch.max(l, dim=1, keepdim=True)
    n_u = int(np.ceil(1.5*T*lam0.max().cpu().numpy()))
    u = torch.rand((n_samples,n_u), device=device, generator=rng) # generate homogeneouos poisson process

    x = torch.cumsum(-(1/lam0)*torch.log(u), 1) # points of homogeneous pp
    admissible_1 = x < T # select those `l` points less than T
    n = torch.sum(admissible_1, dim=1)
    x = x[:, :n.max()]
    admissible_1 = admissible_1[:, :n.max()]
    l = intens(x)/auc*expected_n_particles.T # evaluates intensity function

    l_thresh = l/lam0
    admissible_2 = torch.rand((n_samples, n.max()),device=device, generator=rng) < l_thresh
    admissible = admissible_1 & admissible_2
    return x, admissible

