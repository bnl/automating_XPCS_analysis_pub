from denoising.fit_correlation import stretched_decay
import numpy as np
from scipy.interpolate import interp1d
from joblib import Parallel, delayed

def get_one_time_from_two_time( g12 ):
    
    """
    Calculates one-time correlation function from 
    two-time correlation function g12.
    
    Parameters:
    ----------
    g12 - 2D array
    2TCF for a single ROI. Shape is (N_frames, N_frames)
    
    Returns:
    --------
    g2f12 - 1D array
    1TCF
    
    error_bars - 1D array
    errors for corresponding values of 1TCF, calucalated with 
    the Central Limit Theorem
    """
    
    assert( len(g12.shape) == 2 )
    
    m,n = g12.shape 
    g2f12 = np.array( [ np.nanmean( g12.diagonal(i)) for i in range(m) ] )
    error_bars = np.array(  [ (np.nanstd( g12.diagonal(i)))/np.sqrt(g12.diagonal(i).shape[0]) for i in range(m) ] )
    
    return g2f12, error_bars

def interpolate_parameter(parameter, data_roi, original_length, output_lenght):
    """
    Changes the size of the mesh on which the `parameter` is defined from `original_length`
    to `output_length`
    
    Parametrs:
    ---------
    parameter - string
        name of the parameter ('beta', 'gamma', 'ginf' or 'alpha')
    
    data_roi - dict
        the dictionary for the fitting results for a single roi
    
    original_lenght - int
        length of the parameter vector (in resuls file)
    
    output_lenght - int
        length of the output parameter vector
    
    Returns:
    -------
    p - 1D numpy.array
        interpolation of parameter vector
    
    """
    p = interp1d(np.linspace(0,1, original_length),
                 data_roi['parameters'][parameter])(np.linspace(0,1, output_lenght))
    return p

def reconstruct_2tcf_from_fit_parameters_single_roi(data, size, n_roi):
    
    """
    Reconstruct 2TCF using stretched exponent and the result of the fit
    for a single ROI.
    
    Parameters:
    ----------
    data - dict
        results of the fit (keys structure: roi : parameters : parameter_name)

    size - int
        size of the output 2CTF in frames. Most of the time, this is the 
        size of the original experimental 2TCF. When fit was done on 
        binned frames, the length of the parameters is smaller. This also 
        can be used adjusted when comparing experiments collected
        at different rates.

    n_roi - int
        index of the ROI to be reconstructed, must be present in `data`. 
        
    Returns:
    -------
    2D numpy.array, 2TCF
        
    """
    
 
    fit_size = len(data[n_roi]['parameters']['beta'])# number of points in the fit
                                                     # can be smaller than original if binned fit
    a = np.zeros((size, size))    

    # interpolate the parameters values to match the new output size
    beta = interpolate_parameter('beta', data[n_roi], fit_size, size )
    gamma = interpolate_parameter('gamma', data[n_roi], fit_size, size )
    ginf = interpolate_parameter('ginf', data[n_roi], fit_size, size )
    alpha = interpolate_parameter('alpha', data[n_roi], fit_size, size )

    for j in range(size):
        str_y = [stretched_decay(j, *p) for p in zip(beta[j//2: -j//2],
                                                     gamma[j//2: -j//2],
                                                     ginf[j//2: -j//2], 
                                                     alpha[j//2: -j//2])]
        np.fill_diagonal(a[j:, :], str_y)

    a = a + a.T
    str_y = [stretched_decay(0, *p) for p in zip(beta[j//2: -j//2],
                                                 gamma[j//2: -j//2],
                                                 ginf[j//2: -j//2], 
                                                 alpha[j//2: -j//2])]
    np.fill_diagonal(a[:, :], str_y)
        
    return a

def reconstruct_2tcf_from_fit_parameters(data, size):
    """
    Reconstruct 2TCF using stretched exponent and the result of the fit
    for all ROI fields in the `data`.
    
    Parameters:
    ----------
    data - dict
        results of the fit (keys structure: roi : parameters : parameter_name)
    size - int
        size of the output 2CTF in frames. Most of the time, this is the 
        size of the original experimental 2TCF. When fit was done on 
        binned frames, the length of the parameters is smaller. This also 
        can be used adjusted when comparing experiments collected
        at different rates.
        
    Returns:
    -------
    res -dictionary: keys are ROI index, values are reconstructed 2TCF
    """
    res = {}
    roi_array = list(data.keys())
    reconstr_results = Parallel(n_jobs=-1, backend='threading')(delayed(reconstruct_2tcf_from_fit_parameters_single_roi)(data, size,j) for j in roi_array)
    
    for j, val in enumerate(reconstr_results):
        res[roi_array[j]] = reconstr_results[j]
    
    return res



def get_full_reconstructed_2tcf_single_roi(data, exp_frames, delay_frames, n_roi):
    
    """
    Reconstruct 2TCF from fitting parameters to an arbitrary delays (including
    beoynd observed in the measurements) for a single ROI.
    
    Parameters:
    ----------
    data - dict
        results of the fit (keys structure: roi : parameters : parameter_name)
        
    exp_frames - int
        most likely, the number of frames in the experiment. If binned time slices
        were used for the fitting, this number if larger than the length of the
        parameter vector. Cannot be smaller than the length of the parameter vector
        
    delay_frames - int
        the desired number of frames one wants to propagate the reconstruction of 2TCF.
        Can be larger or smaller than exp_frames.
        
    n_roi - int
        index of the ROI to be reconstructed, must be present in `data`.  
        
    Returns:
    -------
    c - 2D numpy.array,
        reconstracted 2TCF
        
    trust_beta_interpolated - 2D numpy.array
    
    """
    if   len(data[n_roi]['ages']['bin_centers']) *  data[n_roi]['ages']['bin_width'] < exp_frames:
        exp_frames = len(data[n_roi]['ages']['bin_centers']) *  data[n_roi]['ages']['bin_width'] 
        print(f'not all data avaiable, using first {exp_frames} frames')
    n_frames = len(data[n_roi]['parameters']['beta'])
    c = np.zeros((exp_frames, delay_frames))
    
    beta = interpolate_parameter('beta', data[n_roi], n_frames, exp_frames )
    gamma = interpolate_parameter('gamma', data[n_roi], n_frames, exp_frames )
    ginf = interpolate_parameter('ginf', data[n_roi], n_frames, exp_frames )
    alpha = interpolate_parameter('alpha', data[n_roi], n_frames, exp_frames )
    
    def get_stretched_decay_j(j):
        p = (beta[j],gamma[j], ginf[j], alpha[j])
        str_y = stretched_decay(np.arange(delay_frames), *p) 
        return str_y
    
    c = Parallel(n_jobs=-1, backend='threading')(delayed(get_stretched_decay_j)(j) for j in range(exp_frames))
    c = np.array(c)
    
    trust_beta = data[n_roi]['trust_regions']['beta']
    trust_beta_interpolated = np.empty((exp_frames, delay_frames))
    ratio = exp_frames//n_frames

    # if n_frames+1 > len(trust_beta):
    #     trust_beta = np.array([*trust_beta, False])
    
    for j in range(exp_frames):
        trust_beta_interpolated[j, :] = trust_beta[j//ratio]
    
    return c, trust_beta_interpolated

