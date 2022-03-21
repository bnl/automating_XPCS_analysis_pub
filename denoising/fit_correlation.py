import numpy as np
from scipy.optimize import curve_fit#, minimize
from copy import deepcopy
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from joblib import Parallel, delayed

import denoising.g2 as G2
from denoising.fit_parameters import *
from denoising.results import *

import warnings

warnings.simplefilter("ignore", category=RuntimeWarning )

TRUST_REGION_LOWER_BOUND = 5
TRUST_REGION_UPPER_BOUND_COEF = 2
MAXFEV = 10000


def sq_r_mse(x, x_hat):
    return np.sqrt(np.mean((x-x_hat)**2))

def r_squared(x, x_hat):
    """ Calculates R2 for data x and its approximation x_hat"""
    rss = np.sum((x-x_hat)**2)
    x_mean = np.mean(x)
    tss = np.sum((x-x_mean)**2)
    
    return 1 - rss/tss

def stretched_decay(x, beta1, gamma, g_inf, alpha):    
    return beta1 * np.exp(-2*(gamma * x)**alpha) + g_inf

def fit_stretched_exponent(x , y , y_er ,
                            stretched_exp_params = None,
                            cut_tails = False
                          ):
    """
    Fits input data to a stretched exponential function by trying
    several initial parameters and minimizing the R^2 error.
    
    Parameters:
    ----------
    x - array
        the x coordinates (time, frames, etc.)
    
    y - array
        signal to fit
    
    y_er - array
        errors for the signal y. If NaN or zero, then ignored.
    
    stretched_exp_params - instance of class Stretched_Exponent_Parameters. 
        It contains bounds (lower_bounds, upper_bound) for the fitting parameters
        and initial guesses.
        The order of parameters is
        [ amplitude (beta), rate (Gamma), ginf, compression contstant (alpha)]

    cut_tails - boolean
        whether to reduce the impact of the points beyond initial drop of the contrast by factor of e
   
    Returns:
    -------
    (res_vals - values of parameters
     res_cov - covariance metrics of the fit
     r2_fit - R-squared measure of fit)

    (res_vals_no_tail ,
     res_cov_no_tail ,
     r2_fit_no_tail) - analogous values for the fit with cut_tails=True
        ignore when cut_tails=False
    """
       
    if stretched_exp_params is None:
        stretched_exp_params = Stretched_Exponent_Parameters()
    assert( isinstance( stretched_exp_params , Stretched_Exponent_Parameters ) )
       
    N = len(y)
       
    y_fit = np.ones(N) * np.mean(y)
    err2_min = sq_r_mse(y, y_fit)
    
    res_vals = [ np.nan ] * stretched_exp_params.num_params
    res_cov  = np.zeros( ( stretched_exp_params.num_params , stretched_exp_params.num_params ) )

    res_vals_no_tail = [ np.nan ] * stretched_exp_params.num_params
    res_cov_no_tail = np.zeros( ( stretched_exp_params.num_params , stretched_exp_params.num_params ) )
    r2_fit_no_tail = np.nan
    
    if N == 0:
        return (res_vals , res_cov , 0), (res_vals_no_tail , res_cov_no_tail , r2_fit_no_tail)
    
    try:
        if stretched_exp_params.gamma.start_value is None:

                for j in np.linspace(1,100, 5):
                    
                    # search for a better fit        
                    stretched_exp_params.gamma.start_value = np.min([j/N, stretched_exp_params.get_param_bounds()[1][1]])
                    if (np.isnan(y_er).any()) or (y_er==0).any():
                        best_vals, covar = curve_fit(stretched_decay, x, y, maxfev = MAXFEV,
                                                    p0 = stretched_exp_params.get_initial_values(),
                                                    bounds = stretched_exp_params.get_param_bounds())
                    else:
                        best_vals, covar = curve_fit(stretched_decay, x, y, maxfev = MAXFEV,
                                                    sigma = y_er, absolute_sigma = True,
                                                    p0 = stretched_exp_params.get_initial_values(),
                                                    bounds = stretched_exp_params.get_param_bounds())
                    #print(stretched_exp_params.get_param_bounds())
                    y_fit  = stretched_decay(x, *best_vals)
                    err2 = sq_r_mse(y, y_fit)

                    if err2 < err2_min:
                        err2_min = err2
                        break # stop when found a good fit

        else:
        
            if (np.isnan(y_er).any()) or (y_er==0).any():
                    best_vals, covar = curve_fit(stretched_decay, x, y, maxfev = MAXFEV,
                                                p0 = stretched_exp_params.get_initial_values(),
                                                bounds = stretched_exp_params.get_param_bounds())
            else:
                    best_vals, covar = curve_fit(stretched_decay, x, y, maxfev = MAXFEV,
                                                sigma = y_er, absolute_sigma = True,
                                                p0 = stretched_exp_params.get_initial_values(),
                                                bounds = stretched_exp_params.get_param_bounds())
            #print(stretched_exp_params.get_param_bounds())
            y_fit  = stretched_decay(x, *best_vals)
            

        if cut_tails: # assign weights to points where the signal dropped considerably
            stretched_exp_params_no_tail = deepcopy(stretched_exp_params)
            stretched_exp_params_no_tail.ginf.set_upper(best_vals[2] + 1e-4)#+ covar[2,2]**0.5)
            stretched_exp_params_no_tail.ginf.set_lower(best_vals[2] - 1e-4)#- covar[2,2]**0.5)
            stretched_exp_params_no_tail.ginf.set_initial_value(best_vals[2])

            weights = np.ones(len(x))
            beta = best_vals[0]
            ind = y_fit < 1 + beta/np.e
            weights[ind] = np.log(x[ind])
            
            res_vals_no_tail, res_cov_no_tail = curve_fit(stretched_decay, x, y, maxfev = MAXFEV,
                                                sigma = weights, absolute_sigma = False,
                                                p0 = stretched_exp_params_no_tail.get_initial_values(),
                                                bounds = stretched_exp_params_no_tail.get_param_bounds())
            # print(stretched_exp_params_no_tail.get_param_bounds(), best_vals[2])
            y_fit_no_tail  = stretched_decay(x, *res_vals_no_tail)
            r2_fit_no_tail = r_squared(y, y_fit_no_tail)

        res_vals = best_vals
        res_cov  = covar
    
    except:
        pass    

    r2_fit = r_squared(y, y_fit)
       
    return (res_vals , res_cov , r2_fit), (res_vals_no_tail , res_cov_no_tail , r2_fit_no_tail)

    

    

def _cutting_slice(rotated_2tcf, starting_time, step):
    """
    Takes a time-slice (1TCF) of rotated 2TCF between (starting_time, starting_time + step).
    Rotated 2TCF has a checkboard pattern of NaN values to unsure unshifted cuts,
    thus, the time step is doubled.
    
    Parameters:
    ----------
    rotated_2tcf - 2TCF rotated by 45 degrees
    starting time - int, the beginning of cut (in original, non-rotated 2TCF)
    step - width of the cut
    """
    cut = rotated_2tcf[2*starting_time:2*starting_time+2*step, :]
    y = np.nanmean(cut, axis = 0)
    y_er = np.nanstd(cut, axis = 0)
    
    ind = ~np.isnan(y)
    y = y[ind]
    y_er = y_er[ind]
    y_er[-1] = np.max(y_er)
    x = np.arange(y.shape[0])
    
    return x, y, y_er

def cutting_1tcf_slice(two_tcf, bin_center, bin_width):
    """
    Takes a time-slice (1TCF) of non-rotated 2TCF between centered at bin_center
    with width bin_width.
        
    Parameters:
    ----------
    two_tcf - 2TCF, single ROI
    bin_center - int of float, the center of the cut. When float, will be truncated to the closest integer.
    bin_width - width of the cut

    Returns:
    -------
    x - lag times, measured in frames
    y - 1TCF values
    y_er - 1TCF errors
    """
    rotated_2tcf = rotate_45_2tcf(two_tcf)
    starting_time = int(bin_center)-bin_width//2
    step = bin_width

    cut = rotated_2tcf[2*starting_time:2*starting_time+2*step, :]
    y = np.nanmean(cut, axis = 0)
    y_er = np.nanstd(cut, axis = 0)
    
    ind = ~np.isnan(y)
    y = y[ind]
    y_er = y_er[ind]
    y_er[-1] = np.max(y_er)
    x = np.arange(y.shape[0])
    
    return x, y, y_er



def extract_parameters_single(z,
                              stretched_exp_params = None,
                              step = 1,
                              cut_tails = False,
                              trust_region_lower_bound = TRUST_REGION_LOWER_BOUND,
                              trust_region_upper_bound_coeff = TRUST_REGION_UPPER_BOUND_COEF
                              ):
    
    
    """
    Extracts dynamics parameters from cuts of a single 2TCF. Cuts have equal width.
    
    Parameters:
    ----------
    
    z - 2D array
        the 2TCF
    
    stretched_exp_params - instance of class Stretched_Exponent_Parameters. 
        It contains bounds (lower_bounds, upper_bound) for the fitting parameters
        and initial guesses.
        The order of parameters is
        [ amplitude (beta), rate (Gamma), ginf, compression contstant (alpha)]
    
    step - int
        the width of the cut in frames

    cut_tails - boolean
        whether to reduce the impact of the points beyond initial drop of the contrast by factor of e

    trust_region_lower_bound - int
        the minimal number of frames it takes contrast to drop in half

    trust_region_upper_bound_coeff - float
        a minimal number of half-decay periods is contained in available points. For example,
        the contrast drops twice in 10 frames. If trust_region_upper_bound_coeff=3, one need to have 
        at least 30 time points in the 1TCF to trust the result of the fit.
      
    Returns:
    --------    
        fit_results.data_dict - dictionary with results of the fit
    
    """

    def compute_listcorr_from_2darrcov( cov ):
        # Convert covariance matrix to list of correlations
        listcorr = []
        for i in range( cov.shape[0] ):
            for j in range( i+1 , cov.shape[1] ):
                listcorr.append( np.abs( cov[i,j] / (cov[i,i]**0.5) / (cov[j,j]**0.5) ) )
        return listcorr
    
    
    size = z.shape[0]
    
    fit_func = fit_stretched_exponent
    if stretched_exp_params is None:
        stretched_exp_params = Stretched_Exponent_Parameters()
    num_params = len( stretched_exp_params.get_param_bounds()[0] )
    
    n_frames = np.zeros(size//step)
        
    fit_results = OneTimeFitParameterResults_SingleROI( stretched_exp_params , size , step )
    
    rotated_z = rotate_45_2tcf(z)
    
    # get the initial guess of the parameters from the cut with the most points (middle)

    t_middle = rotated_z.shape[1]//2
    if t_middle>10:
        large_step = np.max([step, 10])
    else:
        large_step = step

    # identify the initial guess
    x, y, y_er = _cutting_slice(rotated_z, t_middle, large_step)
    (initial_guess0, _, _), _ = fit_func(x[1:], y[1:], y_er[1:], stretched_exp_params = stretched_exp_params, cut_tails = cut_tails )    
    stretched_exp_params = factory_make_UniqueCollectionFitParameters( repr(stretched_exp_params) ,
                                                                         list_lower = stretched_exp_params.get_param_bounds()[0],
                                                                         list_upper = stretched_exp_params.get_param_bounds()[1],
                                                                         list_start_spec = initial_guess0 )  
    for j in range(0, size-step+1, step):        
        k = j //step
        x, y, y_er = _cutting_slice(rotated_z, j, step)  
        # print(j, fit_func(x[1:], y[1:], y_er[1:],stretched_exp_params = stretched_exp_params,cut_tails = cut_tails ) ) 
        res, res_no_tail = fit_func(x[1:], y[1:], y_er[1:], stretched_exp_params = stretched_exp_params, cut_tails = cut_tails )   
        val, cov, r2 = res
        val_no_tail, cov_no_tail, r2_no_tail = res_no_tail
        fit_results.set_datadict_attr_for_all_keys( 'parameters' , k , val )
        fit_results.set_datadict_attr_for_all_keys( 'parameters_errors' , k , [np.sqrt(cov[n,n]) for n in range(num_params)] )
        fit_results.set_datadict_attr_for_all_keys( 'fit_quality' , k , r2 )
        fit_results.set_datadict_attr_for_all_keys( 'correlations' , k , compute_listcorr_from_2darrcov( cov ) )

        fit_results.set_datadict_attr_for_all_keys( 'parameters_no_tail' , k , val_no_tail )
        fit_results.set_datadict_attr_for_all_keys( 'parameters_errors_no_tail' , k , [np.sqrt(cov_no_tail[n,n]) for n in range(num_params)] )
        fit_results.set_datadict_attr_for_all_keys( 'fit_quality_no_tail' , k , r2_no_tail )
        fit_results.set_datadict_attr_for_all_keys( 'correlations_no_tail' , k , compute_listcorr_from_2darrcov( cov_no_tail ) )

        n_frames[k] = len(y)
        
       
    # calculate trust regions 
    for fit_type in ['', '_no_tail']:
        trust_beta, trust_g_inf = calculate_trust_region_single(n_frames,
                                                                fit_results.data_dict[f'parameters{fit_type}']['gamma'] ,
                                                                fit_results.data_dict[f'parameters{fit_type}']['alpha'] ,
                                                                lower_bound = trust_region_lower_bound,
                                                                upper_bound_coeff = trust_region_upper_bound_coeff)
                                                                
        trust_gamma = trust_beta.copy()
        trust_alpha = trust_beta.copy()
        
        trust_cat = [trust_beta , trust_gamma, trust_g_inf, trust_alpha]
        for (i,p) in enumerate(trust_cat):
            fit_results.set_datadict_attr( f'trust_regions{fit_type}' , fit_results.param_names[i] , p )
    
    return fit_results.data_dict


def extract_parameters(g2,
                       stretched_exp_params_list=None, 
                       steps = 1,
                       save_log_filename='extract_params_log',
                       save_results_filename='extract_params_results',
                       cut_tails = False,
                       roi_list = None,
                       trust_region_lower_bound = TRUST_REGION_LOWER_BOUND,
                       trust_region_upper_bound_coeff = TRUST_REGION_UPPER_BOUND_COEF
                        ):
    
    
    """
    Extracts dynamics' parameters from cuts of multiple 2TCF 
    (performs extract_parameters_single() for each ROI)
    
    Parameters:
    ----------
    
    g2 - 2TCF
    
    save_log_filename - string
        address for saving the log containing the for parameters
    
    stretched_exp_params_list - list[ UniqueCollectionFitParameters ]
        list containing structures with the initial values.
        If only one element in the list or the list is not provided, then the same
        fit parameters are used for each ROI with default type Stretched_Exponent_Parameters.
        If different parameters need to be used for each ROI, the length should
        match the number of ROIs.
    
    steps - int or list(int)
        the widths of the cut in frames. If a single integer, will use the same time for all ROI.
        If a list of int, must contain the step size for each ROI, be the same lenght as roi_list.
        If the number of frames in 2TCF is not fully divided by the step, the remaining last frames
        are droped from the analysis. For example, thare are 51 frames and the step=7. Then first 49
        frames are analyzed and the last 2 frames are left out.

    save_log_filename - file to save fitting parameters

    save_results_filename - file to save results

    cut_tails - boolean
        whether to reduce the impact of the points beyond initial drop of the contrast by factor of e


    roi_list - list or numpy.array, default if None
        list of the ROIs that needs to be analyzed

    trust_region_lower_bound - int
        the minimal number of frames it takes contrast to drop in half

    trust_region_upper_bound_coeff - float
        a minimal number of half-decay periods is contained in available points. For example,
        the contrast drops twice in 10 frames. If trust_region_upper_bound_coeff=3, one need to have 
        at least 30 time points in the 1TCF to trust the result of the fit.

    Returns:
    --------
    global_result - nested dictionary that contains:
        params - [standard and with truncated tails] parameters values
            extracted from each 1TCF cut
        
        params_err - [standard and with truncated tails] errors of
            parameters values extracted from each 1TCF cut
        
        fit_quality (r2_fit) - array, shape(N_roi, N_cuts)
            R-squared measure of goodness-of-fit
        
        ages (in frames) - the mean ages (bin centers) and width of cuts (bin width) 
            values for the time bins for 1TCF cuts
    
    """
    # by default, process all ROIs
    if roi_list is None:
        roi_list = np.arange(g2.shape[-1])        
    
    roi_list = np.array(roi_list, dtype = 'int')

    if stretched_exp_params_list is None:
        stretched_exp_params_list = [Stretched_Exponent_Parameters()]*len(roi_list)
    
    if isinstance(steps, int):
        steps = [steps]*len(roi_list)

    g2_selected = g2[:, :, roi_list]

   
    assert( [ isinstance( pi , UniqueCollectionFitParameters ) for pi in stretched_exp_params_list ] )
    assert( len(stretched_exp_params_list) == len(roi_list) )
    assert( len(steps) == len(roi_list) )
    
    g12 = G2.get_3d_g2_array_Nframe_x_Nframe_x_N_roi(g2_selected)

    
    rois = OneTimeFitParameterResults_MultipleROI( stretched_exp_params_list , g12.shape[0] , steps )
    
    fitting_parameters = []
    
    j_array = [*range(len(stretched_exp_params_list))]
        
    def extract_parameters_single_for_j(j):
        
        return extract_parameters_single(g12[:, :, j],
                              stretched_exp_params_list[j],
                              steps[j],
                              cut_tails = cut_tails,
                              trust_region_lower_bound = trust_region_lower_bound,
                              trust_region_upper_bound_coeff = trust_region_upper_bound_coeff
                              )
    
    fit_results = Parallel(n_jobs=-1, backend='threading')(delayed(extract_parameters_single_for_j)(j) for j in j_array)

    
    for roi, roi_result in enumerate(fit_results):
        fitting_parameters.append(get_fitting_parameters(stretched_exp_params_list[roi], trust_region_lower_bound , trust_region_upper_bound_coeff))
        for field in rois.roi_fields:
            for p in roi_result[field]:
                rois.set_roi_dict_attr( roi , field , p , roi_result[field][p] )
                        
    # record the hyper-parameters
    write_to_file(fitting_parameters, save_log_filename )
        
    global_result = rois.get_nesteddict_roi_field_paramkey(roi_list)
    
    # save the results
    write_to_file(global_result, save_results_filename )
    
        
    return global_result
        
def calculate_half_time(gamma, alpha):
    '''
    Calculates the time it takes for the beta to drop by half
    of the initial value, based on rate and compression constant
    
    Parameters:
    ----------
    gamma - rate
    alpha - compression constant
    '''
    # may need to add handling of very small cases for gamma
    return 1/gamma*(0.5*np.log(2))**(1/alpha)


def calculate_trust_region_single(n_frames, 
                            gamma, 
                            alpha, 
                            lower_bound = TRUST_REGION_LOWER_BOUND,
                            upper_bound_coeff = TRUST_REGION_UPPER_BOUND_COEF, 
                            ):
    """
    Evaluate whether values of beta and g_inf are trustable
    based on half_time to number_of_frames ratio.
    
    n_frames - array of int,
            how many frames in the fit
    gamma - array of floats,
            rate of dynamics (stretched exponent parameter)
    alpha - array of floats,
            rate of dynamics (stretched exponent parameter)
    lower_bound - int,
            minimal number of frames
    upper_bound_coeff - float,
            minimal number of half-times (within the available frames)
    """

      

    half_time = calculate_half_time(gamma, alpha)
    trust_beta = np.array([True]*(len(gamma)))
    trust_g_inf = np.array([True]*(len(gamma)))
    upper_bound = upper_bound_coeff*half_time
    
    
    # work with beta
    trust_beta[n_frames < 5] = False # not enough points to make a fit
    trust_beta[n_frames < upper_bound] = False
    trust_beta[half_time < lower_bound] = False
    
    # work with g_inf
    trust_g_inf[n_frames < 5] = False # not enough points to make a fit
    trust_g_inf[n_frames < upper_bound] = False # dynamics is too slow

    
    return trust_beta, trust_g_inf


def get_fitting_parameters(sxp, trust_region_lower_bound , trust_region_upper_bound_coeff):
    return { 'beta_limits': sxp.beta.limits,
    'gamma_limits': sxp.gamma.limits,
    'ginf_limits': sxp.ginf.limits,
    'alpha_limits': sxp.alpha.limits,
    'beta_start': sxp.beta.start_value,
    'gamma_start': sxp.gamma.start_value,
    'ginf_start': sxp.ginf.start_value,
    'alpha_start': sxp.alpha.start_value,
    'lower_bound_frames': trust_region_lower_bound,
    'upper_bound_half_decays': trust_region_upper_bound_coeff}


def rotate_45_2tcf(z):
    """
    Takes half of the 2TCF, rotates it in 45 to take slices horizontaly.
    
    Parameters:
    ----------
    z -2d numpy.array
        2TCF, shape (N_frames, N_frames)
    
    Returns:
    -------
    rotated_z-2d numpy.array
        rotated half of 2TCF, shape (2*N_frames, N_frames)
    """
    
    n = z.shape[0]
    rotated_z = np.zeros((2*n, n))
    rotated_z[:,:] = np.nan
    for j in range(1, n):
        rotated_z[j:-j:2, j] = np.diagonal(z, j)
    rotated_z[::2, 0] = np.diagonal(z,0)
    
    return rotated_z

