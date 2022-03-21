import numpy as np
import torch
import denoising.nets as nets
import denoising.g2 as G2
from denoising.latent_space import LatentSpaceExtractor
from scipy.stats import gaussian_kde
from joblib import Parallel, delayed

class Scoring:
    """
        Parent class for possible scorings.
        Each scoring methond must contain function score_denoised()
    """

    def __init__(self):
        pass

    def score_denoised(self, x):
        pass

    def better(self, val1, val2):
        return val1 > val2

class LatentSpaceScoring(Scoring):
    """
    Class for evaluating the accuracy score of denoising 2TCF based on the 
    latent space coordinate distribution. 
    The scores based on the latent space are low for accurate predictions and high for potentially inacurate (outliers).
    """
    def __init__(self, autoencoder_filepath, latent_space_scaler_file, distance_ecdf_file, distance_median_file, mode = 'probability'):
        self.lat_space_extractor = LatentSpaceExtractor(autoencoder_filepath, latent_space_scaler_file, distance_ecdf_file, distance_median_file)
        self.mode = mode

    def score_denoised(self, x, x_denoised = None):

        """
        self.mode can be 'probability' or 'relative_distance'. If mode = 'probability',
        calculates the ECDF(x) ccalculated for the training set.
        If  mode = 'relative_distance', calculates the ratio between x and the median distance to the center
        for the examples in the training set.

        This function doesn't need x_denoised and it only works with the latent space representation.
        """

        lat_vector = self.lat_space_extractor.get_latent_space(G2.G2(x))
        distance_to_center = self.lat_space_extractor.calculate_distance_to_training_center(lat_vector)
        if self.mode == 'probability':
            score = self.lat_space_extractor.calc_prob_of_distance_too_big(distance_to_center)
        elif self.mode == 'relative_distance':
            score = self.lat_space_extractor.evaluate_relative_distance(distance_to_center)
        
        return np.ones( x.shape ) * score

    def better(self, val1, val2):
        if self.mode == 'probability':
            return val1 > val2
        elif self.mode == 'relative_distance':
            return val1 < val2






class AutoCorrScoring(Scoring):
    
    """
        Class for calculating the accuracy score of denoising a 2TCF with the
        autoencoder model. The score is based on the pdf of the reconstruction errors
        of the training set. The reconstruction error is calculated as autocorrelation coefficient
        at lag 1 for residuals, integrated along either rows or columns (2D-> 1D).
        Currently, it is not normalized.

        Public interface:
        ----------
        
        AutoCorrScoring::AutoCorrScoring( model_filepath:str , rec_errors_training_path:str = None , autocorr_lag:int = 1 )
        AutoCorrScoring::score_denoised( x:np.array , x_denoised:np.array ) -> np.array
    
    """

    
    # Public methods
    
    def __init__( self , model_filepath , rec_errors_training_path=None , autocorr_lag=1 ):
        
        """
        model_filepath  = string filepath to instance of AutoEncoder_2D
        rec_errors_training_path = string filepath to location of model errors on the training set. 
            Default is None. If None, then the scoring is found automatically.
        """
        
        self.__scoring_kernel = self.__load_scoring_kernel( model_filepath , rec_errors_training_path )
        self.__autocorr_lag   = autocorr_lag
        self.__max_score = self.__find_max_score() # finds the approximate value of the maximum value for normalization
    
        
    def score_denoised( self , x , x_denoised ):
        
        """Calculates accuracy score between the noisy input and denoised output
        using scoring_kernel.
        
        Parameters:
        ----------
        x - array
        input of the model
        
        x_denoised - array
        output of the model
                
        """
        
        error = np.mean( ( x_denoised - x ) , axis=(1) )
        if self.__scoring_kernel is None:
            score = np.nan
        else:
            score = self.__scoring_kernel( self.__autocorr( error , self.__autocorr_lag ) )
            score = np.min([1, score/self.__max_score]) # limit to 1 since the max_score is approximate
        
        return np.ones( x.shape ) * score
    
    
    # Private methods
    
    def __load_scoring_kernel( self , model_filepath , rec_errors_training_path=None ):
        
        if rec_errors_training_path is None:
            kernel = model_filepath.split("kernel_")[-1].split(".")[0]
            try:
                rec_errors_training_path = (
                    f"model_files/rec_errors/reconstraction_error_kernel_{kernel}.pt"
                )
                
            except:
                print('Cannot load the reconstruction errors file. Proceed without accuracy estimation.')
                return None
        rec_errors_training = torch.load(rec_errors_training_path)
        scoring_kernel = gaussian_kde(rec_errors_training)
            

        
        return scoring_kernel

    def __find_max_score(self):
        """
        Estimates the maximum value of pdf for error distribution. 
        Need this for convenience of interpretation of model's accuracy score.
        """
        value_range = np.linspace(-1,1,2000)
        max_index = np.argmax(self.__scoring_kernel(value_range))
        max_value = self.__scoring_kernel(value_range[max_index])

        return max_value

    
    def __autocorr( self , x , t ):
        
        """
        Calculates autocorrelation coefficient with lag t
        Parameters:
        ----------
        x -- np.array
        input signal
        t - int
        lag for correlation
        """
        
        return np.corrcoef(x[:-t], x[t:])[0, 1]

    

class G2Denoiser:

    """         
        Public interface:
        ----------
        
        G2Denoiser::G2Denoiser( nn_denoiser_filepath:str , scoring:Scoring , model_input_size:int = 100 )
        G2Denoiser::apply_model( g2:G2 ) -> np.array
        G2Denoiser::denoise_large_field( x:np.array ) -> ( np.array , np.array )
        G2Denoiser::remove_noise( g2_3D_arr:np.array , coarse:bool = False , step:int = 5 ) -> ( np.array , np.array )
    
    """

    
    # Public methods
    
    def __init__( self , nn_denoiser_filepath , scoring_list , model_input_size=100, device = 'cpu' ):
        
        """ nn_denoiser_filepath  = string filepath to instance of AutoEncoder_2D
            scoring               = instance of Scoring or its child class
            model_input_size      = number of pixels in 1 dimension of the model
        """
        
        self.__nn_denoiser      = self.__load_model_from_file( nn_denoiser_filepath, device)
        self.__scoring_list          = scoring_list
        self.__model_input_size = model_input_size
    
        
    def apply_model( self , g2 ):
        
        """Apply the model to a single input x.

        Parameters:
        -----------
        g2 - instance of the G2 class
        """
        
        with torch.no_grad():
            res = ( self.__nn_denoiser( g2.autoencoder_g2 ).cpu().detach().numpy().reshape( g2.shape ) )
        
        # reverse the standardization
        res = res * g2.stddev + g2.mean
        
        return res
    
    def denoise_large_field( self , x ):
    
        """
        Performs coarse denoising of a 2TCF x by down-sampling the input image
        to the model input size, applying the model, and placing
        the output of the model to corresponding pixels of the original 2TCF.

        Parameters:
        -----------
        x -array, raw 2TCF
        
        """

        large_g2  = G2.LargeG2( x , self.__model_input_size )

        n_scores = len(self.__scoring_list) # number of Scorers

        res_bl    = large_g2.get_nan_array_g2_shape() # bottom left        
        scores_bl = np.array([large_g2.get_nan_array_g2_shape()]*n_scores) 
 
        full_scores = np.array([large_g2.get_zero_array_g2_shape()]*n_scores)        

        res, scores = self.__apply_strided_denoising_to_largeg2( large_g2 , large_g2.get_framed_g2() )

        res_bl      = large_g2.fill_framed_array( res_bl , res ) #fill the values
        for j in range(len(self.__scoring_list)):        
            scores_bl[j]   = large_g2.fill_framed_array( scores_bl[j] , scores[j] )

        if large_g2.bool_more_rows_than_framesize():

            # top right
            res_tr    = large_g2.get_nan_array_g2_shape()            
            scores_tr = np.array([large_g2.get_nan_array_g2_shape()]*n_scores)  

            res = large_g2.get_framed_g2_upper_right() # subselect top right values
            res, scores = self.__apply_strided_denoising_to_largeg2( large_g2 , res )
            
            res_tr    = large_g2.fill_framed_array_upper_right( res_tr    , res ) # fil the values
            for j in range(n_scores):
                scores_tr[j] = large_g2.fill_framed_array_upper_right( scores_tr[j] , scores[j] ) 
                full_scores[j] = np.nanmean([scores_bl[j],scores_tr[j]], axis = 0)
            
            full_result = np.nanmean([res_bl,res_tr], axis = 0) 
            full_result[np.isnan(full_scores[0])] = np.nan

            return full_result, full_scores
        else:
            return res_bl, scores_bl


    def remove_noise( self , g2_3D_arr , coarse = False , step=5 ):

        """Apply denoising model to two-time correlation functions from multiple ROIs.

        Parameters:
        -----------

        g2_3D_arr - 3D array representing 2TCF. 
            Shape is (N_frames, N_frames, N_roi) 
        coarse - boolean,
            indicates whether only down-up sampling is applied (True)
            or the combination of down-up and sliding window (False)
        step - integer,
            the distance between consecutive application of sliding window.
            ignored when coarse = True

        """

        # check the shape of the input and correct if needed 
        g12b = G2.get_3d_g2_array_Nframe_x_Nframe_x_N_roi(g2_3D_arr)
        # g12b = G2.G2Collection( [ gi for gi in g2_3D_arr ] ).get_3d_g2_array_Nframe_x_Nframe_x_N_roi()
        
        N_roi = g12b.shape[-1]

        denoised_g12b = np.empty(g12b.shape)
        accuracy_score = np.empty((len(self.__scoring_list), *g12b.shape))

        if not coarse:
            w = self.__model_input_size
            sh = g12b.shape[0]
            weights = np.zeros((sh, sh))
            # calculate weights for averaging
            for j in range(w, sh + 1, step):
                new_weights = np.ones((w, w))
                weights[j - w : j, j - w : j] += new_weights
            weights[weights > 0] = 1 / weights[weights > 0]
        else:
            weights = None
        
        def process_single_roi(roi):
            return self.__remove_noise_single(g12b[:, :, roi], weights, coarse, step)

        # for j in range(N_roi):
            
        #     denoised_g12b[:, :, j], accuracy_score[:, :, j] = self.__remove_noise_single(
        #         g12b[:, :, j] , coarse , step
        #     )
        
        denoised_results = Parallel(n_jobs=-1, backend='threading')(delayed(process_single_roi)(j) for j in range(N_roi))
        

        for j in range(N_roi):            
            denoised_g12b[:, :, j], accuracy_score[:, :, :, j] = denoised_results[j]

        return denoised_g12b, accuracy_score

    
    
    # Private methods
    
    def __load_model_from_file( self , model_file, device = 'cpu'):
        
        model_params = torch.load( model_file )
        model = nets.AutoEncoder_2D( *model_params["model_init"] )
        model.load_state_dict( model_params["model_state"] )
        model.eval()
        model.to(device)
        
        return model
    
    def make_symmetric_2tcf(self, g2arr):
        N = g2arr.shape[0]
        for j in range(1, N):
            repl = np.diag(g2arr,j)
            np.fill_diagonal(g2arr[j::, ::], repl)

    def __apply_strided_denoising_to_largeg2( self , large_g2 , g2arr ):
        """Denoise a large G2 snapshot by applying the denoising model
           to strided chunks."""

        scores    =  np.array([large_g2.get_zero_array_g2_framed_shape()]*len(self.__scoring_list))

        for chunk in large_g2.make_generator_for_downsampled_g2_strides( g2arr ):
            d      = self.apply_model( \
                        G2.G2( G2.G2( chunk['downsampled_g2'] ).replace_diagonal_values_with_averaged_neighbors() ) , \
                        )
            g2arr  = large_g2.fill_array_with_strided_subarray( g2arr , d , chunk['i0_j0'] )
             

            for j in range(len(self.__scoring_list)):
                scores[j] = large_g2.fill_array_with_strided_subarray(
                             scores[j] ,
                             self.__scoring_list[j].score_denoised( chunk['downsampled_g2'] , d ) ,
                             chunk['i0_j0'] )
        
        # propagate lower half to make symmetric
        self.make_symmetric_2tcf(g2arr)
        for j in range(len(self.__scoring_list)):
            self.make_symmetric_2tcf(scores[j])

        return g2arr , scores 


    def __remove_noise_single( self , x , weights, coarse=False, step=5 ):
        
        """Apply denoising model to a two-time correlation function from a single ROI.

        Parameters:
        -----------
        x - numpy.array
            2-time correlation function, shape is (N_frames, N_frames)
        coarse - boolean,
            indicates whether only down-up sampling is applied (True)
            or the combination of down-up and sliding window (False)
        step - int, positive
            step size of the sliding window
        
        """

        x = G2.G2( x ).replace_diagonal_values_with_averaged_neighbors()

        completed       = np.zeros(x.shape)
        completed_score = np.zeros((len(self.__scoring_list), *x.shape))

        background, background_score = self.denoise_large_field( x )

        s = step
        w = self.__model_input_size
        sh = x.shape[0]

        if not coarse:
            # weights = np.zeros(x.shape)

            # # calculate weights for averaging
            # for j in range(w, sh + 1, s):
            #     new_weights = np.ones((w, w))
            #     weights[j - w : j, j - w : j] += new_weights

            # weights[weights > 0] = 1 / weights[weights > 0]

            # denoise and average results according to the weights
            for j in range(w, sh + 1, s):
                buffer = np.zeros(x.shape)
                score = np.zeros(x.shape)

                q = x[j - w : j, j - w : j]
                buffer[j - w : j, j - w : j] = self.apply_model( G2.G2(q) )

                completed += weights * buffer
                for k in range(len(self.__scoring_list)):
                    score[j - w : j, j - w : j] = self.__scoring_list[k].score_denoised(
                                                  buffer[j - w : j, j - w : j], q )
                    completed_score[k] += weights * score

        completed[completed == 0] = background[completed == 0]
        completed_score[completed_score == 0] = background_score[completed_score == 0]

        return completed, completed_score












                   

        













