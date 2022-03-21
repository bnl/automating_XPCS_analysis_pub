import numpy as np
import torch
import denoising.nets as nets
import denoising.g2 as G2
from scipy.stats import gaussian_kde


class LatentSpaceExtractor:

    """         
        Public interface:
        ----------
        
        
    
    """

    
    # Public methods
    
    def __init__( self , autoencoder_filepath, latent_space_scaler_file, distance_ecdf_file, distance_median_file):
        
        """ 
        """
        
        self.__autoencoder      = self.__load_model_from_file( autoencoder_filepath )
        self.__scaler_params    = torch.load(latent_space_scaler_file) # load stdev and mean for the training set from file
        self.__distance_ecdf = torch.load(distance_ecdf_file)# empirical cdf of the of the deviation of latent space representations from their mean for the training set.
        self.__distance_median = torch.load(distance_median_file)#load from file
         
        
        
    def get_latent_space( self , g2 ):
        
        """Apply the model to a single input x.

        Parameters:
        -----------
        g2 - instance of the G2 class
        """
        
        res = self.__autoencoder.get_latent_space_coordinates( g2.autoencoder_g2 ).cpu().detach().numpy().flatten()
        
        return res
    
    def apply_scaler(self, lat_coord):

        centered_lat_coord = lat_coord - self.__scaler_params['mean']
        normalized_lat_coord = centered_lat_coord/np.sqrt(self.__scaler_params['var'])

        return normalized_lat_coord
    
    def calculate_distance_to_training_center( self , lat_coord):
        
        scaled_lat_coord = self.apply_scaler(lat_coord)
        distance = np.linalg.norm(scaled_lat_coord)

        return distance

    def calc_prob_of_distance_too_big(self, x):
        """
        Calculates the probability of a training example having smaller 
        distance to center of the lattent space than x.
        """
                
        prob = self.__distance_ecdf(x)

        return prob

    def evaluate_relative_distance(self, x):
        """
        Compares the  distance of the latent vector for the input
        with the median distance among the training set.
        """
        relative_distance = x/self.__distance_median

        return relative_distance
    
    
    # Private methods
    
    def __load_model_from_file( self , model_file ):
        
        model_params = torch.load( model_file )
        model = nets.AutoEncoder_2D( *model_params["model_init"] )
        model.load_state_dict( model_params["model_state"] )
        model.eval()
        
        return model




    

    

   












                   

        













