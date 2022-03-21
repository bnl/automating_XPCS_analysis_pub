import numpy as np
import torch

class G2( ):
    
    """Class to handle a single g2 snapshot (a 2TCF for a single ROI)"""
    
    def __init__( self , g2 ):
                
        self.__g2 = g2

    @property
    def g2(self):
        
        return self.__g2.copy()

    @g2.setter
    def g2( self , g2 ):
        """Type checking etc for valid g2"""
        
        assert( isinstance( g2 , np.ndarray ) & ( len( g2.shape ) == 2 ) )
        self.__g2 = g2
        
    @property
    def stddev( self ):
        
        return np.std( self.g2 )

    @property
    def mean( self ):
        
        return np.mean( self.g2 )
    
    @property
    def scaled_g2( self ):
        """Scale g2 to have zero mean, unit variance"""
        
        return ( self.g2 - self.mean ) / self.stddev

    @property
    def autoencoder_g2( self ):
        """Scale and reshape g2 for input to the autoencoder"""
        
        return torch.from_numpy( self.scaled_g2.reshape( -1 , 1 , *self.shape ) ).float()

    @property
    def shape( self ):
        
        return self.g2.shape

    def replace_diagonal_values_with_averaged_neighbors( self ):
        """Center-average diagonal g2 array values and return a copy of the result"""
        
        x = self.g2
        
        for j in range(x.shape[0]-1):
            x[j,j] = 0.5 * ( x[j-1,j] + x[j+1,j] )
        x[0,0] = x[1,1]
        x[-1,-1] = x[-2,-2]
        
        return x
    
    
class LargeG2(G2):
    
    """Subclass meant to handle a g2 that is specifically larger than what the autoencoder input can handle"""
    
    def __init__( self , g2 , model_input_size ):
        
        super().__init__( g2 )
        
        self.model_input_size = model_input_size
        assert( (self.g2.shape[0] >= model_input_size) & (self.g2.shape[1] >= model_input_size) )
    
    @property
    def folds( self ):
        """ How many times the size of the 2TCF is larger than the input of the autoencoder"""
        
        return self.g2.shape[0] // self.model_input_size
    
    @property
    def N_lim( self ):
        """ Maximum number of frames that can be processed my the model without a loss of information.
        If the total number of frames is not dividable by the model input size, N_lim
        is smaller than the number of frames."""
        return self.model_input_size * self.folds
    
    def _get_framed_array( self , arr ):
        """
        Select the portion of array up to N_lim (from [0,0] up)
        """
        return arr[ :self.N_lim , :self.N_lim ].copy()
    
    def _get_framed_array_upper_right( self , arr ):
        """
        Select the portion of array up to N_lim (from [-1,-1] down)
        """
        return arr[ -self.N_lim: , -self.N_lim: ].copy()
    
    def get_framed_g2( self ):
        """
        Select the portion of the 2TCF that can be fully processed by
        the denoising autoencoder (including the lower left corner (t=0))
        """
        return self._get_framed_array( self.g2 )
    
    def get_framed_g2_upper_right( self ):
        """
        Select the portion of the 2TCF that can be fully processed by
        the denoising autoencoder (including the upper right corner (t=0))
        """
        return self._get_framed_array_upper_right( self.g2 )
    
    def fill_framed_array( self , array , filler ):
        """
        Assign new values to lower left corner of the array
        """
        array[ :self.N_lim, :self.N_lim ] = filler
        
        return array
    
    def fill_framed_array_upper_right( self , array , filler ):
        """
        Assign new values to upper right corner of the array
        """
        array[ -self.N_lim:, -self.N_lim: ] = filler
        
        return array
    
    def get_strided_array( self , array , i0 , j0 ):
        """
        Down-sample the 2TCF by selecting every N-th frame,
        where is N is the the number of folds (see 'folds' property)
        """
        return array[ i0:self.N_lim:self.folds , j0:self.N_lim:self.folds ].copy()

    def make_symmetric_with_lower_bottom(self, array):
        N = array.shape[0]
        for j in range(1, N):
            repl = np.diag(array,j)
            np.fill_diagonal(array[j::, ::], repl)
            return array

    
    def make_generator_for_downsampled_g2_strides( self , arr_g2 ):
        """Return a generator that contains:
           dict( [ 'i0_j0' , 'chunk' ] )
           chunk is a G2 subarray starting at (i0,j0), 
           formed by striding 2D array=arr_g2 to N_lims by stride=folds"""
                
        for k in range(self.folds):
            for p in range(self.folds):
                yield dict( zip( ['i0_j0' , 'downsampled_g2'] ,
                                 [ (k,p)  , 
                                 self.make_symmetric_with_lower_bottom(
                                        self.get_strided_array( arr_g2 , k , p ))]  ))
    
    def fill_array_with_strided_subarray( self , array , subarray , i0_j0 ):
        """
        Assign new values to an array for each N-th element in both rows and columns,
        starting with (i,j) = i0_j0. N is the number of folds (see 'folds' property).
        """
        array[ i0_j0[0]:self.N_lim:self.folds , i0_j0[1]:self.N_lim:self.folds ] = subarray[:, :]
        
        return array

    def get_zero_array_g2_shape( self ):
        
        return np.zeros( self.shape )

    def get_nan_array_g2_shape( self ):
        tmp = np.empty( self.shape )
        tmp[:, :] = np.nan        
        return tmp

    def get_zero_array_g2_framed_shape( self ):

        return np.zeros( [ self.N_lim , self.N_lim ] )


    def bool_more_rows_than_framesize( self ):
        """
        Checks if there are more frames available than
        can be processed by the model without loss of information.
        """
        
        return ( self.shape[0] - self.N_lim > 0 )


def get_3d_g2_array_Nframe_x_Nframe_x_N_roi( g12):
    """
    Returns copy of g2_list as a 3d numpy array
    with shape (N_frames, N_frames, N_roi)        
    """
    
    ax1, ax2, ax3 = g12.shape
    if ax1 == ax2:
        return g12
    elif ax2 == ax3:
        return np.swapaxes(g12,0,2)
    else:
        print('Wrong shape of an input array')
        return None





def test_g2( ):
    
    x = np.random.uniform( 0 , 1 , [3,3] )    
    valid = G2( x )        
    print( valid.g2 )
    print( valid.scaled_g2 )
    print( valid.autoencoder_g2 )
    print( valid.shape )

 
        
if __name__ == '__main__':
    test_g2()
