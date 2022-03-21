from asyncore import file_dispatcher
import numpy as np
from abc import ABC , abstractmethod
from denoising.file_handling import write_to_file, load_from_file


def set_start_value_default( lower , upper , start_spec=None ):
    
    if ( start_spec is None ) or ( np.isnan(start_spec) ):
        return 0.5 * ( lower + upper )
    return start_spec
     
def set_start_value_gamma( lower , upper , start_spec=None ):
    
    if ( start_spec is not None) and ( np.isnan(start_spec) ):
        return None
    return start_spec


def set_start_value_ginf( lower , upper , start_spec=None ):
    
    if ( start_spec is None ) or ( np.isnan(start_spec) ):
        return lower + 1e-9
    return start_spec




class FitParameter:
    
    def __init__( self , lower , upper , func_start_value=set_start_value_default , start_spec=None , str_repr='fit_parameter' ):
        
        assert( isinstance( lower , (float,int) ) & isinstance( upper , (float,int) ) & ( lower <= upper ) )
        
        self.__lower = lower
        self.__upper = upper
        self.__repr = str_repr
        
        # start_value and setter function are exposed publicly for setting/getting
        self.set_start_value = func_start_value   # Function f( start_spec : float/None/np.nan ) -> float/None/np.nan
        self.start_value = self.set_start_value( lower , upper , start_spec )
        
    @property
    def lower( self ):
        
        return self.__lower
    
    @property
    def upper( self ):
        
        return self.__upper
        
    @property
    def limits( self ):
        
        return [ self.lower , self.upper ]


    def set_lower(self, value):
        self.__lower = value
        if value >= self.upper:
            print('lower bound cannot be larger than upper bound')
            self.__lower = self.upper - 1e-4
        if self.start_value <= self.__lower:
            self.start_value = 0.5*(self.lower + self.upper)

    def set_upper(self, value):
        self.__upper = value        
        if value <= self.lower:
            print('lower bound cannot be larger than upper bound')
            self.__upper = self.lower + 1e-4
        if self.start_value > self.upper:
            self.start_value = 0.5*(self.lower + self.upper)

    def set_initial_value(self, value):
        if self.upper > value and self.lower< value:
            self.start_value = value
        
    def set_repr( self , strrepr ):
        
        self.__repr = strrepr
        
    def __repr__( self ):
        
        return self.__repr





def factory_make_fit_parameter( strparam , lower=None , upper=None , start_spec=None ):
    
    def p_ctor_args( l , u , f_start ):
        
        return { 'lower'      : l , \
                 'upper'      : u , \
                 'f_start'    : f_start , \
                 'start_spec' : start_spec , \
                 'str_repr'   : strparam }
    
    def limit( val_input , val_default ):
        
        if val_input is None:
            return val_default
        return val_input
    
    limits = lambda low,high : ( limit( lower , low ) , limit( upper , high ) )
    
    factories = {
        'beta'    : ( *limits( 0.0  , 1.0 )  , set_start_value_default ) ,
        'gamma'   : ( *limits( 0.0  , 2.0 )  , set_start_value_gamma ) ,
        'ginf'    : ( *limits( 0.99 , 5.0 )  , set_start_value_ginf ) ,
        'alpha'   : ( *limits( 0.2  , 3.0 )  , set_start_value_default ) ,
    }
    
    if strparam in factories:
        
        return FitParameter( *p_ctor_args( *factories[strparam] ).values() ) 

    

class UniqueCollectionFitParameters( ABC ):
    
    """
    Base class for unique collections of fit parameters
    Child class must define set of required parameters in its ctor
    User must provide one and only one instance of each required parameter
    """
    
    @abstractmethod
    def __init__( self , params ):
        
        """
        params : list( FitParameters ) or None
        self._required_params has to be defined in the child ctor
        """
        
        assert( isinstance( params , list ) | ( params is None ) )

        self._required_params = None
        
    def __iter__( self ):
        
        self._n = 0
        return self
    
    def __next__(self):
        
        if self._n < len( self._params ):
            self._n += 1
            return self._params[self._n-1]
        else:
            raise StopIteration
    
    def get_param_bounds( self ):
        
        return list(zip( *[ pi.limits for pi in self._params ] ) )
        
    def get_initial_values( self ):
        
        return [ pi.start_value for pi in self._params ]
        
    def set_params( self , params ):

        """
        params : list( FitParameter )
        """
        
        for pi in params:
            if repr(pi) in self._required_params:
                setattr( self , repr(pi) , pi )
            else:
                raise ValueError( f'You must enter only parameter types in {self._required_params}' )
        
        if len( set( [ repr(pi) for pi in params ] ) ) != len( self._required_params ):
            raise ValueError( f'Please enter one parameter type for each type in {self._required_params}' )
        
        # Order things according to self._required_params
        actual_order = [ repr(pi) for pi in params ]
        idx = np.squeeze( [ [ j for j in range(len(actual_order)) if self._required_params[i] == actual_order[j] ] for i in range(len(self._required_params)) ] )
        
        self._params = [ params[i] for i in idx ]

    @property
    def num_params( self ):
        
        return len( self._required_params )
    
    @property
    def corr_names( self ):
        
        return [ 'corr_' + self._required_params[i] + '_' + self._required_params[j] for i in range(self.num_params) for j in range(i+1,self.num_params) ]

        
    

class Stretched_Exponent_Parameters( UniqueCollectionFitParameters ):
    
    """
    User must provide one and only one instance of [ 'beta' , 'gamma' , 'ginf' , 'alpha' ]
    """
    
    def __init__( self , params=None ):
        
        super().__init__( params )
        
        self._required_params = [ 'beta' , 'gamma' , 'ginf' , 'alpha' ]
        
        if params is None:
            params = [ factory_make_fit_parameter(pi) for pi in self._required_params ]
            
        self.set_params( params )

    @property
    def num_exponents( self ):
        
        return 1
    
    def __repr__( self ):

        return 'Stretched_Exponent_Parameters'
        


def factory_make_UniqueCollectionFitParameters( strmodel , list_lower=None , list_upper=None , list_start_spec=None ):
    
    """
    Usage assumes that the input lists are ordered according to the order in factories[strmodel]()._required_params
    """
    
    factories = {
        'Stretched_Exponent_Parameters'        : Stretched_Exponent_Parameters ,
    }

    def default_list( specific_list ):
        
        if specific_list is None:
            return [ None for i in factories[strmodel]()._required_params ]
        return specific_list
    
    if strmodel in factories:        
        
        return factories[strmodel]( [ factory_make_fit_parameter( factories[strmodel]()._required_params[i] , \
                                                                  lower=default_list( list_lower )[i] , \
                                                                  upper=default_list( list_upper )[i] , \
                                                                  start_spec=default_list( list_start_spec )[i] )
                                      for i in range(len( default_list( list_lower ) )) ] )
    
    return ValueError( f'entered strmodel unknown to factory' )

def get_trust_bounds_from_file(filename):
    """
    Exctract values of upper and lower limits for trust_region calculation from a file.
    """
    fitting_parameters = load_from_file(filename)[0]
    return fitting_parameters['lower_bound_frames'], fitting_parameters['upper_bound_half_decays']

def make_stretched_exponent_parameter_list_from_file(filename, null_gamma_start = True):
    """
    Constructs the fit parameters from a log file.
    
    Parameters:
    -----------
    filename - string
        name of the log file with parameters
    null_gamma_start - boolen
        if True, makes the initial value of Gamma None. In this case, the initial value will be established
        from fitting 1TCF with maximum available number of delay frames
        
    Returns:
    --------
    fit_param_list - list of instances of Stretched_Exponent_Parameters
    """
    
    fit_param_list = []
    fitting_parameters = load_from_file(filename)
    n_roi = len(fitting_parameters)
    for j in range(n_roi):
        fit_param_list.append(
            __make_stretched_exponent_parameter_from_log_dict(fitting_parameters[j], null_gamma_start))
    return fit_param_list
        
def __make_stretched_exponent_parameter_from_log_dict(d, null_gamma_start = True):
    """
    Constructs a single Stretched_Exponent_Parameters instance from a dictionary of bounds
    and initial values.
    
    Parameters:
    -----------
    d - dictionary
        contained in log files
    null_gamma_start - boolen
        if True, makes the initial value of Gamma None. In this case, the initial value will be established
        from fitting 1TCF with maximum available number of delay frames
    """
    
    if null_gamma_start:
        gamma_start_spec = None
    else:
        gamma_start_spec = d['gamma_start']
        
    stretched_exp_params = Stretched_Exponent_Parameters(
    [ factory_make_fit_parameter( 'beta'  , lower=d['beta_limits'][0], upper=d['beta_limits'][1], start_spec = d['beta_start'] ) ,
      factory_make_fit_parameter( 'gamma' , lower=d['gamma_limits'][0], upper=d['gamma_limits'][1], start_spec = gamma_start_spec ) ,
      factory_make_fit_parameter( 'ginf'  , lower=d['ginf_limits'][0], upper=d['ginf_limits'][1], start_spec=d['ginf_start'] ) ,
      factory_make_fit_parameter( 'alpha' , lower=d['alpha_limits'][0], upper=d['alpha_limits'][1], start_spec=d['alpha_start'] ) ] 
    )
    return stretched_exp_params
    
    

if __name__ == '__main__':

    # Use the factory to instantiate various parameters
    beta  = factory_make_fit_parameter( 'beta' )
    gamma = factory_make_fit_parameter( 'gamma' )
    ginf  = factory_make_fit_parameter( 'ginf' )
    alpha = factory_make_fit_parameter( 'alpha' )

    # gamma1 = factory_make_fit_parameter( 'gamma1' )
    # gamma2 = factory_make_fit_parameter( 'gamma2' )
    # alpha1 = factory_make_fit_parameter( 'alpha1' )
    # alpha2 = factory_make_fit_parameter( 'alpha2' )
    # pstart = factory_make_fit_parameter( 'p_start' )
        
    # Order does not matter in the ctor of UniqueCollectionFitParameters
    # You just have to input a list containing exactly one instance of each required parameter type
    params  = Stretched_Exponent_Parameters( [ beta , gamma , ginf , alpha ] )
    params2 = Stretched_Exponent_Parameters( [ alpha , beta , gamma , ginf ] )
    # dparams = Double_Stretched_Exponent_Parameters( [ alpha1 , alpha2 , beta , gamma1 , gamma2 , pstart , ginf ] )
    
    p = [ params , params2 ]#, dparams ]
    
    for pi in p:
        print( repr(pi) )
        print( 'parameters held : ' )
        print( [ repr(pij) for pij in pi ] )
        print( 'accessing beta directly : ' )
        print( pi.beta , pi.beta.limits , pi.beta.start_value )
        
        # You can also iterate through the params object if you want, like this:
        print( 'accesing all parameters by iteration : ' )
        for pij in pi:
            print( pij , pij.limits , pij.start_value )
        
        # Other methods:
        print( 'parameter bounds : ' )
        print( pi.get_param_bounds() )
        print( 'parameter initial values : ' )
        print( pi.get_initial_values() )
        
        print( '\n' )

