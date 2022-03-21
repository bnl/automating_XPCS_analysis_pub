import numpy as np
from abc import ABC , abstractmethod

import denoising.fit_parameters as FP



class OneTimeFitParameterResults_SingleROI:
    
    def __init__( self , fp , Nframe , step=1 ):
        
        """
        Class responsible for managing data 
        from the one-time parameter model fits 
        for a single ROI
        
        fp : instance of UniqueCollectionFitParameters
        Nframe : int, number of frames
        step : int, width of the one-time parameter cuts in frames
        """
        
        assert( isinstance( fp , FP.UniqueCollectionFitParameters ) )
        
        self.__Nframe = Nframe
        self.__step   = step
        self.__param_names = [ repr(ni) for ni in fp ]
        
        self.__data_dict = self.__construct_datadict_with_empty_data()
        
    @property
    def Nframe( self ):
        
        return self.__Nframe
    
    @property
    def step( self ):
        
        return self.__step

    @property
    def param_names( self ):
        
        return self.__param_names

    @property
    def corr_names( self ):
        
        return [ 'corr_' + self.param_names[i] + '_' + self.param_names[j] for i in range(self.nparams) for j in range(i+1,self.nparams) ]
    
    @property
    def nparams( self ):
        
        return len( self.param_names )

    @property
    def data_dict( self ):
        
        return self.__data_dict
            
    def get_empty_array1d( self ):
        
        return np.zeros( 2*self.Nframe // (2*self.step) )

    def get_ages( self, time_units ):
        
        N_points = 2*self.Nframe // (2*self.step)
        time_slots = np.arange(N_points)


        return time_slots * time_units + time_units/2
    
    def set_trust_regions( self , key , value ):
        
        if key in self.trust_regions.keys():
            self.trust_regions[key] = value
        else:
            raise ValueError( f'Entered key={key}, allowable keys are: {self.trust_region.keys()}' )
    
    def set_correlations( self , key , value ):
        
        if key in self.correlations.keys():
            self.correlations[key] = value
        else:
            raise ValueError( f'Entered key={key}, allowable keys are: {self.correlations.keys()}' )

    def set_datadict_attr( self , strattr , key , value ):
        
        """
        example : self.set_datadict_attr( 'trust_regions' , 'beta' , np.ones(self.Nframe) ) sets data_dict['trust_regions']['beta'] = np.ones(self.Nframe)
        """
        
        if strattr in self.data_dict.keys():
            if key in self.data_dict[strattr].keys():
                self.__data_dict[strattr][key] = value
            else:
                raise ValueError( f'Entered key={key}, allowable keys are: {self.data_dict[strattr].keys()}' )
        else:
            raise ValueError( f'Entered strattr={strattr}, allowable keys are: {self.data_dict.keys()}' )

        
    def reverse_entries_of_datadict_attr( self , strattr ):
        
        """
        example : self.set_datadict_attr( 'trust_regions' ) sets data_dict['trust_regions'][<key_k>] = arr[::-1] where arr is the data held by key_k, for all key_k
        """
        
        if strattr in self.data_dict.keys():
            for (k,key_k) in enumerate(self.data_dict[strattr]):
                self.__data_dict[strattr][key_k] = self.__data_dict[strattr][key_k][::-1]
        else:
            raise ValueError( f'Entered strattr={strattr}, allowable keys are: {self.data_dict.keys()}' )
        

    def set_datadict_attr_for_all_keys( self , strattr , idx_ar , vals ):
        
        """
        example : self.set_datadict_attr_for_all_keys( 'trust_regions' , idx_ar , val ) sets data_dict['trust_regions'][<key[k]>][idx_ar] = val[k] for k=1...Nkeys
        """
        
        if ( len(np.shape(vals)) == 0 ): # if setting a single scalar value, not a list
            vals = [ vals ]
        
        if strattr in self.data_dict.keys():
            for (k,key_k) in enumerate(self.data_dict[strattr]):
                self.__data_dict[strattr][key_k][idx_ar] = vals[k]
        else:
            raise ValueError( f'Entered strattr={strattr}, allowable keys are: {self.data_dict.keys()}' )
    
    
    # Private interface
    
    def __construct_dict_paramnames_to_array1d( self ):
        
        return dict( zip( self.param_names , [ self.get_empty_array1d() for i in range(self.nparams) ] ) )

    def __construct_correlations( self ):
                
        return dict( zip( self.corr_names, [ self.get_empty_array1d() for i in range(len(self.corr_names)) ] ) )

    def __construct_datadict_with_empty_data( self ):
        
        # each key maps to a dict
        
        return {
            'parameters'        : self.__construct_dict_paramnames_to_array1d() , 
            'parameters_errors' : self.__construct_dict_paramnames_to_array1d() , 
            'trust_regions'     : self.__construct_dict_paramnames_to_array1d() , 
            'correlations'      : self.__construct_correlations() , 
            'fit_quality'       : {'r2_fit' : self.get_empty_array1d() }, # this redundant dict is used for consistency with other values
            
            'parameters_no_tail'        : self.__construct_dict_paramnames_to_array1d() , 
            'parameters_errors_no_tail' : self.__construct_dict_paramnames_to_array1d() , 
            'trust_regions_no_tail'     : self.__construct_dict_paramnames_to_array1d() , 
            'correlations_no_tail'      : self.__construct_correlations() , 
            'fit_quality_no_tail'       : {'r2_fit' : self.get_empty_array1d() }, # this redundant dict is used for consistency with other values
           
            'ages'         : {'bin_centers' : self.get_ages(self.step),
                              'bin_width' : self.step} 
        }
        






class OneTimeFitParameterResults_MultipleROI:
    
    def __init__( self , list_fp , Nframe , steps ):
        
        """
        Class responsible for managing data 
        from the one-time parameter model fits 
        for multiple ROIs
        
        list_fp : list[ UniqueCollectionFitParameters ] of length Nroi
        Nframe : int,
            number of frames
        steps : list of int,
            widths of the one-time parameter cuts in frames
        """
        
        assert( np.all( [ isinstance( fp , FP.UniqueCollectionFitParameters ) for fp in list_fp ] ) )
        
        self.__Nframe = Nframe
        self.__steps   = steps
        self.__param_names = [ [ repr(ni) for ni in fp ] for fp in list_fp ]
        self.__list_fp = list_fp
        
        self.list_singleroi_obj = self.__initialize_list_SingleROI_obj( self.__list_fp )
    
    @property
    def Nframe( self ):
        
        return self.__Nframe

    @property
    def Nroi( self ):

        return len( self.param_names )
    
    @property
    def steps( self ):
        
        return self.__steps
    
    @property
    def param_names( self ):
        
        return self.__param_names

    @property
    def roi_fields( self ):
        
        return self.list_singleroi_obj[0].data_dict.keys()
    
    def get_data_dict( self ):
        
        """
        returns : list[ dict ] containing Nroi dicts
        """
        
        return [ obj.get_data_dict() for obj in self.list_singleroi_obj ]
    
    def get_list_of_attr_forall_rois( self , strattr ):
        
        return [ roi.data_dict[strattr] for roi in self.list_singleroi_obj ]

    def get_nesteddict_roi_field_paramkey( self , roi_list):

        """
        returns : nested dict containing all info for all rois, structured by dict[<roi>][<field>][<paramkey>]
        example : dict_out[3]['parameters']['beta'] returns the 1TCF fit for parameter beta in ROI #3
        """

        return { int(roi_list[i]) : self.list_singleroi_obj[i].data_dict for i in range(self.Nroi) }
            
    def set_roi_dict_attr( self , idx_roi , strattr_roi , strattr_2 , valattr ):
        
        """
        example : self.set_roi_dict_attr( 1 , 'trust_regions' , 'gamma1' , np.ones( rois.Nframe ) )
        """
        
        self.list_singleroi_obj[idx_roi].set_datadict_attr( strattr_roi , strattr_2 , valattr )
    
    def set_SingleROI_data( self , fp , idx_roi ):
        
        """
        fp : instance of UniqueCollectionFitParameters
        idx_roi : int , index of ROI
        """
        
        assert( idx_roi < self.Nroi )
        
        self.list_singleroi_obj[idx_roi] = OneTimeFitParameterResults_SingleROI( fp , self.Nframe , self.steps[idx_roi] )
        
    
    # Private interface
    
    def __initialize_list_SingleROI_obj( self , list_fp ):
        
        """
        returns : list[ OneTimeFitParameterResults_SingleROI ] containing Nroi elements
        """
        
        return [ OneTimeFitParameterResults_SingleROI( fp , self.Nframe , self.steps[ind] ) for ind, fp in enumerate(list_fp) ]


if __name__ == '__main__':

    list_params = []
    
    list_params.append( FP.Stretched_Exponent_Parameters() )
    list_params.append( FP.Double_Stretched_Exponent_Parameters() )
    
    rois = OneTimeFitParameterResults_MultipleROI( list_params , 16 )
    
    print( '\n' + repr(rois) + ' object manages list of : \n' )
    
    for roi in rois.list_singleroi_obj:
        
        print( repr(roi) )
        print( 'parameter names : ' )
        print( roi.param_names )
        
        print( 'data dict : ' )
        print( roi.data_dict.keys() )
        
        print( '\n' )
    
    print( 'list of trust_regions for all ROIs : ' )
    
    for el in rois.get_list_of_attr_forall_rois( 'trust_regions' ):
        
        print( el )
    
    print( 'setting parameter gamma1 in trust_regions of roi 2 : ' )
    rois.set_roi_dict_attr( 1 , 'trust_regions' , 'gamma1' , np.ones( rois.Nframe ) )
    print( rois.list_singleroi_obj[1].data_dict['trust_regions'] )

    print( 'rois contains roi_fields : ' )
    print( rois.roi_fields )