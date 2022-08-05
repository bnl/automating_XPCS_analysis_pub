This repository contains the code to perform the analysis of two-time photon correlation functions (X-ray Photon Correlation Spectroscopy), presented in the article "Machine Learning for Analysis of Speckle Dynamics: Quantification and Outlier Detection"(Fig. 2, Fig. 4). The code contains the functions for noise removal from a two-time correlation function, uncertainty quantification, taking one-time correlation cuts and fitting them to a stretched exponent functional form. The repository contains a pre-trained model (kernel size 1) and corresponding uncertainty quantification data.  

  

The content of the folders is the following: 

  

* `denoising` -- analysis libraries 

* `example notebooks` -- a notebook that demonstrates the exaples of application of the library function. An example of a two-time correlation function is included. 

*  `model files` contains a pre-trained model and corresponding files. 

  

To install directly from GitHub use: 

  

`pip install git+https://github.com/bnl/automating_XPCS_analysis_pub` 

  

To install at JupyterHub (for NSLS-II users): 

  

At the beginning of each session, go to `File > New > Terminal` and type `pip install git+https://github.com/bnl/automating_XPCS_analysis_pub`. 

  

To copy the files from this repository, at your terminal, type : `git clone git+https://github.com/bnl/automating_XPCS_analysis_pub` 


The notebook, that contains examples of useful functions is `example_notebook/demonstrate_denoising_pub.ipynb`

DISCLAIMER: 
This effort is an ongoing research project that includes evaluation of the models for wide range applicability.  Upon validation of the analytical methods, a future repository will become available. We welcome any reports of issues based on the code presented here, though we may not be able to address all issues since the repositiory is designed to support the paper and not be a formal python library. 

