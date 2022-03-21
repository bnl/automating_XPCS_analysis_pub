The wrapper functions for denoising autoencoder model for two-time correlation function. 
Model weights and training errors (for estimating the accuracy of the model) are in the compressed archive  `model_files.tar.xz` that needs
to be de-compressed prior to use.
For internal use at NSLS-II servers, all model files are copied to 
`/nsls2/data/projects/ldrd_20_xpcs_ml/model_files`

The demonstration notebook and a single 2TCF are included in the folder `example notebook`.

To install directly from GitHub use:

`pip install git+https://github.com/ML-XPCS-BNL/denoising_model.git`

To install at JupyterHub:


At the beginning of each session, go to `File > New > Terminal` and type `pip install git+https://github.com/ML-XPCS-BNL/denoising_model.git`. It will ask for your GitHub credentials. Enter your username and password when asked.

To copy the files from this repository, at your terminal, type : `git clone https://github.com/ML-XPCS-BNL/denoising_model.git`

The notebook, that contains examples of useful functions and has the correct links to all model-related files is `example_notebook/demonstrate_denoising_works_at_JupyterHub.ipynb`

