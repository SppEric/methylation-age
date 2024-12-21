# methylation-age

## Contents Overview
`data_processing`: Contains script to download the methylation data locally

`images`: Contains example generated images that will be inputs to the model. The model is trained on images without the dots, images with points are to demonstrate underlying distribution

`R_files`: Contains files given originally by the collaborator

`GenSmoothScatter.py`: Transcription of collaborator's graph generation code into Python (original can be found in `R_files/GenSmoothScatter.R`). The file performs a kernel density estimate using `scipy.stats.gaussian_kde` and generates and pickles the graph

`methylation-age-prototype`: This notebook is where everything gets put together

