# nasa-cosmic-dust-classification
Script to generate graphs and plots associated with the paper "30 years of NASA Cosmic Dust Catalog: Insights into the Impact of Space Activities on the Earth's Stratospheric Particle Content" by Taupin et al.

This script is in a jupyter notebook, allowing you to check the plots I originally got from my script and used in the associated paper.
To run it on your machine, please follow these steps :
1) Create a new python virtual environment in your terminal : python -m venv name_your_virtual_environment
2) Activate your newly created virtual environment : source name_your_virtual_environment/bin/activate (N.B.: "/" must be replaced by antislash on Windows)
3) Install the required dependencies : pip install -r requirements.txt

All the digitized spectra and codes for creating the 2D maps for classification and clustering are available in a Zenodo archive accessible here : 10.5281/zenodo.22256697
The original PDF files and their full preprocessing are available in the folder "Data_all" (14 GB)
The digitized version of the spectra (CSV files) after preprocessing are available in the folder "Data_csv" (2 GB)

The data can be plotted in three different modes :
1) NASA preliminary classification (types TCA, AOS, TCN, C, NA) with particle size : to check the consistency between NASA preliminary classification and our classification exclusively based on chemical composition from EDS spectra
2) Clusters determined by K-Means
3) Interactive plot to obtain the name of each particle and its catalog number

An option was added to display the data for some selected years and allows to assess the population evolution of the analyzed stratospheric particles.
Finally, the user can choose to display the EDS spectra and associated BSE images of the particles closest to the barycenter of a selected cluster.
