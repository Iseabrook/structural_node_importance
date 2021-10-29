# Structurally aware node importance

This repository contains the code developed to support the findings to be published 
Seabrook et al., Community aware evaluation of node importance

## Usage
This repository contains four different python modules to produce the analysis shown
in Seabrook et. al., Community aware evaluation of node importance:
1. eigenvectors_and_communities.py
2. node_importance_functions.py
3. node_prediction.py
4. synthetic_networks.py

eigenvectors_and_communities.py contains the code for experiments with a barbell network,
looking at how the different components of the eigenspectrum localise to different
communities in a network.

node_importance_functions.py contains the main functions needed to compute node
importance and the metrics node importance is compared to, as well as the functions 
required for the exploratory analysis presented in the paper.

node_prediction.py contains the functions required to analyse the predictability of 
node changes from the importance of each node. 

synthetic_networks.py provides the capability to generate synthetic importance based
2 snapshot temporal networks from three different initial snapshot networks, as well
as the application of the functions within node_importance_functions.py and node_prediction.py
to these synthetic networks. 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss 
what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
