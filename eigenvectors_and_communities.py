# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 14:57:46 2021

@author: iseabrook1
"""

#This script contains the code required to run experiments to understand how different 
#components of a network's eigenspectra are relevant to different features in the network, for a simple barbell network.

#Isobel Seabrook, ucabeas@ucl.ac.uk
#MIT License. Please reference below publication if used for research purposes.
#Reference: Seabrook et al., Community aware evaluation of node importance
#
################################################################################
#   Instructions for use.
#

#   This script initially produces a dataframe of the eigenvector components
#   indexed by the individual nodes in a barbell network with two cliques, one
#   with 5 nodes and the other with 4, joined by a bridge with two nodes. This
#   dataframe is then used as input in a k-means clustering to cluster the nodes 
#   the network, and the network is then plotted with the nodes coloured by the 
#   result of the k-means.
###############################################################################

from sklearn.cluster import KMeans
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

#Assessing the different eigenvector scales for the different communities
#Unequal size communities:
def barbell_network():
    """ Function to generate an unweighted barbell network with clique sizes 4 
        and 5, bridge size 2. 
        
        Parameters:
            none
        
        Returns:
            G_comm: networkx graph object of barbell network
    """
    G_comm = nx.barbell_graph(4, 2)
    #adding in additional node to make one cluster larger than the other
    G_comm.add_node(10)
    G_comm.add_edge(10, 9)
    G_comm.add_edge(10, 8)
    G_comm.add_edge(10, 7)
    G_comm.add_edge(10, 6)
    
    for (u,v,w) in G_comm.edges(data=True):
            w['weight'] = 1
    return(G_comm)


if __name__ == "__main__":
    
    G_comm = barbell_network()
    A = nx.to_pandas_adjacency(G_comm,weight='weight')
    #generate the eigenvector -eigenvalue pairs, create dataframe of eigenvectors
    eigenvalues, eigvecs = np.linalg.eig(A) 
    eigvecs_df = pd.DataFrame(eigvecs)
    eigvecs_df.columns=eigenvalues
    eigvecs_df.index = A.columns
    
    #running a k-means on the different components to identify the communities
    X = np.array(eigvecs_df.iloc[:,0:3])
    #We're expecting 3 clusters - 2 cliques, 1 bridge
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    k_means_predicted = kmeans.predict(X)
    
    #plot network with predicted clusters coloured

    nx.draw(G_comm, with_labels=False, node_color = k_means_predicted)
    plt.show()


