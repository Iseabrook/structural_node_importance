# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 07:48:32 2021

@author: iseabrook1

"""
#This script contains the code required to run the analysis of inter-country 
#trades in financial services, to produce the results shown in Seabrook et. al., 
#Community aware evaluation of node importance.

#Isobel Seabrook, ucabeas@ucl.ac.uk
#MIT License. Please reference below publication if used for research purposes.
#Reference: Seabrook et al., Community aware evaluation of node importance
#
################################################################################
#   Instructions for use. First, you will need to download the dataset from 
#   https://stats.oecd.org/Index.aspx?DataSetCode=BATIS_EBOPS2010, and save it 
#   in your chosen location and update the path_to_data below. Following this,
#   the code can be run to reproduce the results presented in the Supplementary
#   information of Seabrook et. al.  
#   
#   In contrast to the analysis presented in  Seabrook et al., the dataset used 
#   here has a natural persistence of activity (generally, countries that trade 
#   with each other continue to do so year on year) so instead of looking to predict
#   whether or not a node will be present in the subsequent snapshot, we instead 
#   look to predict whether or not a node will change.

import node_importance_functions as nif
import node_prediction as pred
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
import seaborn as sns

if __name__ == "__main__":
    
    path_to_data = 'C:/Users/iseabrook1/OneDrive - Financial Conduct Authority/Network_analytics/PhD/Data/'
    raw_data = pd.read_csv(path_to_data+'wto_data.csv')
    
    #select only financial services (SG), inter-country only (rather than 
    #partner-world trades), imports only to avoid double counting
    raw_data=raw_data[(raw_data.type_Partner=='c')&(raw_data.Final_value>0)&(raw_data.Flow=='IMP')&(raw_data.Item_code=='SG')]
    raw_data=raw_data[['Reporter', 'Partner', 'Year', 'Final_value']]
    
    raw_data.columns = ["buyer id", "seller id", "trade date time", "total_value"]
    print(raw_data.shape)
    
    #Calculate feature dataset containing node importance/other node level measures
    #along with the relative subsequent change to node strengh. 
    ds_mi = nif.da_le_pairs_test(raw_data.sort_values(by='trade date time'), test_type='all', weight=False)
    ds_mi.to_csv(path_to_data+'ds_mi_wto.csv')
#    ds_mi = pd.read_csv(path_to_data+'ds_mi_wto.csv')
    ds_mi.replace([np.inf, -np.inf], np.nan)
    ds_mi.fillna(0, inplace=True)

    ds_mi['change_bool']= np.multiply(abs(ds_mi.S_fin)>0,1)
    
    
    print(ds_mi.change_bool.value_counts())
    reload(nif)
    nif.plot_violinplots_multimeasure(ds_mi)
    
    #Produce a correlation heatmap for the different features
    corr = ds_mi[['m_a' ,'m_b','m_c','m_d','degree', 'pagerank','eig_cent']].fillna(0).corr()    
    plt.subplots(figsize=(10,10))
    sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns, annot=True, annot_kws={"fontsize":10})
    plt.show()
    X = ds_mi[[ "m_a", "m_b", "m_c", "m_d", 'eig_cent']].fillna(0)
    y = ds_mi['change_bool']
    
    #run the node change prediction 

    pred.node_change_prediction(ds_mi, X, y)
    
    #plot network with the nodes coloured by their eigenvalue ranking
    nif.plot_network_ev_rankings(raw_data, ds_mi)

    #plot the modularity of the network across time
    modularities = nif.modularity_function(raw_data)
    fig, ax=plt.subplots(figsize=(15,10))
    ax.plot([x for x in modularities.index],modularities.values)
    ax.set_xlabel('Trade execution date')
    ax.set_ylabel('Modularity')
   
    plt.show()
    
    plt.figure()
