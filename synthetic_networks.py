# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 14:57:46 2021

@author: iseabrook1
"""

#This script contains the code required to generate static networks, from which 
#two-snapshot importance based temporal networks are then generated. 
#It also includes the subsequent analysis of the synthetic networks to produce 
#the results presented in Seabrook et. al., Community aware evaluation of node importance

#Isobel Seabrook, ucabeas@ucl.ac.uk
#MIT License. Please reference below publication if used for research purposes.
#Reference: Seabrook et al., Community aware evaluation of node importance
#
################################################################################
#   Instructions for use.
#   The functions static_hub_gen, static_bb_gen and static_er_gen can all be used 
#   to produce static networks in the form of pandas edgelist dataframes.
#
#   The static networks can then be passed to netstats_generator, which generates
#   a boolean indicator of change for each node based on its importance, and then 
#   calculates the node level metrics m_{a-c}, eigenvector centrality, pagerank,
#   degree, and community label. This can then be used as input in various functions
#   in the modules 'node_importance_functions.py' and 'node_prediction.py' to 
#   assess how these networks evolve over time and how node importance and other 
#   node level metrics relate to the network evolution. 

import networkx as nx
import node_importance_functions as nif
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import node_prediction as pred

def static_hub_gen(n,m,y, timestamp):
    """ Function to produce an unweighted static Multi-star graph, with 6 stars 
    of varying sizes, some of which are joined together The user can decide on the 
    sizes of the stars (three of which are equal size). 
    Parameters:
        n,m,y: number of nodes to make up the cliques
        timestamp: if generating a static graph as a time snapshot instance of 
        a temporal graph, specify the timestamp for the snapshot.

    Returns:
        pandas edgelist dataframe. 
    """
    G_1 = nx.star_graph(n)
    G_2 = nx.star_graph(m)
    G_3 = nx.star_graph(y)
    G_4 = nx.star_graph(y)
    G_5 = nx.star_graph(y)
    G_6 = nx.star_graph(y)

    G = nx.union(G_1, G_2, rename=("n", "m"))
    G = nx.union(G, G_3, rename=("", "y"))
    G = nx.union(G, G_4, rename=("", "a"))
    G = nx.union(G, G_5, rename=("", "b"))
    G = nx.union(G, G_6, rename=("", "c"))

    G.add_edge('n0', 'm0')
    G.add_nodes_from(['x1', 'x2', 'x3', 'x4'])
    G.add_edge('n3', 'x1')
    G.add_edge('x2', 'x1')
    G.add_edge('x2', 'x3')
    G.add_edge('x3', 'x4')
    G.add_edge('x0', 'y0')

    G.add_nodes_from(['p1', 'p2', 'p3', 'p4'])
    G.add_edge('c3', 'p1')
    G.add_edge('p2', 'p1')
    G.add_edge('p2', 'p3')
    G.add_edge('p3', 'p4')
    for (u,v,w) in G.edges(data=True):
        w['weight'] = 1
    graph_data = nx.to_pandas_edgelist(G, source="seller id", target="buyer id")
    graph_data.columns = ["seller id","buyer id", "total_value"]
    graph_data["tuple_id"] = list(G.edges())
    graph_data['trade date time'] = timestamp
    return(graph_data)
    
def static_er_gen(n, timestamp):
    """ Function to produce a static unwieghted erdos renyi graph, with n nodes 
    and connection probability 0.5.  
    Parameters:
        n: number of nodes 
        timestamp: if generating a static graph as a time snapshot instance of 
        a temporal graph, specify the timestamp for the snapshot.
        
    Returns:
        pandas edgelist dataframe. 
    """
   
    G = nx.erdos_renyi_graph(n, 0.05)
    
    for (u,v,w) in G.edges(data=True):
        w['weight'] = 1
    G=G.to_undirected()
    graph_data = nx.to_pandas_edgelist(G, source="seller id", target="buyer id")
    graph_data.columns = ["seller id","buyer id", "total_value"]
    graph_data["tuple_id"] = list(G.edges())
    graph_data['trade date time'] = timestamp
    return(graph_data)
    
def static_bb_gen(n,m, timestamp):
    """ Function to produce an unweighted static barbell graph, with two nodes 
    in the bridge and n nodes in the cliques. 
    Parameters:
        n: number of nodes to make up the clique
        timestamp: if generating a static graph as a time snapshot instance of 
        a temporal graph, specify the timestamp for the snapshot.
    Returns:
        pandas edgelist dataframe. 
    """
    G = nx.barbell_graph(n, m)
    G.add_node(n+1)
    G.add_edge(n+1, n-1)
    G.add_edge(n+1, n-2)
    G.add_edge(n+1, n-3)
    G.add_edge(n+1, n-4)
    G.add_edge(n+1, n-5)
    G.add_edge(n+1, n-6)
    for (u,v,w) in G.edges(data=True):
        w['weight'] = 1
    graph_data = nx.to_pandas_edgelist(G, source="seller id", target="buyer id")
    graph_data.columns = ["seller id","buyer id", "total_value"]
    graph_data["tuple_id"] = list(G.edges())
    graph_data['trade date time'] = timestamp
    return(graph_data)

    
def netstats_generator(g_df):
    ''' Function to calculate the values of the four structural node importance 
    metrics, along with pagerank, centrality, degree and community label. 
    Parameters:
        g_df: pandas edgelist dataframe with columns 'buyer id', 'seller id' and 'total_value' 
    Returns:
        netstats: dataframe containing the above metrics computed for each node
    '''
    G = nx.from_pandas_edgelist(g_df, source="seller id", target = "buyer id",
                                edge_attr = ['total_value'],
                                create_using=nx.MultiDiGraph())
    monthly_values_list = [[] for i in range(9)]
    whole_graph_uni = nif.multi_edge_to_uni(G)
    for node in whole_graph_uni.nodes():
        monthly_values_list[0].append(node)

        x = g_df.groupby(g_df["trade date time"], 
                         axis=0).apply(lambda x: nif.mi_generator_symm_tests(x, 
                               node, test_type='all', 
                               weight=False)).apply(pd.Series)
        monthly_values_list[1].append(x.iloc[:,0])
        monthly_values_list[2].append(x.iloc[:,1])
        monthly_values_list[3].append(x.iloc[:,2])
        monthly_values_list[4].append(x.iloc[:,3])  
        monthly_values_list[5].append(x.iloc[:,4])  
        monthly_values_list[6].append(x.iloc[:,5])  
        monthly_values_list[7].append(x.iloc[:,6])  
        monthly_values_list[8].append(x.iloc[:,7])  

        m_a = pd.DataFrame.from_records(monthly_values_list[1] , 
                                        index=monthly_values_list[0])
        m_b = pd.DataFrame.from_records(monthly_values_list[2] , 
                                        index=monthly_values_list[0])
        m_c = pd.DataFrame.from_records(monthly_values_list[3] , 
                                        index=monthly_values_list[0])
        m_d = pd.DataFrame.from_records(monthly_values_list[4] , 
                                        index=monthly_values_list[0])
        eig_cent = pd.DataFrame.from_records(monthly_values_list[5] , 
                                             index=monthly_values_list[0])
        pagerank = pd.DataFrame.from_records(monthly_values_list[6] , 
                                             index=monthly_values_list[0])
        degree = pd.DataFrame.from_records(monthly_values_list[7] , 
                                           index=monthly_values_list[0])
        community = pd.DataFrame.from_records(monthly_values_list[8] , 
                                              index=monthly_values_list[0])
    netstats = pd.concat([m_a.melt().reset_index().set_index(['index', 'trade date time']),
                          m_b.melt().reset_index().set_index(['index', 'trade date time']),
                          m_c.melt().reset_index().set_index(['index', 'trade date time']),
                          m_d.melt().reset_index().set_index(['index', 'trade date time']),
                          eig_cent.melt().reset_index().set_index(['index', 'trade date time']),
                          pagerank.melt().reset_index().set_index(['index', 'trade date time']),
                          degree.melt().reset_index().set_index(['index', 'trade date time']),
                          community.melt().reset_index().set_index(['index', 'trade date time'])],
                         axis=1)
    netstats.columns = ['m_a', 'm_b', 'm_c', 'm_d','eig_cent','pagerank','degree','community']
    netstats =netstats[[type(x)!=str for x in netstats.m_a]] 
    netstats =netstats[[type(x)!=str for x in netstats.m_b]] 
    netstats =netstats[[type(x)!=str for x in netstats.m_c]] 
    netstats =netstats[[type(x)!=str for x in netstats.m_d]] 
    netstats =netstats[[type(x)!=str for x in netstats.eig_cent]] 
    netstats =netstats[[type(x)!=str for x in netstats.pagerank]] 
    netstats =netstats[[type(x)!=str for x in netstats.degree]] 
    netstats =netstats[[type(x)!=str for x in netstats.community]] 
    return(netstats)
    

if __name__ == "__main__":

    ######### Multi-star network
    uw_ms = static_hub_gen(50, 10,10, 0)
    netstats_hub = netstats_generator(uw_ms)
    netstats_hub.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    scaler = MinMaxScaler()
    netstats_hub['m_b_scaled'] = scaler.fit_transform(netstats_hub['m_b'].values.reshape(-1,1))
    netstats_hub['change_bool'] = [np.random.binomial(1, min(i, 0.99), 1)[0] for i in netstats_hub.m_b_scaled]
    print(netstats_hub.change_bool.value_counts())
    g=nx.from_pandas_edgelist(uw_ms,source="seller id", target = "buyer id",edge_attr = ['total_value'],create_using=nx.Graph())
    nodes=pd.Series(list(g.nodes()))
    plt.figure(figsize=(10,10))
    pos = nx.spring_layout(g, k=0.15, iterations=20)
    ec = nx.draw_networkx_edges(g, pos)
    nc = nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color='r', alpha=0.5)
    plt.axis('off')
    plt.show()
    corr = netstats_hub[[ "m_a", "m_b",'eig_cent','pagerank', 'community', 'degree']].corr(method='kendall')
    plt.subplots(figsize=(10,10))
    sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns, annot=True, annot_kws={"fontsize":10})
    plt.show() 
    #removing the correlated correlated variables - remove >0.7. m_a so not pr, comm. eig_cent so not comm. 
    X_hub= netstats_hub[[ "m_a", "m_b", "m_c", "m_d",'eig_cent','degree']].fillna(0) 
    y_hub = netstats_hub['change_bool']
    pred.node_change_prediction(netstats_hub, X_hub,y_hub)
        
    ########### Erdos-Renyi
    uw_er = static_er_gen(400,0)
    netstats_er = netstats_generator(uw_er)
    netstats_er.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    corr = netstats_er.corr()
    plt.subplots(figsize=(10,10))
    sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns, annot=True, annot_kws={"fontsize":10})
    plt.show()
    scaler = MinMaxScaler()
    netstats_er['m_b_scaled'] = scaler.fit_transform(netstats_er['m_b'].values.reshape(-1,1))
    netstats_er['change_bool'] = [np.random.binomial(1, min(i, 0.99), 1)[0] for i in netstats_er.m_b_scaled]
    print(netstats_er.change_bool.value_counts())
    X_er = netstats_er[[ "m_a", "m_b", "m_c", 'degree', 'community']].fillna(0)
    y_er = netstats_er['change_bool']
    pred.node_change_prediction(netstats_er, X_er,y_er)
    
    ############# Barbell
    uw_sbm = static_bb_gen(50, 100,0)
    netstats_bb = netstats_generator(uw_sbm)
    netstats_bb.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    corr = netstats_bb.corr()
    plt.subplots(figsize=(10,10))
    sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns, annot=True, annot_kws={"fontsize":10})
    plt.show()
    scaler = MinMaxScaler()
    netstats_bb['m_b_scaled'] = scaler.fit_transform(netstats_bb['m_b'].values.reshape(-1,1))
    netstats_bb['change_bool'] = [np.random.binomial(1, min(i, 0.99), 1)[0] for i in netstats_bb.m_b_scaled]
    print(netstats_bb.change_bool.value_counts())
    X_bb = netstats_bb[[ "m_a", "m_b", "m_c", "m_d",'pagerank']].fillna(0)
    y_bb = netstats_bb['change_bool']
    pred.node_change_prediction(netstats_bb, X_bb,y_bb)
