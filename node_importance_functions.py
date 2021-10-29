# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 14:57:46 2021

@author: iseabrook1
"""

#This script contains the functions to calculate node importance
#and related analyses as presented in Seabrook et. al., 
#Community aware evaluation of node importance 

#Isobel Seabrook, ucabeas@ucl.ac.uk
#MIT License. Please reference below publication if used for research purposes.
#Reference: Seabrook et al., Community aware evaluation of node importance
#

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import networkx.algorithms.community as nxcom
from itertools import count

    
def multi_edge_to_uni(G):
    """This function takes in a graph which has multiple edges between two nodes,
    and sums them to a single edge.
    
    Parameters:
        G: networkx graph with potentially multi-edges
        
    Returns:
        G_uni: graph with multi-edge weights summed to give single total weight.
    """
    G_uni = nx.Graph()
    for u,v,data in G.edges(data=True):
        w = data['total_value']
        if G_uni.has_edge(u,v):
            G_uni[u][v]['total_value'] += w
        else:
            G_uni.add_edge(u, v, total_value=w)
    return(G_uni)


def set_node_community(G, communities):
    '''Add community to node attributes'''
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = c + 1
        
        
def mi_generator_symm_tests(data, node, test_type, weight=True, google_matrix=False):
    """ Function to generate the values of m_{a-c}, eigenvector centrality, pagerank, degree and 
    community for an individual node
    Parameters:
        data: pandas edgelist dataframe for network snapshot - columns seller id, buyer id, total_value 
        and trade date time.
        edge: edge tuple (seller id, buyer id) 
    Returns:
        tuple of values of calculated metrics. Where nodes are not found in the giant component, no value is returned.
    """
    #print(node)
    M = nx.from_pandas_edgelist(data, source="seller id", target = "buyer id",
                                edge_attr = ['total_value'],
                                create_using=nx.MultiGraph())
    if node in M.nodes():
        G = nx.Graph()#DiGraph if directed
        G.add_nodes_from(M)
        for u,v,data1 in M.edges(data=True):
            w = data1['total_value'] if 'total_value' in data else 1.0
            if G.has_edge(u,v):
                G[u][v]['weight'] += w
            else:
                G.add_edge(u, v, weight=w)
        if google_matrix==True:
            A = nx.google_matrix(G)
        else:
            A = nx.to_numpy_matrix(G.to_undirected())        
        #S = A.sum(axis=0)#   
        S = pd.Series([val for (node, val) in G.degree(weight='weight')], index=[node for (node, val) in G.degree(weight='weight')])
        #print(S)
        eigenvalues, eigvecs = np.linalg.eigh(A)
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]
        eigvecs = eigvecs[:,idx]
        #retrieve all of the eigenvector components for node 
        #print(eigenvalues)
        eigvecs_df = pd.DataFrame(eigvecs, index=G.nodes, columns = eigenvalues)#.abs()
        #print(eigvecs_df)
        max_eigvec = eigvecs_df.iloc[:,0]
        grad_A = max_eigvec.loc[node]*max_eigvec
        m_i_a = (2/S.loc[node])*(grad_A.sum())     
        if weight == True:
            max_eigvec = eigvecs_df[[x for x in eigenvalues if x>0]].abs().max(axis=1)
            max_eigvec_eigval = eigvecs_df[[x for x in eigenvalues if x>0]].abs().idxmax(axis=1)
            weighted_max_eigvec = max_eigvec.multiply(max_eigvec_eigval)
            lead_eig = eigenvalues[0]
            max_eigvec = weighted_max_eigvec/lead_eig
        else:
            max_eigvec = eigvecs_df[[x for x in eigenvalues if x>0]].abs().max(axis=1)
        grad_A = max_eigvec.loc[node]*max_eigvec
        m_i_b = (2/S.loc[node])*(grad_A.sum()) 
        eigvecs_sum = eigvecs_df.sum(axis=1)
        grad_A = eigvecs_sum.loc[node]*eigvecs_sum
        m_i_c = (2/S.loc[node])*(grad_A.sum())
        eigvals_pos = [x for x in eigenvalues if x > 0]
        eigvecs_sum = eigvecs_df.loc[:,eigvals_pos].sum(axis=1)
        grad_A = eigvecs_sum.loc[node]*eigvecs_sum
        m_i_d = (2/S.loc[node])*(grad_A.sum()) 
        #eig_cent
        if node in G.nodes():      
            try:
                eig_cent = nx.eigenvector_centrality(G, weight='total_value')[node]
            except:
                try:
                    eig_cent = nx.eigenvector_centrality_numpy(G, weight='total_value',max_iter=1000)[node]  
                except:
                    eig_cent=0            
        else:
            eig_cent = 0
        #pagerank 
        if node in G.nodes():
            try:
                pagerank = nx.pagerank(G, weight='total_value')[node]
            except:
                pagerank=0
        else:
            pagerank = 0
        #degree
        if node in G.nodes():
            try:
                deg = G.degree()[node]
            except:
                deg=0
        else:
            deg = 0
        #community
        communities = sorted(nxcom.greedy_modularity_communities(G.to_undirected()), key=len, reverse=True)
        set_node_community(G, communities)
        if node in G.nodes():
            try:
                comm = nx.get_node_attributes(G, 'community')[node]
            except:
                comm=0
        else:
            comm = 0
        return((m_i_a, m_i_b, m_i_c, m_i_d, eig_cent, pagerank, deg, comm))
    else:
        return(["no value" for i in range(8)])

    

def da_le_pairs_test(g_df, test_type='all', weight=True):
    """ Function to generate dataframe of dS m_i pairs. 
    Parameters:
        g_df: pandas edgelist dataframe for network snapshot - columns seller id, buyer id, total_value and trade date time.
    Returns: dataframe, columns:
        trade_date_time: timestamp  
        m_{a-d}: values of structural importance calculated for each node
        eig_cent: eigenvector centrality of each node
        pagerank: pagerank of each node
        degree: degree of each node
        community: greedy modularity community label for each node
        delta_S_act: subsequent obserenrved dA value. 
        S_init: initial strength
        S_fin: final strength
        delta_S_rel: relative change in Strength (S_fin-S_init/S_init)
        variable: edge tuple
        log_delta_S_rel: natural log of delta_S_rel
    """
    G = nx.from_pandas_edgelist(g_df, source="seller id", target = "buyer id",edge_attr = ['total_value'],create_using=nx.MultiGraph())
    monthly_values_list = [[] for i in range(10)]
    whole_graph_uni = multi_edge_to_uni(G)
    for node in whole_graph_uni.nodes():
        monthly_values_list[0].append(node)
        strength_series = pd.Series(g_df[(g_df["buyer id"]==node) | (g_df["seller id"]==node)].sort_values(by="trade date time").groupby("trade date time").total_value.sum())
        monthly_values_list[1].append(strength_series)
        x = g_df.groupby(g_df["trade date time"], axis=0).apply(lambda x: mi_generator_symm_tests(x, node, test_type, weight)).apply(pd.Series)
        monthly_values_list[2].append(x.iloc[:,0])
        monthly_values_list[3].append(x.iloc[:,1])
        monthly_values_list[4].append(x.iloc[:,2])
        monthly_values_list[5].append(x.iloc[:,3])
        monthly_values_list[6].append(x.iloc[:,4])
        monthly_values_list[7].append(x.iloc[:,5])  
        monthly_values_list[8].append(x.iloc[:,6])
        monthly_values_list[9].append(x.iloc[:,7])  
    # create a dataframe with columns for the value of m_i for a given node for a given time, including both relative and absolute change and timestamp
    A= pd.DataFrame.from_records(monthly_values_list[1] , index=monthly_values_list[0])
    A = A.apply(lambda series: series.loc[:series.last_valid_index()].ffill(), axis=1)
    A_T=A.T
    A_T_shift = A.shift(-1,axis=1).T
    A_T.index.name = "trade date time"
    A_T_shift.index.name = "trade date time"
    m_a = pd.DataFrame.from_records(monthly_values_list[2] , index=monthly_values_list[0])
    m_b = pd.DataFrame.from_records(monthly_values_list[3] , index=monthly_values_list[0])
    m_c = pd.DataFrame.from_records(monthly_values_list[4] , index=monthly_values_list[0])
    m_d = pd.DataFrame.from_records(monthly_values_list[5] , index=monthly_values_list[0])
    eig_cent = pd.DataFrame.from_records(monthly_values_list[6] , index=monthly_values_list[0])
    pagerank = pd.DataFrame.from_records(monthly_values_list[7] , index=monthly_values_list[0])
    degree = pd.DataFrame.from_records(monthly_values_list[8] , index=monthly_values_list[0])
    community = pd.DataFrame.from_records(monthly_values_list[9] , index=monthly_values_list[0])
    ds_mi = pd.concat([A_T.reset_index().melt(id_vars="trade date time").set_index(["trade date time","variable"]),\
                       A_T_shift.reset_index().melt(id_vars="trade date time").set_index(["trade date time","variable"]),\
                           m_a.T.reset_index().melt(id_vars="trade date time").set_index(["trade date time","variable"]), \
                           m_b.T.reset_index().melt(id_vars="trade date time").set_index(["trade date time","variable"]), \
                           m_c.T.reset_index().melt(id_vars="trade date time").set_index(["trade date time","variable"]), \
                           m_d.T.reset_index().melt(id_vars="trade date time").set_index(["trade date time","variable"]), \
                           eig_cent.T.reset_index().melt(id_vars="trade date time").set_index(["trade date time","variable"]), \
                           pagerank.T.reset_index().melt(id_vars="trade date time").set_index(["trade date time","variable"]), \
                           degree.T.reset_index().melt(id_vars="trade date time").set_index(["trade date time","variable"]), \
                           community.T.reset_index().melt(id_vars="trade date time").set_index(["trade date time","variable"])],
                          join = 'inner', axis=1, sort=True)
    ds_mi.reset_index(level=0, inplace=True)
    ds_mi.columns = ["trade date time","S_init","S_fin", "m_a","m_b","m_c","m_d", "eig_cent", "pagerank", "degree", "community" ]
    ds_mi["delta_S_act"] = ds_mi.S_fin - ds_mi.S_init
    ds_mi =ds_mi[[type(x)!=str for x in ds_mi.m_a]] 
    ds_mi =ds_mi[[type(x)!=str for x in ds_mi.m_b]] 
    ds_mi =ds_mi[[type(x)!=str for x in ds_mi.m_c]] 
    ds_mi =ds_mi[[type(x)!=str for x in ds_mi.m_d]] 
    ds_mi =ds_mi[[type(x)!=str for x in ds_mi.eig_cent]] 
    ds_mi =ds_mi[[type(x)!=str for x in ds_mi.pagerank]] 
    ds_mi =ds_mi[[type(x)!=str for x in ds_mi.degree]] 
    ds_mi =ds_mi[[type(x)!=str for x in ds_mi.community]] 
    ds_mi.reset_index(inplace=True)
    ds_mi["delta_S_rel"] = ds_mi.delta_S_act/ds_mi["S_init"]
    ds_mi = ds_mi[ds_mi.delta_S_rel!=np.inf]
    ds_mi["log_delta_S_rel"] = np.log(1+ds_mi.delta_S_rel)
    ds_mi=ds_mi[ds_mi.log_delta_S_rel!=-np.inf]
    return(ds_mi)


def plot_violinplots_multimeasure(ds_mi, measure_columns = ["m_a", "m_b", "m_c", "m_d",'eig_cent','pagerank','degree', 'community']):
    """ Function to generate boxplots for the distributions
        of measure values for nodes which don't change in comparison
        to those that do. 
        Parameters:
            ds_mi: pandas dataframe with columns for each node including measure_columns, and:
                trade_date_time: timestamp  
                change_bool: boolean indicator for whether or not node subsequently changes
        Returns: p: p-value of t-test applied to differences in the means of the distributions of the measures
        in measure_columns, for nodes that change vs. nodes that don't change
        plot: boxplots showing the distributions of each of the measures in measure_columns for nodes 
        that change vs. nodes that don't change.
    """
    #scale each of the measures to a range of 0,1. 
    ds_mi[measure_columns] -= ds_mi[measure_columns].min()  # equivalent to df = df - df.min()
    ds_mi[measure_columns] /= ds_mi[measure_columns].max()  # equivalent to df = df / df.max()
    #melt dataframe 
    ds_mi_melt = ds_mi[measure_columns].melt(id_vars=['change_bool'], var_name=["measure"])
    ds_mi_melt["measure"] = ds_mi_melt.measure.str.replace("m_","")
    ds_mi_melt["measure"] = ds_mi_melt.measure.str.replace("eig_cent","eig. cent.")
    def ttest(ds):
        g1 = ds[(ds['Change indicator']==0)]["value"].values
        g2 = ds[(ds['Change indicator']==1)]["value"].values
        t,p=stats.ttest_ind(g1,g2)
        return(p)
    ds_mi_melt.columns = ["Change indicator", "measure", "value"]
    p = ds_mi_melt.groupby('measure').apply(ttest)
    print(p)
    fig, axs=plt.subplots(1,1, figsize=(12,5))
    sns.violinplot(x="measure", y="value", hue='Change indicator',  data=ds_mi_melt, ax=axs)
    #axs.set_title(f'p-value = {p:.3e}')
    axs.set_xlabel("Change label")
    axs.set_ylabel("Measure value (Scaled)")
    plt.show()
    return(p)
    
def plot_network_ev_rankings(raw_data, ds_mi):
    """ Function to produce a visualisation of a network, with nodes colored and numbered by the 
    ranking of the eigenvalue that localises to that node. 
        Parameters:
            raw_data: pandas edgelist dataframe for network snapshot - columns seller id, buyer id, total_value and trade date time.
            ds_mi: pandas dataframe with columns for each node including measure_columns, and:
                trade_date_time: timestamp  
                change_bool: boolean indicator for whether or not node subsequently changes
        Returns: plot of network with nodes coloured and numbered by the eigenvalue ranking.
    """
    raw_data.sort_values(by=['buyer id', 'seller id', 'trade date time'])
    raw_data['change_bool'] = raw_data.groupby(['buyer id', 'seller id']).total_value.diff(periods = -1).fillna(0)!=0
    init_snapshot = raw_data[raw_data['trade date time']==min(raw_data['trade date time'])]
    init_snapshot["tuple_id"]=[(u, v) for u, v in zip(init_snapshot['buyer id'], init_snapshot['seller id'])]
    changing_nodes = pd.unique(raw_data[raw_data.change_bool==True][['buyer id', 'seller id']].values.ravel('K'))
    g = nx.from_pandas_edgelist(init_snapshot, source="seller id", target = "buyer id",edge_attr = ['total_value'],create_using=nx.MultiGraph())
    A = nx.to_numpy_matrix(g, weight='total_value')
    #coloring the nodes according to eigenvalue localisation
    eigenvalues, eigvecs = np.linalg.eigh(A)
    idx = eigenvalues.argsort()[::-1]   
    eigenvalues = eigenvalues[idx]
    eigvecs = eigvecs[:,idx]
    #retrieve all of the eigenvector components for node 
    eigvecs_df = pd.DataFrame(eigvecs, index=g.nodes, columns = eigenvalues)#.abs()
    max_eigvec_eigval = eigvecs_df[[x for x in eigenvalues if x>0]].abs().idxmax(axis=1).to_frame()
    def rank_unique(x, **kwargs):
        sx = sorted(set(x), **kwargs)# will put x in order of min to max. So high rank is 
        invsx = {s: i for i, s in enumerate(sx)}
        return [1 + invsx[v] for v in x]
    max_eigvec_eigval['ev_rank']=rank_unique(max_eigvec_eigval[0])
    node_attr = dict(zip(max_eigvec_eigval.index, max_eigvec_eigval['ev_rank']))
    weights = [np.sqrt(w.get('total_value'))/50 for u,v,w in g.edges(data=True)]
    nx.set_node_attributes(g,node_attr,'node_attr')
    # create number for each group to allow use of colormap
    # get unique groups
    groups = set(nx.get_node_attributes(g,'node_attr').values())
    mapping = dict(zip(sorted(groups),count()))
    nodes = pd.Series(list(g.nodes()))
    ds_mi = ds_mi[ds_mi.variable.isin(nodes)]
    changing_nodes = changing_nodes[pd.Series(changing_nodes).isin(nodes)]
    colors_changing = [mapping[g.nodes(data=True)[n]['node_attr']] if n in ds_mi['variable'].values else 0 for n in changing_nodes]
    colors_unchanging = [mapping[g.nodes(data=True)[n]['node_attr']] if n in ds_mi['variable'].values else 0 for n in nodes[~nodes.isin(changing_nodes)]]
    # drawing nodes and edges separately so we can capture collection for colobar
    plt.figure(figsize=(10,10))
    pos = nx.spring_layout(g, k=0.15, iterations=20)
    ec = nx.draw_networkx_edges(g, pos, width=weights)
    v_max = max(max([int(i) for i in colors_unchanging]), max([int(i) for i in colors_changing]))
    nc = nx.draw_networkx_nodes(g, pos, nodelist=changing_nodes, node_color=colors_changing, alpha=0.5,
                                 node_size=100,cmap=plt.get_cmap('viridis'), vmin=0, vmax=v_max)
    nc = nx.draw_networkx_nodes(g, pos, nodelist=nodes[~nodes.isin(changing_nodes)], node_color=colors_unchanging,alpha=0.5, 
                                node_size=100, node_shape='s',cmap=plt.get_cmap('viridis'), vmin=0, vmax=v_max)
    nx.draw_networkx_labels(g,pos, nx.get_node_attributes(g, 'node_attr'))
    plt.colorbar(nc)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    None
