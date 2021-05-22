import pandas as pd
import networkx as nx
import numpy as np


def build_flights_graph(flights_df: pd.DataFrame) -> nx.Graph:
    """
    Build a graph where:
     - a node is an airport
     - an edge means that there were flights between the airports
     - edges are weighted by the number of flights between the corresponding airports
    :param flights_df: [Origin, Dest, ArrTs] dataframe
    :return: the graph described above
    """
    node_pairs = flights_df[['Origin', 'Dest']].values
    indices = node_pairs.argsort(axis=1)
    edges = pd.DataFrame({
        'source': np.take_along_axis(node_pairs, indices[:, 0].reshape(-1, 1), axis=1).flatten(),
        'target': np.take_along_axis(node_pairs, indices[:, 1].reshape(-1, 1), axis=1).flatten()
    })
    edges = edges.groupby(['source', 'target']).apply(len)
    edges = edges.to_frame(name='weight').reset_index()
    edges.weight /= edges.weight.sum()

    graph = nx.from_pandas_edgelist(edges, edge_attr=['weight'])

    return graph
