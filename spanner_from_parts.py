from imports import *
from utils import *
from spanner_local_graph import *

def spanner_from_coloring(G, non_intersecting_sets, add_err=2):
#def spanner_from_coloring(G, pairs, approx_path_lengths, non_intersecting_sets):
  #'''
  approx_path_lengths = dict(nx.all_pairs_dijkstra_path_length(G))
  approx_path_lengths = additive_error_metric(approx_path_lengths, add_err)

  pairs = []
  for u in approx_path_lengths:
      for v in approx_path_lengths[u]:
          pairs.append((u,v))
  #'''

  EG = nx.Graph()
  k = 0

  found_spanner = False
  #count = 0
  while not found_spanner:
    print(k)
    EG.add_weighted_edges_from([(u,v,1) for u, v in non_intersecting_sets[k]])
    #print(nx.is_connected(EG))

    distances = {}
    for source_node in EG.nodes():
      lengths, paths = nx.single_source_dijkstra(EG, source_node, weight='weight')
      distances[source_node] = (lengths, paths)

    '''
    EG_pairs = []
    EG_nodes = list(EG.nodes())
    for i in range(len(EG_nodes)):
      u = EG_nodes[i]
      for j in range(i+1,len(EG_nodes)):
        v = EG_nodes[j]
        EG_pairs.append((u,v))
    '''

    EG_pairs = []
    EG_nodes = list(EG.nodes())
    for u,v in pairs:
      if (u in EG_nodes) and (v in EG_nodes):
        EG_pairs.append((u,v))

    #print(G, pairs, approx_path_lengths, np.sqrt(n), forward_distances, backward_distances)
    try:
      spanner = compute_without_elipsoid(EG, EG_pairs, approx_path_lengths, np.sqrt(EG.number_of_nodes()), distances)
    except Exception as exc:
      print(exc)
      spanner = None

    try:
      if is_pairwise_spanner(spanner, pairs, approx_path_lengths):
          found_spanner = True
    except Exception as exc:
      print(exc)

    k += 1

  return spanner

#def spanner_from_coloring(G, non_intersecting_sets, add_err=2):
def spanner_from_coloring_pairwise(G, pairs, approx_path_lengths, non_intersecting_sets):
  '''
  approx_path_lengths = dict(nx.all_pairs_dijkstra_path_length(G))
  approx_path_lengths = additive_error_metric(approx_path_lengths, add_err)

  pairs = []
  for u in approx_path_lengths:
      for v in approx_path_lengths[u]:
          pairs.append((u,v))
  '''

  EG = nx.Graph()
  k = 0

  found_spanner = False
  #count = 0
  while not found_spanner:
    print(k)
    EG.add_weighted_edges_from([(u,v,1) for u, v in non_intersecting_sets[k]])
    #print(nx.is_connected(EG))

    distances = {}
    for source_node in EG.nodes():
      lengths, paths = nx.single_source_dijkstra(EG, source_node, weight='weight')
      distances[source_node] = (lengths, paths)

    '''
    EG_pairs = []
    EG_nodes = list(EG.nodes())
    for i in range(len(EG_nodes)):
      u = EG_nodes[i]
      for j in range(i+1,len(EG_nodes)):
        v = EG_nodes[j]
        EG_pairs.append((u,v))
    '''

    EG_pairs = []
    EG_nodes = list(EG.nodes())
    for u,v in pairs:
      if (u in EG_nodes) and (v in EG_nodes):
        EG_pairs.append((u,v))

    #print(G, pairs, approx_path_lengths, np.sqrt(n), forward_distances, backward_distances)
    try:
      spanner = compute_without_elipsoid(EG, EG_pairs, approx_path_lengths, np.sqrt(EG.number_of_nodes()), distances)
    except Exception as exc:
      print(exc)
      spanner = None

    try:
      if is_pairwise_spanner(spanner, pairs, approx_path_lengths):
          found_spanner = True
    except Exception as exc:
      print(exc)

    k += 1

  return spanner

def prune_spanner(G, pairs):
    """
    Build a subgraph H containing only edges needed to preserve
    shortest-path distances for the given vertex pairs.
    """
    # Initialize H with the same nodes as G, but no edges
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    for u, v in pairs:
        # Shortest path and distance in G
        path_G = nx.shortest_path(G, u, v, weight="weight")
        dist_G = nx.shortest_path_length(G, u, v, weight="weight")

        # Check shortest path in H (if it exists)
        try:
            dist_H = nx.shortest_path_length(H, u, v, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            dist_H = float("inf")

        # If H does not preserve the shortest-path distance, add edges from G's path
        if dist_H != dist_G:
            for a, b in zip(path_G[:-1], path_G[1:]):
                # Copy the edge with its weight
                H.add_edge(a, b, weight=G[a][b]["weight"])

    return H

