
from imports import *
from utils import *

def sample_with_settlement(graph, approx_path_lengths, beta, thick_pairs, distances):
    """
    This method computes a graph spanner using random sampling.

    Parameters:
    - graph: a networkx graph object
    - sample_rate: percentage (between 0 and 1) of edges to sample for the spanner

    Returns:
    - spanner: a subgraph of the original graph (networkx Graph object)
    """
    spanner_edges = set() # E' in Berman et al., Approximation algorithms for spanner problems and Directed Steiner Forest
    selected_vertices = set() # S in Berman at al.
    n = graph.number_of_nodes()
    for i in range(1, int(beta*np.log(n)+1)):
      v = random.randint(0, n-1)
      #print(v)
      #arborescence = nx.minimum_spanning_arborescence(graph, root=v)
      # not minimum spanning arborescence, it should be shortest path arborescence
      #arborescence = minimum_spanning_arborescence(graph, v)
      # shortest path in arborescence
      #graph = graph.reverse()
      #in_shortest_paths = nx.single_source_dijkstra_path(graph, source=v)
      #in_shortest_paths = {}
      #if v in backward_distances:
      #  in_shortest_paths = copy.deepcopy(backward_distances[v][1])
      #for u,p in in_shortest_paths.items():
      #  in_shortest_paths[u]=p[::-1]
      #graph = graph.reverse()
      # shortest path out arborescence
      #out_shortest_paths = nx.single_source_dijkstra_path(graph, source=v)
      shortest_paths = {}
      if v in distances:
        shortest_paths = distances[v][1]
      # add all edges
      #for p in in_shortest_paths.values():
      #  spanner_edges.update(path_edges(p))
      for p in shortest_paths.values():
        spanner_edges.update(path_edges(p))
      # add v to S
      selected_vertices.add(v)
    #print('spanner', spanner_edges, 'graph', graph.edges())
    spanner = nx.Graph()
    for e in spanner_edges:
      u, v = e
      spanner.add_edge(u, v, weight=graph[u][v]['weight'])
    spanner_path_lengths = dict(nx.all_pairs_dijkstra_path_length(spanner))
    #thick_edges = find_thick_edges(graph, approx_path_lengths, beta)
    #thick_edges = find_thick_pairs(graph, approx_path_lengths, beta)
    # Add all unsettled thick edges
    #for e in thick_edges:
    for e in thick_pairs:
      u, v = e
      #if spanner_path_lengths[u][v]>approx_path_lengths[u][v]:
      if u not in spanner_path_lengths or v not in spanner_path_lengths[u] or spanner_path_lengths[u][v]>approx_path_lengths[u][v]:
        lengths, paths = distances[u]
        for e in path_edges(paths[v]):
          p, q = e
          spanner.add_edge(p, q, weight=graph[p][q]['weight'])
    return spanner

def lengths_from_source(G, source_node, threshold, distances):
  #print(G, source_node, threshold, distances)
  # Compute shortest path lengths and paths from the source
  #lengths, paths = nx.single_source_dijkstra(G, source_node, weight='weight')
  lengths, paths = distances[source_node]

  # Filter paths within the threshold
  shortest_paths_within_threshold = {}
  for target, length in lengths.items():
      if length <= threshold:
          shortest_paths_within_threshold[target] = length

  return shortest_paths_within_threshold

def is_thick_pair(G, e, threshold, beta, distances):
  s, t = e
  #print("s, t, threshold", s, t, threshold)
  try:
    source_d = lengths_from_source(G, s, threshold, distances)
  except:
    print("error in lengths_from_source")
  try:
    #target_d = lengths_to_target(G, t, threshold, already_computed)
    target_d = lengths_from_source(G, t, threshold, distances)
  except:
    print("error in lengths_to_target")
  commons = set(source_d.keys()) & set(target_d.keys())
  count = 0
  for u in commons:
    try:
      if source_d[u]+target_d[u]<=threshold:
        count += 1
    except:
      i=0
  return count>=beta

def find_pairwise_thick_pairs(G, pairs, thresholds, beta, distances):
  #print('beta:', beta)
  P = set()
  for u, v in pairs:
    try:
      if is_thick_pair(G, (u,v), thresholds[u][v], beta, distances):
        P.add((u,v))
    except Exception:
      i=0
      #print(Exception)
  return P


def compute_without_elipsoid(graph, pairs, approx_path_lengths, sample_rate, distances):
    #print('********************************************************')
    #print(list(graph.nodes()))
    #print(list(graph.edges()))
    #global total_time_block1
    #global total_time_block2
    #global total_time_block3
    #global total_time_block4
    start_time = time.time()
    n = graph.number_of_nodes()
    thick_pairs = find_pairwise_thick_pairs(graph, pairs, approx_path_lengths, sample_rate, distances)
    #print('thick_pairs', thick_pairs)
    thin_pairs = []
    for u, v in pairs:
      if (u,v) not in thick_pairs and u!=v:
        thin_pairs.append((u,v))
    #print('thin_pairs', thin_pairs)
    end_time = time.time()
    #total_time_block1 += (end_time - start_time)
    start_time = time.time()
    #print('before sample_pairwise')
    #spanner_thick = sample_pairwise(graph, approx_path_lengths, sample_rate, thick_pairs, forward_distances, backward_distances)
    spanner_thick = sample_with_settlement(graph, approx_path_lengths, sample_rate, thick_pairs, distances)
    #print('after sample_pairwise')
    end_time = time.time()
    #total_time_block2 += (end_time - start_time)
    #print("Thick pairs:", thick_pairs)
    #print("Thin pairs:", thin_pairs)
    start_time = time.time()
    min_spanner = nx.Graph()
    min_spanner_weight = 0
    for u,v in thin_pairs:
      lengths, paths = distances[u]
      for e in path_edges(paths[v]):
        p, q = e
        wgt = graph[p][q]['weight']
        min_spanner.add_edge(p, q, weight=wgt)
        min_spanner_weight += wgt
    end_time = time.time()
    #total_time_block3 += (end_time - start_time)
    #print('Thick & thin spanner edges:', spanner_thick.number_of_edges(), min_spanner.number_of_edges())
    start_time = time.time()
    spanner = nx.Graph()
    for e in spanner_thick.edges():
      u, v = e
      spanner.add_edge(u, v, weight=spanner_thick[u][v]['weight'])
    if min_spanner_weight!=None:
      for e in min_spanner.edges():
        u, v = e
        spanner.add_edge(u, v, weight=min_spanner[u][v]['weight'])
    end_time = time.time()
    #total_time_block4 += (end_time - start_time)
    return spanner

def compute_pairwise_spanner_with_guarantee(graph, pairs, approx_path_lengths, sample_rate, distances):
  found_spanner = False
  #count = 0
  while not found_spanner:
    #print('Computing for node size:', graph.number_of_nodes())
    #spnnr = compute_pairwise_spanner(graph, pairs, approx_path_lengths, sample_rate, forward_distances, backward_distances)
    spnnr = compute_without_elipsoid(graph, pairs, approx_path_lengths, sample_rate, distances)
    #count += 1
    #print('Graph and spanner size:', graph.number_of_edges(), spnnr.number_of_edges())
    #print(list(graph.edges()))
    #print(list(spnnr.edges()))
    if is_pairwise_spanner(spnnr, pairs, approx_path_lengths):
      found_spanner = True
    #else:
    #  print('Not a valid spanner, recomputing ...')
    #  #input()
  #print('Computed spanner after', count, 'tries')
  return spnnr