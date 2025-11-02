from imports import *

def create_graph(n, p):
  G = nx.erdos_renyi_graph(n, p)

  for u, v in G.edges():
      #G_directed[u][v]['weight'] = random.randint(1, 10)  # Random weights for edges
      G[u][v]['weight'] = 1

  return G

def is_pairwise_spanner(spanner, pairs, approx_path_lengths):
  for u in approx_path_lengths:
    if u not in spanner.nodes():
      spanner.add_node(u)
  spanner_path_lengths = dict(nx.all_pairs_dijkstra_path_length(spanner))
  '''
  for u in spanner.nodes():
    if u not in spanner_path_lengths:
      spanner_path_lengths[u] = {u:0}
    else:
      spanner_path_lengths[u][u] = 0
  '''
  #print(spanner_path_lengths)
  #print(approx_path_lengths)
  for u, v in pairs:
    if u not in spanner_path_lengths:
      print(u, ' not in spanner_path_lengths')
      return False
    if v not in spanner_path_lengths[u]:
      print(v, ' not in spanner_path_lengths[', u, ']')
      return False
    if spanner_path_lengths[u][v]>approx_path_lengths[u][v]:
      print('spanner_path_lengths[', u, '][', v, ']>approx_path_lengths[u][v]')
      return False
  return True

def path_edges(path):
  edges_in_path = []
  for i in range(len(path) - 1):
      u = path[i]
      v = path[i+1]
      edges_in_path.append((u, v))

  return edges_in_path