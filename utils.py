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

def additive_error_metric(metric, error):
  res_metric = {}
  for u in metric:
    res_metric[u] = {}
    for v in metric[u]:
      res_metric[u][v] = metric[u][v] + error
  return res_metric

def graph_weight(G):
  wght = 0
  for e in G.edges():
    u, v = e
    wght += G[u][v]['weight']
  return wght

def ccw(ax, ay, bx, by, cx, cy):
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

def get_crossings(G, pos):
  # --- Build coordinate arrays ---
  edges = np.array(list(G.edges()))
  E = len(edges)
  x1 = np.array([pos[u][0] for u, v in edges])
  y1 = np.array([pos[u][1] for u, v in edges])
  x2 = np.array([pos[v][0] for u, v in edges])
  y2 = np.array([pos[v][1] for u, v in edges])

  # --- Generate all unique edge pairs ---
  i_idx, j_idx = np.triu_indices(E, k=1)

  # Filter out pairs that share a vertex
  shared_vertex = (
      (edges[i_idx, 0] == edges[j_idx, 0]) |
      (edges[i_idx, 0] == edges[j_idx, 1]) |
      (edges[i_idx, 1] == edges[j_idx, 0]) |
      (edges[i_idx, 1] == edges[j_idx, 1])
  )
  i_idx, j_idx = i_idx[~shared_vertex], j_idx[~shared_vertex]

  # --- Get coordinates for all pairs ---
  x1_i, y1_i, x2_i, y2_i = x1[i_idx], y1[i_idx], x2[i_idx], y2[i_idx]
  x1_j, y1_j, x2_j, y2_j = x1[j_idx], y1[j_idx], x2[j_idx], y2[j_idx]

  ccw1 = ccw(x1_i, y1_i, x1_j, y1_j, x2_j, y2_j)
  ccw2 = ccw(x2_i, y2_i, x1_j, y1_j, x2_j, y2_j)
  ccw3 = ccw(x1_i, y1_i, x2_i, y2_i, x1_j, y1_j)
  ccw4 = ccw(x1_i, y1_i, x2_i, y2_i, x2_j, y2_j)

  intersects = (ccw1 != ccw2) & (ccw3 != ccw4)

  # --- Extract the crossing pairs ---
  crossings = np.column_stack((i_idx[intersects], j_idx[intersects]))
  cross_pairs = [(tuple(edges[i]), tuple(edges[j])) for i, j in crossings]

  print(f"Found {len(cross_pairs)} crossings (vectorized CPU version)")
  return crossings, cross_pairs