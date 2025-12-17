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

def edge_coloring(G, cross_pairs):
  start_time = time.time()
  unassigned = set(list(G.edges()))
  non_intersecting_sets = []
  crossings = set(cross_pairs)

  while unassigned:
      layer = set()
      for e in list(unassigned):
          # Check if e crosses any edge already in this layer
          if all(((e, other) not in crossings) for other in layer):
              layer.add(e)
      # Remove selected edges from unassigned
      unassigned -= layer
      non_intersecting_sets.append(layer)
  end_time = time.time()
  # --- Output results ---
  print(f"Number of non-intersecting sets: {len(non_intersecting_sets)}")
  for i, s in enumerate(non_intersecting_sets, 1):
      print(f"Set {i}: {len(s)} edges")
  print(f"Time taken: {end_time-start_time} seconds")

  return non_intersecting_sets

def direct_edge(e):
    """ Return a canonically ordered edge tuple so (u,v) == (v,u). """
    u, v = e
    return (u, v) if u <= v else (v, u)


def get_crossings_partial(G, pos, already_considered):
    """
    Same as get_crossings, but ignores all crossings involving edges in already_considered.
    already_considered: list or set of (u,v) tuples
    """
    already = {direct_edge(e) for e in already_considered}

    edges = np.array([direct_edge(e) for e in G.edges()])
    E = len(edges)

    # Determine which edges are allowed
    allowed_mask = np.array([direct_edge(tuple(e)) not in already for e in edges])
    allowed_idx = np.where(allowed_mask)[0]

    # If ≤1 edge remains, no crossings possible
    if len(allowed_idx) <= 1:
        return np.empty((0, 2), dtype=int), []

    # Build coordinate arrays
    x1 = np.array([pos[u][0] for u, v in edges])
    y1 = np.array([pos[u][1] for u, v in edges])
    x2 = np.array([pos[v][0] for u, v in edges])
    y2 = np.array([pos[v][1] for u, v in edges])

    # Generate all *unique* pairs among allowed edges
    allowed_pairs = [(i, j) for i in allowed_idx for j in allowed_idx if i < j]
    if not allowed_pairs:
        return np.empty((0, 2), dtype=int), []

    i_idx = np.array([p[0] for p in allowed_pairs])
    j_idx = np.array([p[1] for p in allowed_pairs])

    # Filter out pairs sharing a vertex
    shared_vertex = (
        (edges[i_idx, 0] == edges[j_idx, 0]) |
        (edges[i_idx, 0] == edges[j_idx, 1]) |
        (edges[i_idx, 1] == edges[j_idx, 0]) |
        (edges[i_idx, 1] == edges[j_idx, 1])
    )
    i_idx, j_idx = i_idx[~shared_vertex], j_idx[~shared_vertex]

    if len(i_idx) == 0:
        return np.empty((0, 2), dtype=int), []

    # Coordinates
    x1_i, y1_i, x2_i, y2_i = x1[i_idx], y1[i_idx], x2[i_idx], y2[i_idx]
    x1_j, y1_j, x2_j, y2_j = x1[j_idx], y1[j_idx], x2[j_idx], y2[j_idx]

    # Reuse your ccw formula but vectorized
    ccw1 = (y2_j - y1_i) * (x1_j - x1_i) > (y1_j - y1_i) * (x2_j - x1_i)
    ccw2 = (y2_j - y2_i) * (x1_j - x2_i) > (y1_j - y2_i) * (x2_j - x2_i)
    ccw3 = (y2_i - y1_i) * (x2_i - x1_i) > (y2_i - y1_i) * (x1_j - x1_i)
    ccw4 = (y2_i - y1_i) * (x2_i - x1_i) > (y2_j - y1_i) * (x1_j - x1_i)

    intersects = (ccw1 != ccw2) & (ccw3 != ccw4)

    crossings = np.column_stack((i_idx[intersects], j_idx[intersects]))
    cross_pairs = [(tuple(edges[i]), tuple(edges[j])) for i, j in crossings]

    return crossings, cross_pairs

def edge_coloring_progressive(G, cross_pairs, crossing_limits):
    """
    Progressive edge layering with dynamic relaxation:
    - Layer 1: 0-crossing edges
    - Layer i>1: begins as previous layer and adds edges whose crossings with the
      existing layer are <= limit_i
    - If no new edges are added, the crossing limit is doubled and retried
    - Guaranteed that the last layer contains all edges.
    """

    import time
    start_time = time.time()

    # Unassigned edges
    unassigned = set(G.edges())

    # Convert crossing pairs to lookup dict
    crossings = {}
    for e1, e2 in cross_pairs:
        crossings.setdefault(e1, set()).add(e2)
        crossings.setdefault(e2, set()).add(e1)

    def count_crossings(e, layer):
        return len(crossings.get(e, set()) & layer)

    layers = []
    layer_index = 0

    while unassigned:
        # Determine starting crossing limit for this layer
        if layer_index < len(crossing_limits):
            limit = crossing_limits[layer_index]
        else:
            limit = crossing_limits[-1]  # continue with last limit

        # Create layer
        if layer_index == 0:
            # Layer 1: classical non-crossing edges
            cur_layer = set()
            for e in list(unassigned):
                if count_crossings(e, cur_layer) == 0:
                    cur_layer.add(e)

            newly_added = cur_layer
            unassigned -= newly_added

            layers.append(cur_layer)
            layer_index += 1
            continue

        # Layer i > 1
        prev_layer = layers[-1]
        cur_layer = set(prev_layer)

        # --- Try to add edges. If none added, double limit and retry. ---
        while True:
            newly_added = set()
            for e in list(unassigned):
                if count_crossings(e, cur_layer) <= limit:
                    newly_added.add(e)

            if newly_added:
                # success
                cur_layer |= newly_added
                unassigned -= newly_added
                break
            else:
                # no edges can be added: relax constraint
                limit *= 2
                # loop and try again

        layers.append(cur_layer)
        layer_index += 1

    end_time = time.time()

    # --- Output ---
    print(f"Number of layers: {len(layers)}")
    for i, layer in enumerate(layers, 1):
        eff_lim = crossing_limits[min(i-1, len(crossing_limits)-1)]
        print(f"Layer {i}: {len(layer)} edges (starting limit={eff_lim})")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    return layers

def edge_coloring_MST(G, pos):
    """
    Partition edges of G into layers, where each layer is a Minimum Spanning Forest
    computed using weights derived from edge crossing ratios among unassigned edges.
    """
    edges = [direct_edge(e) for e in G.edges()]
    edges_set = set(edges)
    remaining = set(edges)
    partitions = []

    partition_id = 1

    while remaining:
        print(f"Computing partition {partition_id}...")

        # Compute crossings among remaining edges
        _, cross_pairs = get_crossings_partial(
            G, pos,
            already_considered=edges_set - remaining
        )

        # Count crossings for each remaining edge
        cross_count = {e: 0 for e in remaining}
        for e1, e2 in cross_pairs:
            if e1 in remaining:
                cross_count[e1] += 1
            if e2 in remaining:
                cross_count[e2] += 1

        total_crossings = len(cross_pairs)
        if total_crossings == 0:
            # No crossings → everything can go in one final forest
            partitions.append(list(remaining))
            break

        # Weight is ratio of crossings the edge participates in
        weights = {
            e: cross_count[e] / total_crossings
            for e in remaining
        }

        # Build weighted graph induced by remaining edges
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        for (u, v), w in weights.items():
            H.add_edge(u, v, weight=w)

        # Compute Minimum Spanning Forest
        forest_edges = []
        for comp in nx.connected_components(H):
            sub = H.subgraph(comp)
            T = nx.minimum_spanning_tree(sub, weight='weight')
            forest_edges.extend(direct_edge(e) for e in T.edges())

        partitions.append(forest_edges)

        # Remove selected edges from remaining
        for e in forest_edges:
            remaining.discard(e)

        partition_id += 1

    # -------------------------
    # PRINT PARTITION SIZES
    # -------------------------
    print("\n=== Partition Sizes ===")
    for i, part in enumerate(partitions):
        print(f"Partition {i} size: {len(part)}")
    print("=======================\n")

    return partitions

def edge_coloring_MST_weighted(G, pos, alpha=2.0, beta=1.0):
    """
    Partition edges of G into layers (forests) using weighted MST.
    Weight of each remaining edge is based on ALL crossings:
      - alpha for crossings with already-selected edges
      - beta  for crossings with not-yet-selected edges
    All weights are then normalized to [0,1].

    Returns: list of partitions (each partition = list of edge tuples)
    """
    edges = [direct_edge(e) for e in G.edges()]
    edges_set = set(edges)
    remaining = set(edges)
    partitions = []

    partition_id = 1

    while remaining:
        print(f"Computing weighted partition {partition_id}...")

        # ---- Determine which edges are already considered in previous partitions ----
        already_considered = edges_set - remaining
        already_arr = np.array([direct_edge(e) for e in already_considered])

        # ---- Compute ALL pairwise crossings (no ignoring) ----
        # Reuse get_crossings_partial with empty already_considered
        # since we want full crossings
        _, full_cross_pairs = get_crossings_partial(G, pos, already_considered=[])

        # ---- Initialize cross weights ----
        # For each remaining edge, how many α/β it receives
        weight_raw = {e: 0.0 for e in remaining}

        # Mapping edge -> for speed
        remaining_set = set(remaining)
        already_set = already_considered

        # ---- Accumulate weighted crossings ----
        for e1, e2 in full_cross_pairs:
            # Count contributions only for edges still eligible (remaining)
            e1_in = e1 in remaining_set
            e2_in = e2 in remaining_set

            if e1_in:
                # e1 crosses an already-selected edge
                if e2 in already_set:
                    weight_raw[e1] += alpha
                else:
                    weight_raw[e1] += beta

            if e2_in:
                if e1 in already_set:
                    weight_raw[e2] += alpha
                else:
                    weight_raw[e2] += beta

        # ---- Normalize weights for remaining edges ----
        max_w = max(weight_raw.values()) if weight_raw else 0
        if max_w < 1e-12:
            # No crossings → everything forms a final forest
            partitions.append(list(remaining))
            break

        norm_weights = {e: (w / max_w) for e, w in weight_raw.items()}

        # ---- Build graph induced by remaining edges ----
        H = nx.Graph()
        H.add_nodes_from(G.nodes())

        for (u, v), w in norm_weights.items():
            H.add_edge(u, v, weight=w)

        # ---- Compute minimum spanning forest ----
        forest_edges = []
        for comp in nx.connected_components(H):
            sub = H.subgraph(comp)
            T = nx.minimum_spanning_tree(sub, weight="weight")
            forest_edges.extend(direct_edge(e) for e in T.edges())

        partitions.append(forest_edges)

        # Remove these edges from remaining
        for e in forest_edges:
            remaining.discard(e)

        partition_id += 1

    # ---- Print partition summary ----
    print("\n=== Weighted Partition Sizes ===")
    for i, part in enumerate(partitions):
        print(f"Partition {i} size: {len(part)}")
    print("===============================\n")

    return partitions

