#weight_attr = 'weights'
weight_attr = 'weight'

def build_sparse_graph(G_original, t):
    """
    Build a sparse graph by adding edges from G_original in increasing weight order.
    An edge (u, v) is only added if the shortest path from u to v in the current graph
    is greater than t times the edge's weight.

    Parameters:
        G_original (nx.Graph): Original undirected weighted graph.
        t (float): Stretch factor threshold.

    Returns:
        G_sparse (nx.Graph): Sparse graph built under the given rule.
    """
    # Sort edges by weight
    sorted_edges = sorted(G_original.edges(data=True), key=lambda x: float(x[2][weight_attr]))

    # Create an empty graph with the same nodes
    G_sparse = nx.Graph()
    G_sparse.add_nodes_from(G_original.nodes)

    # Process edges in sorted order
    for u, v, data in sorted_edges:
        weight = float(data[weight_attr])
        try:
            # Try to compute shortest path length between u and v
            current_dist = nx.dijkstra_path_length(G_sparse, u, v, weight='weight')
        except nx.NetworkXNoPath:
            # If no path exists, treat it as infinite
            current_dist = float('inf')

        # Add edge only if the shortest path is greater than t * weight
        if current_dist > t * weight:
            G_sparse.add_edge(u, v, weight=weight)
            #print(f"Added edge ({u}, {v}) with weight {weight}")
        #else:
        #    print(f"Skipped edge ({u}, {v}) — existing path length {current_dist} <= {t} * {weight}")

    return G_sparse