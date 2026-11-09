def compute_ideal_allreduce_time(collective_size: float, bw_matrix: list[list[float]], link_latency: float = 0.0) -> float:
    """
    Computes the theoretically ideal All-Reduce performance.

    Args:
        collective_size: The total size of the data payload to be reduced.
        bw_matrix: A 2D list (n x n) representing the network topology.
                   bw_matrix[i][j] is the bandwidth from device i to device j.
                   A value of 0 indicates no direct link.
        link_latency: The base latency per hop (added to the diameter).

    Returns:
        The ideal execution time for the collective operation.
    """
    n = len(bw_matrix)

    if n <= 1:
        return 0.0

    # 1. Calculate the Bottleneck Bandwidth of any NPU (min BW_N)
    # We define NPU bandwidth as the minimum of its total ingress or egress capacity.
    min_node_bw = float('inf')
    for i in range(n):
        node_egress = sum(bw_matrix[i][j] for j in range(n) if i != j)
        node_ingress = sum(bw_matrix[j][i] for j in range(n) if i != j)

        # Avoid counting completely disconnected nodes if they exist
        if node_egress > 0 and node_ingress > 0:
            min_node_bw = min(min_node_bw, node_egress, node_ingress)

    if min_node_bw == 0 or min_node_bw == float('inf'):
        raise ValueError("Network is disconnected or invalid bandwidths provided.")

    # 2. Calculate the Network Diameter (Max of All-Pairs Shortest Paths)
    # Initialize distance matrix: 1 hop for existing links, infinity otherwise
    dist = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0.0  # Distance to self is 0
        for j in range(n):
            if i != j and bw_matrix[i][j] > 0:
                dist[i][j] = 1.0

    # Floyd-Warshall Algorithm for shortest paths
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    # Find the maximum shortest path (Diameter)
    hop_diameter = 0.0
    for i in range(n):
        for j in range(n):
            if dist[i][j] != float('inf') and dist[i][j] > hop_diameter:
                hop_diameter = dist[i][j]

    if hop_diameter == float('inf'):
        raise ValueError("Network graph is not fully connected.")

    # Calculate total latency based on hops
    diameter_latency = hop_diameter * link_latency

    # 3. Calculate the Ideal Collective Time
    # Formula: (CollectiveSize * (2 * (n - 1) / n)) / min_BW + Diameter
    data_transfer_volume = collective_size * (1 * (n - 1) / n)
    serialization_delay = data_transfer_volume / min_node_bw

    ideal_time = serialization_delay + diameter_latency

    return ideal_time

# --- Example Usage ---
if __name__ == "__main__":
    # Example: 4 devices, Ring Topology, unidirectional
    # Bandwidths are in GB/s
    bw_matrix_ring = [
        [0, 100, 0, 0],   # Node 0 sends to Node 1
        [0, 0, 100, 0],   # Node 1 sends to Node 2
        [0, 0, 0, 100],   # Node 2 sends to Node 3
        [100, 0, 0, 0]    # Node 3 sends to Node 0
    ]

    data_size_gb = 2.0  # 2 GB of gradients
    per_hop_latency = 0.000001 # 1 microsecond per hop

    time_taken = compute_ideal_allreduce_time(data_size_gb, bw_matrix_ring, per_hop_latency)
    print(f"Ideal execution time: {time_taken:.6f} seconds")
