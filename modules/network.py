import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def visualize_strong_connections(
    df,
    rank=1,
    measure="containers",
    degree=1,
    layout="spring",
    spacing=1.0,
    iterations=100,
    figsize=(14, 10),
):
    """
    Visualize nodes with strong edge connections and their neighbors.

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the network data
    rank : int
        The rank of the edge to focus on (1 = strongest, 2 = second strongest, etc.)
    measure : str
        The measure to use for edge weights: 'containers', 'std_cartons', or 'revenue'
    degree : int
        The degree of neighbors to include (1 = direct neighbors, 2 = neighbors of neighbors, etc.)
    layout : str
        The layout algorithm to use: 'spring', 'kamada_kawai', 'fruchterman_reingold', 'spectral', or 'circular'
    spacing : float
        A multiplier for node spacing (higher values spread nodes further apart)
    iterations : int
        Number of iterations for force-directed layouts (higher values may give better results but take longer)
    figsize : tuple
        Figure size as (width, height) in inches

    Returns:
    --------
    None, displays a plot
    """

    # Validate measure parameter
    if measure not in ["containers", "std_cartons", "revenue"]:
        raise ValueError(
            "measure must be either 'containers', 'std_cartons', or 'revenue'"
        )

    # Validate rank parameter
    if not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer")

    # Prepare the weighted edges dataframe
    if measure == "containers":
        weighted_edges_df = (
            df.dropna(subset=["buyer_id", "seller_id", "container_number"])
            .groupby(["buyer_id", "seller_id"])["container_number"]
            .nunique()  # Count distinct containers
            .reset_index()
            .rename(columns={"container_number": "weight"})
            .sort_values("weight", ascending=False)
        )
        weight_label = "containers"
    elif measure == "std_cartons":  # measure == 'std_cartons'
        weighted_edges_df = (
            df.dropna(subset=["buyer_id", "seller_id", "std_cartons"])
            .groupby(["buyer_id", "seller_id"])["std_cartons"]
            .sum()
            .reset_index()
            .rename(columns={"std_cartons": "weight"})
            .sort_values("weight", ascending=False)
        )
        weight_label = "std_cartons"
    else:  # measure == 'revenue'
        weighted_edges_df = (
            df.dropna(subset=["buyer_id", "seller_id", "income"])
            .groupby(["buyer_id", "seller_id"])["income"]
            .sum()
            .reset_index()
            .rename(columns={"income": "weight"})
            .sort_values("weight", ascending=False)
        )
        weight_label = "revenue"

    # Check if rank is valid given the data
    if rank > len(weighted_edges_df):
        raise ValueError(
            f"rank ({rank}) exceeds number of edges ({len(weighted_edges_df)})"
        )

    # # Display the top pairs
    # print(f"Top 10 Strongest Buyer-Seller Connections by {weight_label.capitalize()}:")
    # print(weighted_edges_df.head(10))

    # Create a weighted graph
    weighted_G = nx.Graph()

    # Add weighted edges to the graph
    for _, row in weighted_edges_df.iterrows():
        weighted_G.add_edge(row["buyer_id"], row["seller_id"], weight=row["weight"])

    # Find the edge with the specified rank
    sorted_edges = sorted(
        weighted_G.edges(data=True), key=lambda x: x[2]["weight"], reverse=True
    )
    if rank <= len(sorted_edges):
        target_edge = sorted_edges[rank - 1]  # -1 because rank starts at 1
        node1, node2 = target_edge[0], target_edge[1]
        edge_weight = target_edge[2]["weight"]
    else:
        print(f"Rank {rank} exceeds the number of edges in the graph.")
        return

    # print(f"\nRank {rank} strongest connection: {node1} and {node2}")
    # print(f"{weight_label.capitalize()}: {edge_weight}")

    # Get nodes to include in the subgraph based on degree
    subgraph_nodes = {node1, node2}
    current_frontier = {node1, node2}

    for d in range(degree):
        next_frontier = set()
        for node in current_frontier:
            next_frontier.update(weighted_G.neighbors(node))
        subgraph_nodes.update(next_frontier)
        current_frontier = next_frontier

    # Create the subgraph
    subgraph = weighted_G.subgraph(subgraph_nodes)

    # Plot the subgraph
    plt.figure(figsize=figsize)

    # Choose layout based on parameter
    if layout == "spring":
        pos = nx.spring_layout(subgraph, seed=42, k=spacing, iterations=iterations)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(subgraph)
    elif layout == "fruchterman_reingold":
        pos = nx.fruchterman_reingold_layout(
            subgraph, seed=42, k=spacing, iterations=iterations
        )
    elif layout == "spectral":
        pos = nx.spectral_layout(subgraph)
    elif layout == "circular":
        pos = nx.circular_layout(subgraph)
    else:
        # Default to spring layout
        pos = nx.spring_layout(subgraph, seed=42, k=spacing, iterations=iterations)

    # Apply spacing factor
    if layout != "kamada_kawai":  # Kamada-Kawai already handles spacing well
        pos = {
            node: (coords[0] * spacing, coords[1] * spacing)
            for node, coords in pos.items()
        }

    # Adjust positions to emphasize the main nodes
    # Move the two main nodes slightly apart from each other
    if node1 in pos and node2 in pos:
        # Calculate the midpoint between the two nodes
        mid_x = (pos[node1][0] + pos[node2][0]) / 2
        mid_y = (pos[node1][1] + pos[node2][1]) / 2

        # Calculate the vector from midpoint to each node
        vec1_x = pos[node1][0] - mid_x
        vec1_y = pos[node1][1] - mid_y
        vec2_x = pos[node2][0] - mid_x
        vec2_y = pos[node2][1] - mid_y

        # Normalize and scale the vectors
        magnitude1 = np.sqrt(vec1_x**2 + vec1_y**2) or 1.0  # Avoid division by zero
        magnitude2 = np.sqrt(vec2_x**2 + vec2_y**2) or 1.0

        emphasis_factor = 0.3  # Adjust this to control how much to emphasize
        pos[node1] = (
            mid_x + (vec1_x / magnitude1) * emphasis_factor,
            mid_y + (vec1_y / magnitude1) * emphasis_factor,
        )
        pos[node2] = (
            mid_x + (vec2_x / magnitude2) * emphasis_factor,
            mid_y + (vec2_y / magnitude2) * emphasis_factor,
        )

    # Get edge weights for line thickness
    max_edge_weight = weighted_edges_df["weight"].max()
    edge_weights = [
        subgraph[u][v]["weight"] / max_edge_weight * 10 for u, v in subgraph.edges()
    ]

    # Determine node colors and sizes
    node_colors = []
    node_sizes = []

    for node in subgraph.nodes():
        # Set node size based on importance
        if node in [node1, node2]:
            node_sizes.append(1000)  # Main nodes are larger
        else:
            # Calculate distance from main nodes
            distance_to_main = min(
                nx.shortest_path_length(
                    subgraph, source=node, target=node1, weight=None
                ),
                nx.shortest_path_length(
                    subgraph, source=node, target=node2, weight=None
                ),
            )
            node_sizes.append(
                800 - (distance_to_main * 200)
            )  # Size decreases with distance

        # Set node color based on type and importance
        if node in [node1, node2]:
            node_colors.append("gold")  # Main nodes are gold
        else:
            if (
                node in weighted_edges_df["buyer_id"].values
                and node in weighted_edges_df["seller_id"].values
            ):
                node_colors.append("purple")  # Both buyer and seller
            elif node in weighted_edges_df["buyer_id"].values:
                node_colors.append("red")  # Buyer only
            else:
                node_colors.append("green")  # Seller only

    # Draw the graph with weighted edges
    nx.draw(
        subgraph,
        pos,
        with_labels=True,
        node_size=node_sizes,
        node_color=node_colors,
        font_size=10,
        font_weight="bold",
        # width=edge_weights,
        alpha=0.8,
    )

    # Highlight the target edge
    target_edge_list = [(node1, node2)]
    nx.draw_networkx_edges(
        subgraph, pos, edgelist=target_edge_list, width=10, edge_color="red", alpha=0.7
    )

    # Add edge labels with weights
    edge_labels = {
        (u, v): f"{int(d['weight']):,}".replace(",", " ")
        for u, v, d in subgraph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=8)

    # Update title with formatted weight
    formatted_weight = f"{int(edge_weight):,}".replace(",", " ")
    plt.title(
        f"Rank {rank} Connection ({node1}-{node2}) with Degree {degree} Neighbors\n"
        f"Weight = {formatted_weight} {weight_label}"
    )
    plt.show()

    # Additional information about the two main nodes
    # print("\nDetails about the main nodes:")
    node1_buyer = node1 in weighted_edges_df["buyer_id"].values
    node1_seller = node1 in weighted_edges_df["seller_id"].values
    node2_buyer = node2 in weighted_edges_df["buyer_id"].values
    node2_seller = node2 in weighted_edges_df["seller_id"].values

    node1_type = []
    if node1_buyer:
        node1_type.append("Buyer")
    if node1_seller:
        node1_type.append("Seller")

    node2_type = []
    if node2_buyer:
        node2_type.append("Buyer")
    if node2_seller:
        node2_type.append("Seller")

    # print(f"{node1} type: {' & '.join(node1_type)}")
    # print(f"{node2} type: {' & '.join(node2_type)}")

    # # Calculate stats for the connected nodes
    # node1_neighbors = list(weighted_G.neighbors(node1))
    # node2_neighbors = list(weighted_G.neighbors(node2))

    # print(f"Number of {node1}'s direct connections: {len(node1_neighbors)}")
    # print(f"Number of {node2}'s direct connections: {len(node2_neighbors)}")

    # # Calculate total weight for each node
    # node1_total_weight = sum(
    #     weighted_G[node1][neighbor]["weight"] for neighbor in node1_neighbors
    # )
    # node2_total_weight = sum(
    #     weighted_G[node2][neighbor]["weight"] for neighbor in node2_neighbors
    # )

    # # Format total weights with thousand separators
    # formatted_node1_weight = f"{int(node1_total_weight):,}".replace(",", " ")
    # formatted_node2_weight = f"{int(node2_total_weight):,}".replace(",", " ")

    # print(f"\nTotal {weight_label} for {node1}: {formatted_node1_weight}")
    # print(f"Total {weight_label} for {node2}: {formatted_node2_weight}")

    # # Find common neighbors between the two nodes
    # common_neighbors = set(node1_neighbors).intersection(set(node2_neighbors))
    # print(f"\nNumber of common neighbors: {len(common_neighbors)}")
    # if common_neighbors and len(common_neighbors) <= 10:  # Only show if not too many
    #     print("Common neighbors:")
    #     for neighbor in common_neighbors:
    #         print(f"  - {neighbor}")
    # elif common_neighbors:
    #     print(f"Common neighbors (showing 10 of {len(common_neighbors)}):")
    #     for neighbor in list(common_neighbors)[:10]:
    #         print(f"  - {neighbor}")


def visualize_company_network(
    df,
    company_id,
    measure="containers",
    degree=1,
    layout="spring",
    spacing=1.5,
    iterations=100,
    figsize=(14, 10),
):
    """
    Visualize a network centered around a specific company (buyer or seller).

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the network data
    company_id : int or str
        The ID of the company to focus on (will match to buyer_id or seller_id)
    measure : str
        The measure to use for edge weights: 'containers', 'std_cartons', or 'revenue'
    degree : int
        The degree of neighbors to include (1 = direct neighbors, 2 = neighbors of neighbors, etc.)
    layout : str
        The layout algorithm to use: 'spring', 'kamada_kawai', 'fruchterman_reingold', 'spectral', or 'circular'
    spacing : float
        A multiplier for node spacing (higher values spread nodes further apart)
    iterations : int
        Number of iterations for force-directed layouts (higher values may give better results but take longer)
    figsize : tuple
        Figure size as (width, height) in inches

    Returns:
    --------
    None, displays a plot
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np

    # Validate measure parameter
    if measure not in ["containers", "std_cartons", "revenue"]:
        raise ValueError(
            "measure must be either 'containers', 'std_cartons', or 'revenue'"
        )

    # Check if company_id exists in the data
    is_buyer = company_id in df["buyer_id"].values
    is_seller = company_id in df["seller_id"].values

    if not (is_buyer or is_seller):
        raise ValueError(
            f"Company ID {company_id} not found as either buyer or seller in the data"
        )

    # Prepare the weighted edges dataframe
    if measure == "containers":
        weighted_edges_df = (
            df.dropna(subset=["buyer_id", "seller_id", "container_number"])
            .groupby(["buyer_id", "seller_id"])["container_number"]
            .nunique()  # Count distinct containers
            .reset_index()
            .rename(columns={"container_number": "weight"})
            .sort_values("weight", ascending=False)
        )
        weight_label = "containers"
    elif measure == "std_cartons":  # measure == 'std_cartons'
        weighted_edges_df = (
            df.dropna(subset=["buyer_id", "seller_id", "std_cartons"])
            .groupby(["buyer_id", "seller_id"])["std_cartons"]
            .sum()
            .reset_index()
            .rename(columns={"std_cartons": "weight"})
            .sort_values("weight", ascending=False)
        )
        weight_label = "std_cartons"
    else:  # measure == 'revenue'
        weighted_edges_df = (
            df.dropna(subset=["buyer_id", "seller_id", "income"])
            .groupby(["buyer_id", "seller_id"])["income"]
            .sum()
            .reset_index()
            .rename(columns={"income": "weight"})
            .sort_values("weight", ascending=False)
        )
        weight_label = "revenue"

    # Create a weighted graph
    weighted_G = nx.Graph()

    # Add weighted edges to the graph
    for _, row in weighted_edges_df.iterrows():
        weighted_G.add_edge(row["buyer_id"], row["seller_id"], weight=row["weight"])

    # Check if company_id exists in the graph
    if company_id not in weighted_G.nodes():
        raise ValueError(f"Company ID {company_id} has no connections in the graph")

    # Get nodes to include in the subgraph based on degree
    subgraph_nodes = {company_id}
    current_frontier = {company_id}

    for d in range(degree):
        next_frontier = set()
        for node in current_frontier:
            next_frontier.update(weighted_G.neighbors(node))
        subgraph_nodes.update(next_frontier)
        current_frontier = next_frontier

    # Create the subgraph
    subgraph = weighted_G.subgraph(subgraph_nodes)

    # Plot the subgraph
    plt.figure(figsize=figsize)

    # Choose layout based on parameter
    if layout == "spring":
        pos = nx.spring_layout(subgraph, seed=42, k=spacing, iterations=iterations)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(subgraph)
    elif layout == "fruchterman_reingold":
        pos = nx.fruchterman_reingold_layout(
            subgraph, seed=42, k=spacing, iterations=iterations
        )
    elif layout == "spectral":
        pos = nx.spectral_layout(subgraph)
    elif layout == "circular":
        pos = nx.circular_layout(subgraph)
    else:
        # Default to spring layout
        pos = nx.spring_layout(subgraph, seed=42, k=spacing, iterations=iterations)

    # Apply spacing factor
    if layout != "kamada_kawai":  # Kamada-Kawai already handles spacing well
        pos = {
            node: (coords[0] * spacing, coords[1] * spacing)
            for node, coords in pos.items()
        }

    # Get edge weights for line thickness
    max_edge_weight = weighted_edges_df["weight"].max()
    edge_weights = [
        subgraph[u][v]["weight"] / max_edge_weight * 10 for u, v in subgraph.edges()
    ]

    # Determine node colors and sizes
    node_colors = []
    node_sizes = []

    for node in subgraph.nodes():
        # Set node size based on importance
        if node == company_id:
            node_sizes.append(1200)  # Main node is larger
        else:
            # Calculate distance from main node
            distance_to_main = nx.shortest_path_length(
                subgraph, source=node, target=company_id, weight=None
            )
            node_sizes.append(
                800 - (distance_to_main * 150)
            )  # Size decreases with distance

        # Set node color based on type and importance
        if node == company_id:
            node_colors.append("yellow")  # Main node is yellow as specified
        else:
            if (
                node in weighted_edges_df["buyer_id"].values
                and node in weighted_edges_df["seller_id"].values
            ):
                node_colors.append("purple")  # Both buyer and seller
            elif node in weighted_edges_df["buyer_id"].values:
                node_colors.append("red")  # Buyer only
            else:
                node_colors.append("green")  # Seller only

    # Draw the graph with weighted edges
    nx.draw(
        subgraph,
        pos,
        with_labels=True,
        node_size=node_sizes,
        node_color=node_colors,
        font_size=10,
        font_weight="bold",
        # width=edge_weights,
        alpha=0.8,
    )

    # Add edge labels with weights as integers with thousand separators
    edge_labels = {
        (u, v): f"{int(d['weight']):,}".replace(",", " ")
        for u, v, d in subgraph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=8)

    # Create a title for the plot
    company_type = []
    if is_buyer:
        company_type.append("Buyer")
    if is_seller:
        company_type.append("Seller")
    company_type_str = " & ".join(company_type)

    plt.title(
        f"Network for Company {company_id} ({company_type_str})\n"
        f"Showing Degree {degree} Neighbors with {weight_label.capitalize()} as Edge Weights"
    )

    # Add a legend
    import matplotlib.patches as mpatches

    legend_elements = [
        mpatches.Patch(color="yellow", label=f"Focus Company ({company_id})"),
        mpatches.Patch(color="red", label="Buyers"),
        mpatches.Patch(color="green", label="Sellers"),
        mpatches.Patch(color="purple", label="Both Buyer & Seller"),
    ]

    plt.legend(handles=legend_elements, loc="upper right")
    plt.show()

    # # Print network statistics
    # print(f"\nNetwork Statistics for Company {company_id} ({company_type_str}):")
    # print(f"Number of nodes in network: {len(subgraph.nodes())}")
    # print(f"Number of connections: {len(subgraph.edges())}")

    # Calculate direct connections
    direct_neighbors = list(weighted_G.neighbors(company_id))
    # direct_buyer_count = sum(1 for n in direct_neighbors if n in weighted_edges_df["buyer_id"].values)
    # direct_seller_count = sum(1 for n in direct_neighbors if n in weighted_edges_df["seller_id"].values)

    # print(f"\nDirect connections: {len(direct_neighbors)}")
    # print(f"  - Buyers: {direct_buyer_count}")
    # print(f"  - Sellers: {direct_seller_count}")

    # Calculate total weight
    total_weight = sum(
        weighted_G[company_id][neighbor]["weight"] for neighbor in direct_neighbors
    )
    # formatted_total_weight = f"{int(total_weight):,}".replace(",", " ")
    # print(f"\nTotal {weight_label}: {formatted_total_weight}")

    # Find strongest connections
    connection_weights = [
        (neighbor, weighted_G[company_id][neighbor]["weight"])
        for neighbor in direct_neighbors
    ]
    connection_weights.sort(key=lambda x: x[1], reverse=True)

    if connection_weights:
        # print(f"\nTop 5 strongest connections:")
        for i, (neighbor, weight) in enumerate(connection_weights[:5]):
            neighbor_type = []
            if neighbor in weighted_edges_df["buyer_id"].values:
                neighbor_type.append("Buyer")
            if neighbor in weighted_edges_df["seller_id"].values:
                neighbor_type.append("Seller")
            neighbor_type_str = " & ".join(neighbor_type)

            formatted_weight = f"{int(weight):,}".replace(",", " ")
            # print(f"{i+1}. {neighbor} ({neighbor_type_str}): {formatted_weight} {weight_label}")


def create_network_timelapse(
    df,
    interval=1500,
    figsize=(12, 8),
    seed=42,
):
    """
    Create an animation showing the network evolution over time by packing week.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing network data with columns: buyer_id, seller_id, packing_week, container_number
        Must be pre-filtered to remove null values.
    interval : int, default=1500
        Milliseconds between frames for the animation display
    figsize : tuple, default=(12, 8)
        Figure size as (width, height) in inches
    seed : int, default=42
        Random seed for layout generation

    Returns:
    --------
    animation : FuncAnimation
        The animation object that can be displayed in a notebook
    """

    # Get the unique weeks in sorted order
    unique_weeks = sorted(df["packing_week"].unique())

    # Create a function to build a graph for a specific week
    def build_graph_for_week(week):
        # Filter the DataFrame for the specific week
        week_df = df[df["packing_week"] == week]

        # Create a DataFrame with unique buyer-seller pairs for this week
        week_pairs = (
            week_df.groupby(["buyer_id", "seller_id"])["container_number"]
            .nunique()
            .reset_index()
        )

        # Create a graph
        G_week = nx.Graph()

        # Add edges to the graph
        for _, row in week_pairs.iterrows():
            G_week.add_edge(row["buyer_id"], row["seller_id"])

        return G_week, week_df

    # Build graphs for each week and keep the week dataframes
    week_data = {week: build_graph_for_week(week) for week in unique_weeks}
    week_graphs = {week: data[0] for week, data in week_data.items()}
    week_dfs = {week: data[1] for week, data in week_data.items()}

    # Function to get the largest connected component
    def get_largest_component(G):
        if len(G.nodes()) == 0:
            return nx.Graph()  # Return empty graph if input graph has no nodes

        connected_components = list(nx.connected_components(G))
        if not connected_components:
            return nx.Graph()  # Return empty graph if no connected components

        largest_component = max(connected_components, key=len)
        return G.subgraph(largest_component)

    # Get the largest connected component for each week
    largest_components = {
        week: get_largest_component(graph) for week, graph in week_graphs.items()
    }

    # Create a figure for animation
    fig, ax = plt.subplots(figsize=figsize)

    # Use a consistent layout across frames to maintain node positions
    # First, create a combined graph from all weeks to compute a consistent layout
    combined_graph = nx.Graph()
    for _, graph in week_graphs.items():
        combined_graph.add_edges_from(graph.edges())

    pos = nx.spring_layout(combined_graph, seed=seed)

    # Function to get weekly statistics
    def get_week_stats(graph, week_df):
        # Count unique sellers and buyers
        unique_sellers = week_df["seller_id"].nunique()
        unique_buyers = week_df["buyer_id"].nunique()

        # Find top seller and buyer by degree centrality
        if len(graph.nodes()) > 0:
            degree_centrality = nx.degree_centrality(graph)

            # Separate sellers and buyers
            seller_centrality = {
                node: degree_centrality[node]
                for node in graph.nodes()
                if node in week_df["seller_id"].values
            }
            buyer_centrality = {
                node: degree_centrality[node]
                for node in graph.nodes()
                if node in week_df["buyer_id"].values
            }

            # Find top seller and buyer
            top_seller = (
                max(seller_centrality.items(), key=lambda x: x[1])[0]
                if seller_centrality
                else "None"
            )
            top_buyer = (
                max(buyer_centrality.items(), key=lambda x: x[1])[0]
                if buyer_centrality
                else "None"
            )
        else:
            top_seller = "None"
            top_buyer = "None"

        # Find most popular variety if available
        if "variety_name" in week_df.columns and not week_df.empty:
            variety_counts = week_df["variety_name"].value_counts()
            most_popular_variety = (
                variety_counts.idxmax() if not variety_counts.empty else "Unknown"
            )
        else:
            most_popular_variety = "Unknown"

        # Count total containers
        total_containers = week_df["container_number"].nunique()

        return {
            "unique_sellers": unique_sellers,
            "unique_buyers": unique_buyers,
            "top_seller": top_seller,
            "top_buyer": top_buyer,
            "most_popular_variety": most_popular_variety,
            "total_containers": total_containers,
        }

    # Function to update the plot for each frame (week)
    def update(frame):
        week = unique_weeks[frame]
        ax.clear()

        G = largest_components[week]
        week_df = week_dfs[week]

        # Calculate weekly statistics
        stats = get_week_stats(week_graphs[week], week_df)

        # Use the pre-computed positions, filtering for nodes in the current graph
        current_pos = {node: pos[node] for node in G.nodes() if node in pos}

        # If a node doesn't have a pre-computed position, place it randomly
        for node in G.nodes():
            if node not in current_pos:
                current_pos[node] = np.random.rand(2)

        # Calculate node sizes based on degree
        node_sizes = [100 + 50 * G.degree(node) for node in G.nodes()]

        # Determine node colors: red for buyers, green for sellers
        node_colors = []
        for node in G.nodes():
            if node in df["buyer_id"].values and node in df["seller_id"].values:
                node_colors.append("purple")  # Both buyer and seller
            elif node in df["buyer_id"].values:
                node_colors.append("red")  # Buyer only
            else:
                node_colors.append("green")  # Seller only

        # Draw the graph
        nx.draw(
            G,
            pos=current_pos,
            with_labels=True,
            node_size=node_sizes,
            node_color=node_colors,
            font_size=8,
            font_weight="bold",
            edge_color="gray",
            ax=ax,
        )

        # Add statistics as text
        stats_text = (
            f"Week: {week}\n"
            f"Sellers: {stats['unique_sellers']}\n"
            f"Buyers: {stats['unique_buyers']}\n"
            f"Top Seller: {stats['top_seller']}\n"
            f"Top Buyer: {stats['top_buyer']}\n"
            f"Top Variety: {stats['most_popular_variety']}\n"
            f"Total Containers: {stats['total_containers']}"
        )

        # Add text box with stats
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        ax.set_title(f"Largest Connected Component - Weekk0 {week}")

        return (ax,)

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=len(unique_weeks), interval=interval, blit=False
    )

    # Close the static figure to prevent duplicate display
    plt.close()

    return ani
