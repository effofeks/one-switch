import matplotlib.pyplot as plt
import networkx as nx
import io
import base64
import pandas as pd
import streamlit as st


def fig_to_image(fig):
    """Convert a matplotlib figure to an image for Streamlit"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf


def anim_to_html(anim):
    """Convert animation to HTML for display in Streamlit"""
    if anim is None:
        return None

    try:
        import tempfile
        import os
        
        # Create a temporary file with a .gif extension
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Save the animation to the temporary file
        anim.save(temp_path, writer='pillow', fps=1)
        
        # Read the file into memory
        with open(temp_path, 'rb') as f:
            data = f.read()
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        # Convert to base64
        encoded = base64.b64encode(data).decode('utf-8')
        
        # Create HTML
        return f'<img src="data:image/gif;base64,{encoded}" alt="Network Animation">'
    except Exception as e:
        import traceback
        print(f"Error converting animation to HTML: {str(e)}")
        print(traceback.format_exc())
        return None


def get_strong_connection_focal_info(agg_df, rank=1, measure="containers"):
    """Extract the focal edge information from the strong connections data"""
    # Check if agg_df is None
    if agg_df is None:
        return None
        
    # Prepare the weighted edges dataframe - must exactly match the logic in visualize_strong_connections
    if measure == "containers":
        weighted_edges_df = (
            agg_df.dropna(subset=["buyer_id", "seller_id", "container_number"])
            .groupby(["buyer_id", "seller_id"])["container_number"]
            .nunique()  # Count distinct containers
            .reset_index()
            .rename(columns={"container_number": "weight"})
            .sort_values("weight", ascending=False)
        )
    elif measure == "std_cartons":
        weighted_edges_df = (
            agg_df.dropna(subset=["buyer_id", "seller_id", "std_cartons"])
            .groupby(["buyer_id", "seller_id"])["std_cartons"]
            .sum()
            .reset_index()
            .rename(columns={"std_cartons": "weight"})
            .sort_values("weight", ascending=False)
        )
    else:  # measure == 'revenue'
        weighted_edges_df = (
            agg_df.dropna(subset=["buyer_id", "seller_id", "income"])
            .groupby(["buyer_id", "seller_id"])["income"]
            .sum()
            .reset_index()
            .rename(columns={"income": "weight"})
            .sort_values("weight", ascending=False)
        )

    # Check if rank is valid given the data
    if rank > len(weighted_edges_df) or weighted_edges_df.empty:
        return None

    # Get the focal edge (rank is 1-indexed)
    focal_edge = weighted_edges_df.iloc[rank - 1]
    return {
        "node1": focal_edge["buyer_id"],
        "node2": focal_edge["seller_id"],
        "edge_weight": focal_edge["weight"],
    }


def create_weighted_graph(agg_df, measure="containers"):
    """Create a weighted graph from network data based on the specified measure."""
    # Create an undirected graph for ALL measures
    G = nx.Graph()

    # Prepare edges based on measure
    if measure == "containers":
        edges_df = (
            agg_df.dropna(subset=["buyer_id", "seller_id", "container_number"])
            .groupby(["buyer_id", "seller_id"])["container_number"]
            .nunique()
            .reset_index()
            .rename(columns={"container_number": "weight"})
        )
    elif measure == "std_cartons":
        edges_df = (
            agg_df.dropna(subset=["buyer_id", "seller_id", "std_cartons"])
            .groupby(["buyer_id", "seller_id"])["std_cartons"]
            .sum()
            .reset_index()
            .rename(columns={"std_cartons": "weight"})
        )
    else:  # revenue
        edges_df = (
            agg_df.dropna(subset=["buyer_id", "seller_id", "income"])
            .groupby(["buyer_id", "seller_id"])["income"]
            .sum()
            .reset_index()
            .rename(columns={"income": "weight"})
        )

    # Add all edges to the graph
    for _, row in edges_df.iterrows():
        G.add_edge(row["buyer_id"], row["seller_id"], weight=row["weight"])

    return G


def format_network_stats(G, viz_type, params=None, focal_info=None, full_graph=None):
    """Format network statistics into a structured display for Streamlit"""
    if G is None and viz_type not in ["Heatmap", "Packing Week Heatmap", "Concentration Bubble Plot", "heatmap", "packing_week_heatmap", "concentration_bubble"]:
        return None

    stats = {}

    # Map new visualization types to old ones for backward compatibility
    viz_type_map = {
        "relationship": "Strong Connections",
        "company_network": "Company Network",
        "network_timelapse": "Network Timelapse"
    }
    
    # Convert new viz_type to old format if needed
    display_viz_type = viz_type_map.get(viz_type, viz_type)

    # Additional stats depending on visualization type
    if display_viz_type == "Strong Connections":
        # Get parameters
        rank = params.get("rank", 1)
        measure = params.get("measure", "containers")

        # Define weight label based on measure
        weight_label = {
            "containers": "containers",
            "std_cartons": "cartons",
            "revenue": "revenue",
        }.get(measure, measure)

        # Use focal info that was stored when creating the visualization
        if focal_info:
            node1 = focal_info.get("node1")
            node2 = focal_info.get("node2")
            edge_weight = focal_info.get("edge_weight")

            if node1 and node2 and edge_weight is not None:
                # Format company IDs as integers
                try:
                    formatted_node1 = str(int(float(node1)))
                    formatted_node2 = str(int(float(node2)))
                except (ValueError, TypeError):
                    # If conversion fails, use the original values
                    formatted_node1 = str(node1)
                    formatted_node2 = str(node2)

                # Get direct neighbors for each node
                node1_neighbors = list(G.neighbors(node1))
                node2_neighbors = list(G.neighbors(node2))

                # Calculate total weight for each node
                node1_total_weight = sum(
                    G[node1][neighbor].get("weight", 0)
                    for neighbor in node1_neighbors
                )
                node2_total_weight = sum(
                    G[node2][neighbor].get("weight", 0)
                    for neighbor in node2_neighbors
                )

                # Format weights for display
                if measure == "revenue":
                    # Use ZAR prefix and no cents for revenue
                    formatted_edge_weight = f"ZAR {int(edge_weight):,}"
                    formatted_node1_weight = f"ZAR {int(node1_total_weight):,}"
                    formatted_node2_weight = f"ZAR {int(node2_total_weight):,}"
                else:
                    formatted_edge_weight = f"{edge_weight:,}"
                    formatted_node1_weight = f"{node1_total_weight:,}"
                    formatted_node2_weight = f"{node2_total_weight:,}"

                # Create the overview text in the requested format
                overview_text = f"""
**Rank {rank}**: `{formatted_node1}` and `{formatted_node2}`

**Edge weight** ({measure}): {formatted_edge_weight}

Company `{formatted_node1}`
- Direct connections: {len(node1_neighbors)}
- Total edge weights ({weight_label}): {formatted_node1_weight}

Company `{formatted_node2}`:
- Direct connections: {len(node2_neighbors)}
- Total edge weights ({weight_label}): {formatted_node2_weight}
                """

                # Store the text in stats
                stats["Network Overview"] = overview_text
            else:
                stats["Network Overview"] = (
                    "Could not identify focal nodes for the selected rank."
                )
        else:
            # Fallback if no focal info provided
            stats["Network Overview"] = (
                "Detailed connection information is not available."
            )

        # Include parameters used
        if params:
            stats["Parameters"] = {
                "Rank": str(params.get("rank")),
                "Measure": str(params.get("measure")),
                "Degree": str(params.get("degree")),
                "Layout": str(params.get("layout")),
            }

    elif display_viz_type == "Company Network":
        # Get parameters
        measure = params.get("measure", "containers")
        focal_company = params.get("company_id") if params else None

        # Define weight label based on measure
        weight_label = {
            "containers": "containers",
            "std_cartons": "cartons",
            "revenue": "revenue",
        }.get(measure, measure)

        # Basic network statistics
        stats["Network Overview"] = {
            "Nodes": str(len(G.nodes)),
            "Edges": str(len(G.edges)),
        }

        # For company network, show centrality measures
        if len(G.nodes) > 0 and focal_company and focal_company in G.nodes:
            try:
                # Format focal company ID as integer
                try:
                    formatted_focal_company = str(int(float(focal_company)))
                except (ValueError, TypeError):
                    # If conversion fails, use the original value
                    formatted_focal_company = str(focal_company)

                # USE THE FULL GRAPH for getting all connections, not just the visualization subgraph
                active_graph = (
                    full_graph if full_graph and focal_company in full_graph else G
                )

                # Direct connections and weights
                direct_neighbors = list(active_graph.neighbors(focal_company))

                # Calculate total weight FROM THE FULL GRAPH
                total_weight = sum(
                    active_graph[focal_company][neighbor].get("weight", 0)
                    for neighbor in direct_neighbors
                )

                # Format total weight based on measure
                if measure == "revenue":
                    # For revenue, use ZAR prefix and no cents
                    formatted_total_weight = f"ZAR {int(total_weight):,}"
                else:
                    # For containers or cartons, use comma as thousands separator
                    formatted_total_weight = f"{int(total_weight):,}"

                # Get the top 5 strongest connections from the full graph
                connection_weights = []
                for neighbor in direct_neighbors:
                    weight = active_graph[focal_company][neighbor].get("weight", 0)
                    connection_weights.append((neighbor, weight))

                # Sort by weight in descending order
                connection_weights.sort(key=lambda x: x[1], reverse=True)

                # Detailed focal company information
                focal_company_text = f"""
**Company `{formatted_focal_company}`**

Direct connections: {len(direct_neighbors)}

Total edge weight ({weight_label}): {formatted_total_weight}

Top 5 strongest connections ({weight_label}):
"""

                # Add the top 5 connections (or fewer if there aren't 5)
                for i, (neighbor, weight) in enumerate(connection_weights[:5], 1):
                    # Format neighbor ID as integer
                    try:
                        formatted_neighbor = str(int(float(neighbor)))
                    except (ValueError, TypeError):
                        # If conversion fails, use the original value
                        formatted_neighbor = str(neighbor)

                    if measure == "revenue":
                        # For revenue, use ZAR prefix and no cents
                        formatted_weight = f"ZAR {int(weight):,}"
                    else:
                        # For containers or cartons, use comma as thousands separator and whole numbers
                        formatted_weight = f"{int(weight):,}"

                    focal_company_text += (
                        f" - Company `{formatted_neighbor}`: {formatted_weight}\n"
                    )

                # Store the text in stats
                stats["Network Overview"] = focal_company_text

            except Exception as e:
                stats["Network Overview"] = (
                    f"Error calculating detailed statistics: {str(e)}"
                )

        # Include parameters used
        if params:
            stats["Parameters"] = {
                "Company ID": str(params.get("company_id")),
                "Measure": str(params.get("measure")),
                "Degree": str(params.get("degree")),
                "Layout": str(params.get("layout")),
            }

    elif display_viz_type == "Network Timelapse":
        # Basic network statistics
        stats["Network Overview"] = {
            "Nodes": str(len(G.nodes)),
            "Edges": str(len(G.edges)),
        }

        # For timelapse, include temporal stats if available
        if hasattr(G, "graph") and "time_periods" in G.graph:
            stats["Temporal Information"] = {
                "Time Periods": str(G.graph["time_periods"]),
                "Start Period": str(G.graph.get("start_period")),
                "End Period": str(G.graph.get("end_period")),
            }

        # Include parameters used
        if params:
            # Handle both old and new parameter formats
            if "interval" in params:
                stats["Parameters"] = {
                    "Interval": f"{params.get('interval')}ms",
                    "FPS": str(params.get("fps")),
                    "Random Seed": str(params.get("seed")),
                }
            else:
                stats["Parameters"] = {
                    "Measure": str(params.get("measure")),
                    "Time Window": str(params.get("time_window")),
                    "Min Weight": str(params.get("min_weight")),
                    "Layout": str(params.get("layout")),
                }

    elif display_viz_type in ["Heatmap", "heatmap"]:
        # Include parameters used
        if params:
            stats["Parameters"] = {
                "Row Variable": params.get("row_col", params.get("y_axis", "")).replace("_", " ").title(),
                "Column Variable": params.get("col_col", params.get("x_axis", "")).replace("_", " ").title(),
                "Measure": params.get("measure", params.get("color_by", "containers")).title(),
                "Normalize Rows": "Yes" if params.get("normalize_rows") else "No",
                "Color Map": params.get("cmap", "YlGnBu"),
                "Significance Level": str(params.get("significance_level", 0.05)),
            }
            
    elif display_viz_type in ["Packing Week Heatmap", "packing_week_heatmap"]:
        # Include parameters used
        if params:
            stats["Parameters"] = {
                "Column Variable": params.get("col_col", params.get("y_axis", "")).replace("_", " ").title(),
                "Measure": params.get("measure", params.get("color_by", "containers")).title(),
                "Normalize Rows": "Yes" if params.get("normalize_rows") else "No",
                "Color Map": params.get("cmap", "YlGnBu"),
                "Significance Level": str(params.get("significance_level", 0.05)),
                "Top N Columns": str(params.get("top_n", 15)),
            }
            
    elif display_viz_type in ["Concentration Bubble Plot", "concentration_bubble"]:
        # Include parameters used
        if params:
            stats["Parameters"] = {
                "Primary Variable": params.get("entity1_col", "").replace("_", " ").title(),
                "Secondary Variable": params.get("entity2_col", "").replace("_", " ").title(),
                "Measure": params.get("measure", params.get("color_by", "containers")).title(),
                "Minimum Pallets": str(params.get("min_pallets", 0)),
                "Label Threshold": str(params.get("label_threshold", 0)),
            }
            
            # Add an explanation of HHI
            stats["What is HHI?"] = """
The **Herfindahl-Hirschman Index (HHI)** is a measure of market concentration.

- **Raw HHI** is the sum of squared market shares (0 to 1)
- **Normalized HHI** adjusts for the number of market participants (0 to 1)
  - 0 = Perfect competition (equal distribution)
  - 1 = Complete concentration (monopoly)
            
The bubble size represents the number of containers, and the color represents 
the percentage of containers going to the top participant.
            """

    return stats


def display_network_stats(stats):
    """Display the formatted network statistics in Streamlit.
    This function works the same as display_visualization_stats for backward compatibility.
    """
    # Simply call the generalized function
    display_visualization_stats(stats)


def format_heatmap_stats(data, viz_type, params=None):
    """Format heatmap statistics into a structured display for Streamlit"""
    if data is None:
        return None
    
    stats = {}
    
    # Common heatmap statistics
    x_axis = params.get("x_axis", "")
    y_axis = params.get("y_axis", "")
    color_by = params.get("color_by", "")
    
    # Basic data overview
    stats["Data Overview"] = {
        "Data Points": str(len(data)),
        "X-axis": x_axis,
        "Y-axis": y_axis,
        "Color by": color_by,
    }
    
    # Additional stats depending on heatmap type
    if viz_type == "heatmap":
        # Extract specific heatmap statistics like min/max values
        if "value" in data.columns:
            min_value = data["value"].min()
            max_value = data["value"].max()
            avg_value = data["value"].mean()
            
            # Format values for display
            if color_by == "revenue":
                formatted_min = f"ZAR {int(min_value):,}"
                formatted_max = f"ZAR {int(max_value):,}"
                formatted_avg = f"ZAR {int(avg_value):,}"
            else:
                formatted_min = f"{min_value:,.2f}"
                formatted_max = f"{max_value:,.2f}"
                formatted_avg = f"{avg_value:,.2f}"
            
            stats["Value Range"] = {
                "Minimum": formatted_min,
                "Maximum": formatted_max,
                "Average": formatted_avg,
            }
    
    elif viz_type == "packing_week_heatmap":
        # Extract specific statistics for packing week heatmaps
        stats["Time Range"] = {
            "Start Week": str(data[x_axis].min()) if x_axis in data.columns else "N/A",
            "End Week": str(data[x_axis].max()) if x_axis in data.columns else "N/A",
            "Total Weeks": str(data[x_axis].nunique()) if x_axis in data.columns else "N/A",
        }
        
        # Count unique categories on y-axis
        if y_axis in data.columns:
            stats["Categories"] = {
                "Unique Items": str(data[y_axis].nunique()),
                "Top Items": ", ".join(data[y_axis].value_counts().nlargest(5).index.tolist()),
            }
    
    # Include parameters used
    if params:
        filtered_params = {k: v for k, v in params.items() if v is not None and k not in ["x_axis", "y_axis", "color_by"]}
        if filtered_params:
            stats["Additional Parameters"] = filtered_params
    
    return stats


def format_bubble_stats(data, params=None):
    """Format bubble plot statistics into a structured display for Streamlit"""
    if data is None:
        return None
    
    stats = {}
    
    # Extract parameters
    x_axis = params.get("x_axis", "")
    y_axis = params.get("y_axis", "")
    size_by = params.get("size_by", "")
    color_by = params.get("color_by", "")
    
    # Basic data overview
    stats["Data Overview"] = {
        "Data Points": str(len(data)),
        "X-axis": x_axis,
        "Y-axis": y_axis,
        "Size by": size_by,
        "Color by": color_by,
    }
    
    # Generate statistics for the axes
    for axis_name, axis_col in [("X-axis", x_axis), ("Y-axis", y_axis)]:
        if axis_col in data.columns and data[axis_col].dtype.kind in 'ifc':  # if numeric
            min_val = data[axis_col].min()
            max_val = data[axis_col].max()
            mean_val = data[axis_col].mean()
            
            stats[f"{axis_name} Statistics"] = {
                "Minimum": f"{min_val:,.2f}",
                "Maximum": f"{max_val:,.2f}",
                "Average": f"{mean_val:,.2f}",
            }
    
    # Statistics for the size dimension
    if size_by in data.columns and data[size_by].dtype.kind in 'ifc':
        min_size = data[size_by].min()
        max_size = data[size_by].max()
        mean_size = data[size_by].mean()
        
        stats["Size Dimension"] = {
            "Minimum": f"{min_size:,.2f}",
            "Maximum": f"{max_size:,.2f}",
            "Average": f"{mean_size:,.2f}",
        }
    
    # Statistics for color categories if categorical
    if color_by in data.columns and data[color_by].dtype.kind not in 'ifc':
        unique_colors = data[color_by].nunique()
        top_categories = data[color_by].value_counts().nlargest(5)
        
        stats["Color Categories"] = {
            "Unique Categories": str(unique_colors),
            "Top Categories": ", ".join(top_categories.index.tolist()),
        }
    
    # Include parameters used
    if params:
        filtered_params = {k: v for k, v in params.items() 
                          if v is not None and k not in ["x_axis", "y_axis", "size_by", "color_by"]}
        if filtered_params:
            stats["Additional Parameters"] = filtered_params
    
    return stats


def display_visualization_stats(stats):
    """Display the formatted visualization statistics in Streamlit.
    This is a generic version of display_network_stats that can handle any visualization type.
    """
    if not stats:
        st.info("No statistics available for this visualization.")
        return

    # Display top-level items
    for section, content in stats.items():
        if isinstance(content, dict):
            # If the content is a dictionary, display as a table
            st.subheader(section)
            for key, value in content.items():
                st.text(f"{key}: {value}")
        elif isinstance(content, list):
            # If the content is a list, display as bullet points
            st.subheader(section)
            for item in content:
                st.markdown(f"- {item}")
        else:
            # If the content is a string, display as markdown
            st.subheader(section)
            st.markdown(content)


def format_stats(visualization):
    """
    Format statistics for any visualization type using the unified visualization structure.
    
    Parameters:
    -----------
    visualization : dict
        The unified visualization structure from st.session_state.visualization
        
    Returns:
    --------
    dict or None
        Formatted statistics or None if not available
    """
    if not visualization or not visualization["type"]:
        return None
        
    viz_type = visualization["type"]
    params = visualization["params"]
    result_df = visualization["result_df"]
    graph = visualization["graph"]
    metadata = visualization["metadata"]
    
    # Route to the appropriate formatter based on visualization type
    if viz_type in ["relationship", "company_network", "network_timelapse"]:
        focal_info = metadata.get("focal_info")
        full_graph = metadata.get("full_graph")
        return format_network_stats(graph, viz_type, params, focal_info, full_graph)
    
    elif viz_type in ["heatmap", "packing_week_heatmap"]:
        return format_heatmap_stats(result_df, viz_type, params)
    
    elif viz_type == "concentration_bubble":
        return format_bubble_stats(result_df, params)
    
    return None 