import matplotlib.pyplot as plt
import networkx as nx
import io
import base64
import pandas as pd
import streamlit as st
import numpy as np


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


def get_strong_connection_focal_info(df, rank=1, measure="containers"):
    """Extract the focal edge information from the strong connections data"""
    # Check if df is None
    if df is None:
        return None
        
    # Prepare the weighted edges dataframe - must exactly match the logic in visualise_strong_connections
    if measure == "containers":
        weighted_edges_df = (
            df.dropna(subset=["buyer_id", "seller_id", "container_number"])
            .groupby(["buyer_id", "seller_id"])["container_number"]
            .nunique()  # Count distinct containers
            .reset_index()
            .rename(columns={"container_number": "weight"})
            .sort_values("weight", ascending=False)
        )
    elif measure == "std_cartons":
        weighted_edges_df = (
            df.dropna(subset=["buyer_id", "seller_id", "std_cartons"])
            .groupby(["buyer_id", "seller_id"])["std_cartons"]
            .sum()
            .reset_index()
            .rename(columns={"std_cartons": "weight"})
            .sort_values("weight", ascending=False)
        )
    else:  # measure == 'revenue'
        weighted_edges_df = (
            df.dropna(subset=["buyer_id", "seller_id", "income"])
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


def create_weighted_graph(df, measure="containers"):
    """Create a weighted graph from network data based on the specified measure."""
    # Create an undirected graph for ALL measures
    G = nx.Graph()

    # Prepare edges based on measure
    if measure == "containers":
        edges_df = (
            df.dropna(subset=["buyer_id", "seller_id", "container_number"])
            .groupby(["buyer_id", "seller_id"])["container_number"]
            .nunique()
            .reset_index()
            .rename(columns={"container_number": "weight"})
        )
    elif measure == "std_cartons":
        edges_df = (
            df.dropna(subset=["buyer_id", "seller_id", "std_cartons"])
            .groupby(["buyer_id", "seller_id"])["std_cartons"]
            .sum()
            .reset_index()
            .rename(columns={"std_cartons": "weight"})
        )
    else:  # revenue
        edges_df = (
            df.dropna(subset=["buyer_id", "seller_id", "income"])
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

    # Map new visualisation types to old ones for backward compatibility
    viz_type_map = {
        "relationship": "Strong Connections",
        "company_network": "Company Network",
        "network_timelapse": "Network Timelapse"
    }
    
    # Convert new viz_type to old format if needed
    display_viz_type = viz_type_map.get(viz_type, viz_type)

    # Additional stats depending on visualisation type
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

        # Use focal info that was stored when creating the visualisation
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

                # USE THE FULL GRAPH for getting all connections, not just the visualisation subgraph
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

    return stats


def display_network_stats(stats):
    """Display the formatted network statistics in Streamlit.
    This function works the same as display_visualisation_stats for backward compatibility.
    """
    # Simply call the generalized function
    display_visualisation_stats(stats)


def format_heatmap_stats(data, params=None, metadata=None):
    """Format heatmap statistics into a structured display for Streamlit"""
    if data is None or not params:
        return None
    
    try:
        stats = {}
        
        # Get parameters safely, ensuring defaults if values are missing
        row_col = params.get("row_col", "")
        col_col = params.get("col_col", "") 
        value_col = params.get("value_col", "")
        
        # Include parameters used as a dictionary
        param_dict = {}
        if row_col:
            param_dict["Row Variable"] = str(row_col)
        if col_col:
            param_dict["Column Variable"] = str(col_col)
        if value_col:
            param_dict["Measure"] = str(value_col)
            
        param_dict["Significance Level"] = str(params.get("significance_level", 0.05))
        param_dict["Correct Multiple Tests"] = str(params.get("correct_multiple_tests", True))
        param_dict["Minimum Effect Size"] = str(params.get("min_effect_size", 5.0))
            
        stats["Parameters"] = param_dict
        
        # Add p-value table if available in metadata
        if metadata and "pvalue_table" in metadata and "pivot_data" in metadata:
            pvalue_df = metadata["pvalue_table"]
            pivot_df = metadata["pivot_data"]
            
            try:
                # Format the p-value table
                # The pvalue_df contains strings in format "0.xxxx (n)" 
                # We need to extract the p-value and count, reformat them, and recombine
                formatted_df = pvalue_df.copy()
                
                # Process each cell to reformat
                for i in range(len(formatted_df.index)):
                    for j in range(len(formatted_df.columns)):
                        cell_value = formatted_df.iloc[i, j]
                        
                        if cell_value and isinstance(cell_value, str):
                            # Extract p-value and count using regex
                            import re
                            match = re.match(r"([\d\.]+) \((\d+)\)", cell_value)
                            if match:
                                p_value = float(match.group(1))
                                count = int(match.group(2))
                                
                                # Format p-value as percentage with 1 decimal place
                                p_value_fmt = f"{p_value * 100:.1f}%"
                                
                                # Format count with thousand separators (space)
                                count_fmt = f"{count:,}".replace(",", " ")
                                
                                # Combine into new format
                                formatted_df.iloc[i, j] = f"{p_value_fmt} ({count_fmt})"
                
                # Sort the table by totals (similar to how the heatmap is sorted)
                # Calculate totals for rows and columns
                try:
                    # Calculate row and column totals from the pivot data
                    row_totals = pivot_df.sum(axis=1).sort_values(ascending=False)
                    col_totals = pivot_df.sum(axis=0).sort_values(ascending=False)
                    
                    # Filter to only include rows and columns in the formatted_df
                    common_rows = [idx for idx in row_totals.index if idx in formatted_df.index]
                    common_cols = [col for col in col_totals.index if col in formatted_df.columns]
                    
                    # Reorder the formatted DataFrame according to the totals, descending
                    formatted_df = formatted_df.loc[common_rows, common_cols]
                    
                    # Add the sorted formatted table to stats
                    stats["P-Value Table"] = formatted_df
                except Exception as e:
                    # If sorting fails, still show the table without sorting
                    stats["P-Value Table"] = formatted_df
                    stats["P-Value Table Sorting Error"] = str(e)
            except Exception as e:
                # Still include the original p-value table if formatting fails
                stats["P-Value Table"] = pvalue_df
                stats["P-Value Formatting Error"] = str(e)
        elif metadata and "pvalue_table" in metadata:
            # If we have p-values but no pivot data, still show the table
            stats["P-Value Table"] = metadata["pvalue_table"]
            
        # Add data information analysis - ensure all fields exist
        if not isinstance(data, pd.DataFrame):
            stats["Data Error"] = "Invalid data format"
            return stats
            
        df = data
        df_copy = df.copy()
        
        # Data overview (dictionary format)
        stats["Data Overview"] = {
            "Total Rows": str(len(df))
        }
        
        # Check each column for existence before using it
        columns_exist = True
        for col in [col for col in [row_col, col_col, value_col] if col]:
            if col not in df_copy.columns:
                columns_exist = False
                stats["Data Error"] = f"Column '{col}' not found in data"
        
        if not columns_exist:
            return stats
        
        # Replace 'None' string values with NaN if columns exist
        columns_to_check = [col for col in [row_col, col_col, value_col] if col and col in df_copy.columns]
        for col in columns_to_check:
            df_copy.loc[df_copy[col] == 'None', col] = np.nan
        
        # Get columns that actually exist in the dataframe for filtering
        filter_cols = [col for col in [row_col, col_col] if col and col in df_copy.columns]
        if len(filter_cols) >= 1:
            filtered_df = df_copy.dropna(subset=filter_cols)
            stats["Data Overview"]["Rows Used in Heatmap"] = str(len(filtered_df))
        else:
            filtered_df = df_copy
            stats["Data Overview"]["Rows Used in Heatmap"] = "N/A (no filter columns)"
        
        # Critical fields information (as a dataframe)
        columns_to_analyze = [col for col in [row_col, col_col] if col and col in filtered_df.columns]
        
        if columns_to_analyze:
            try:
                # Create a proper dataframe for Critical Fields
                col_data = {
                    'Column': columns_to_analyze,
                    'Non-null Values': [filtered_df[col].count() for col in columns_to_analyze],
                    'Non-Null Unique Values': [filtered_df[col].nunique() for col in columns_to_analyze],
                    'Dtype': [filtered_df[col].dtype for col in columns_to_analyze],
                }
                
                # Only add sample values if there are values to sample
                sample_values = []
                for col in columns_to_analyze:
                    values = filtered_df[col].head(3).tolist() if not filtered_df[col].empty else []
                    sample_values.append(str(values))
                
                col_data['Sample Values'] = sample_values
                
                col_stats = pd.DataFrame(col_data)
                col_stats = col_stats.astype(str)
                stats["Critical Fields"] = col_stats
                
                # Create a proper dataframe for Missing Values Analysis
                missing_columns = [col for col in [row_col, col_col, value_col] if col and col in df_copy.columns]
                if missing_columns:
                    na_data = {
                        'Column': missing_columns,
                        'Missing Values': [df_copy[col].isna().sum() for col in missing_columns],
                        'Missing Percent': [df_copy[col].isna().mean() * 100 for col in missing_columns]
                    }
                    
                    na_counts = pd.DataFrame(na_data)
                    na_counts['Missing Percent'] = na_counts['Missing Percent'].round(2)
                    na_counts = na_counts.astype(str)
                    stats["Missing Values Analysis"] = na_counts
            except Exception as e:
                stats["Data Analysis Error"] = str(e)
        
        return stats
    except Exception as e:
        import traceback
        return {
            "Error": str(e),
            "Traceback": traceback.format_exc()
        }


def display_visualisation_stats(stats):
    """Display the formatted visualisation statistics in Streamlit.
    This is a generic version of display_network_stats that can handle any visualisation type.
    """
    if not stats:
        st.info("No statistics available for this visualisation.")
        return

    # Display top-level items
    for section, content in stats.items():
        st.subheader(section)
        
        if isinstance(content, pd.DataFrame):
            # For P-Value Table, show the index (row names)
            if section == "P-Value Table":
                st.text("The P-Value Table shows two-sided statistical testing based on a chi-squared independence model.")
                st.markdown("""
- The P-Value Table shows the statistical likelihood (as percentages) that the observed proportion difference could happen by chance.
- Lower p-values suggest stronger evidence of a real association.
- The number in parentheses shows the actual count of items in that cell.
- Ordered by row and column totals to match the heatmap, making it easier to cross-reference.
                            """)
                st.dataframe(content)
            else:
                # For other DataFrames, hide the index
                st.dataframe(content, hide_index=True)
        elif isinstance(content, dict):
            # If the content is a dictionary, convert to DataFrame and display
            df = pd.DataFrame(list(content.items()), columns=["Metric", "Value"])
            st.dataframe(df, hide_index=True)
        elif isinstance(content, list):
            # If the content is a list, display as bullet points
            for item in content:
                st.markdown(f"- {item}")
        else:
            # If the content is a string, display as markdown
            st.markdown(content)


def format_stats(visualisation):
    """
    Format statistics for any visualisation type using the unified visualisation structure.
    
    Parameters:
    -----------
    visualisation : dict
        The unified visualisation structure from st.session_state.visualisation
        
    Returns:
    --------
    dict or None
        Formatted statistics or None if not available
    """
    if not visualisation or not visualisation["type"]:
        return None
        
    viz_type = visualisation["type"]
    params = visualisation["params"]
    # result_df = visualisation["result_df"]
    graph = visualisation["graph"]
    metadata = visualisation["metadata"]
    
    # Route to the appropriate formatter based on visualisation type
    if viz_type in ["relationship", "strong_connections", "company_network", "network_timelapse"]:
        focal_info = metadata.get("focal_info")
        full_graph = metadata.get("full_graph")
        return format_network_stats(graph, viz_type, params, focal_info, full_graph)
    
    elif viz_type in ["heatmap", "packing_week_heatmap", "standard_heatmap"]:
        # Ensure we have value_col in params for heatmap data analysis
        value_col = params.get("value_col", params.get("measure", "containers"))
        if "value_col" not in params and value_col:
            params["value_col"] = value_col
        return format_heatmap_stats(visualisation["data"], params, metadata)
    
    return None


def export_plot_as_png(fig, filename="plot.png", dpi=300):
    """
    Converts a matplotlib figure to a PNG image and creates a download button.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to be exported
    filename : str, optional
        The name of the file to be downloaded (default: "plot.png")
    dpi : int, optional
        The resolution of the exported image (default: 300)
        
    Returns:
    --------
    None
    """
    if fig is None:
        st.warning("No plot available to export.")
        return
    
    import io
    
    # Create a BytesIO buffer to save the figure
    buf = io.BytesIO()
    
    # Save the figure to the buffer with specified DPI
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    
    # Set the buffer position to the beginning
    buf.seek(0)
    
    # Create a download button
    st.download_button(
        label="📥 Export as PNG",
        data=buf,
        file_name=filename,
        mime="image/png",
        key="download_plot"
    )


def get_heatmap_interpretation():
    """
    Returns a detailed explanation of how to interpret heatmap visualisations.
    
    Returns:
    --------
    str
        Markdown formatted text explaining heatmap interpretation
    """
    interpretation = """
### Basic Structure
- Each cell in the heatmap represents the proportion of the selected measure (e.g., standard cartons) for a specific row-column combination.
- Colors intensity indicates the proportion value - darker cells have higher proportions.
- Each row sums to 1 (or 100%), meaning the values show the distribution across columns for each row.
- For example, if a cell shows 0.25 (or 25%), it means that 25% of the total value for that row went to that column.
- This allows you to compare distribution patterns across different rows even when their total volumes differ significantly.
- Rows with larger total values (shown in the rightmost column) contribute more to the overall weighted average.

### Statistical Significance Testing
The statistical test is based on a chi-squared independence model, which determines if the observed proportion differs significantly from what would be expected if the row and column variables were independent:

- **Expected values** are calculated using both row and column marginals:
- **Two-sided testing** is used to identify both higher and lower than expected proportions:
  - Cells marked with * have significantly **higher** proportions than expected
  - Cells marked with † have significantly **lower** proportions than expected
- A cell is marked as statistically significant when:
  1. The absolute difference between observed and expected proportions exceeds the minimum effect size
  2. There's sufficient sample size in that row to make a reliable determination
  3. The p-value falls below the significance threshold after any multiple testing correction

### Interpreting Significant Cells
- A significant cell means there's a meaningful association between the row and column variables.
- For example, if a cell for variety "Red Globe" and country "China" is marked with *:
  - A higher percentage of Red Globe grapes go to China than would be expected based on the overall distribution.
  - This suggests a special relationship or preference - China may particularly favor Red Globe compared to other varieties.
- Conversely, a cell marked with † indicates a significantly lower proportion than expected, suggesting potential avoidance or negative association.

### Understanding the 'Other' Category
- When there are many categories (more than 19), less frequent items are grouped into an 'Other' category.
- An 'Other' category is marked as statistically significant if **any** of its component categories show significance.
- The direction (* or †) is based on the first significant component found in the group.
- Significance in 'Other' does not mean the entire group is significant, only that at least one component shows a significant pattern.
- When 'Other' shows significance, it suggests further investigation into the specific components may reveal interesting patterns.
"""
    return interpretation 