import streamlit as st
import matplotlib.pyplot as plt
from viz_utils import get_strong_connection_focal_info, create_weighted_graph, anim_to_html
from network_analysis import (
    visualize_strong_connections,
    visualize_company_network,
    create_network_timelapse,
)
from cg_analysis import (
    create_heatmap,
    create_heatmap_packing_week,
    calculate_concentration_metrics,
    plot_concentration_bubble,
)
from matplotlib.colors import LinearSegmentedColormap


def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    # Data state
    if "cg_df" not in st.session_state:
        st.session_state.cg_df = None
    if "agg_df" not in st.session_state:
        st.session_state.agg_df = None
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "company_ids" not in st.session_state:
        st.session_state.company_ids = []
    
    # Unified visualization state
    if "visualization" not in st.session_state:
        st.session_state.visualization = {
            "type": None,           # Type of visualization (e.g., "relationship", "heatmap")
            "params": {},           # Parameters used to generate the visualization 
            "result_df": None,      # Result dataframe (e.g., pivot_df, metrics_df)
            "graph": None,          # Network graph for network visualizations
            "plot": None,           # Matplotlib figure for static visualizations
            "html": None,           # HTML content for interactive visualizations
            "metadata": {}          # Additional metadata specific to visualization type
        }
    
    # For backward compatibility during transition
    if "network_data" not in st.session_state:
        st.session_state.network_data = None
    if "viz_data" not in st.session_state:
        st.session_state.viz_data = None
    if "current_plot" not in st.session_state:
        st.session_state.current_plot = None
    if "current_html" not in st.session_state:
        st.session_state.current_html = None

    # Initialize the visualisation state flags if they don't exist
    if "generating_strong_connections" not in st.session_state:
        st.session_state.generating_strong_connections = False
    if "generating_company_network" not in st.session_state:
        st.session_state.generating_company_network = False
    if "generating_network_timelapse" not in st.session_state:
        st.session_state.generating_network_timelapse = False
    if "generating_heatmap" not in st.session_state:
        st.session_state.generating_heatmap = False
    if "generating_packing_week_heatmap" not in st.session_state:
        st.session_state.generating_packing_week_heatmap = False
    if "generating_concentration_bubble" not in st.session_state:
        st.session_state.generating_concentration_bubble = False


def clear_visualisation_state():
    """Clear all visualisation-related state variables when loading new data."""
    # Clear the unified visualization state
    st.session_state.visualization = {
        "type": None,
        "params": {},
        "result_df": None,
        "graph": None,
        "plot": None,
        "html": None,
        "metadata": {}
    }
    
    # For backward compatibility during transition
    st.session_state.network_data = None
    st.session_state.viz_data = None
    st.session_state.current_plot = None
    st.session_state.current_html = None


def update_visualization(viz_type, params, result_df=None, graph=None, plot=None, html=None, metadata=None):
    """
    Update the unified visualization state with new data.
    
    Parameters:
    -----------
    viz_type : str
        Type of visualization (e.g., "relationship", "heatmap")
    params : dict
        Parameters used to generate the visualization
    result_df : pandas.DataFrame, optional
        Result dataframe (e.g., pivot_df, metrics_df)
    graph : networkx.Graph, optional
        Network graph for network visualizations
    plot : matplotlib.figure.Figure, optional
        Matplotlib figure for static visualizations
    html : str, optional
        HTML content for interactive visualizations
    metadata : dict, optional
        Additional metadata specific to visualization type
    """
    # Update the unified visualization state
    st.session_state.visualization = {
        "type": viz_type,
        "params": params,
        "result_df": result_df,
        "graph": graph,
        "plot": plot,
        "html": html,
        "metadata": metadata or {}
    }
    
    # For backward compatibility during transition
    st.session_state.current_plot = plot
    st.session_state.current_html = html

    # Increment the viz ID to update statistics display
    increment_viz_id()


def increment_viz_id():
    """Increment the visualisation ID to ensure statistics update properly."""
    if "last_viz_id" in st.session_state:
        st.session_state.last_viz_id += 1
    else:
        st.session_state.last_viz_id = 1


def update_data_state(cg_df):
    """
    Update the data-related state variables.
    
    Parameters:
    -----------
    cg_df : pandas.DataFrame
        The carton groupings dataframe with essential data
    """
    if cg_df is not None:
        st.session_state.cg_df = cg_df
        st.session_state.data_loaded = True
        
        # Initialize company_ids as empty list - will be populated when cg_df is created
        st.session_state.company_ids = []
        
        # Calculate company IDs if cg_df is provided
        if cg_df is not None:
            buyer_ids = cg_df["buyer_id"].dropna().unique().tolist()
            seller_ids = cg_df["seller_id"].dropna().unique().tolist()
            st.session_state.company_ids = sorted(list(set(buyer_ids + seller_ids)))
    else:
        st.session_state.cg_df = None
        st.session_state.data_loaded = False
        st.session_state.company_ids = []


def generate_strong_connections_viz(params):
    """Generate the Strong Connections visualisation."""
    try:
        # Check if agg_df exists in session state and is not None
        if st.session_state.agg_df is None:
            st.error("No aggregated data available. Please reload the data.")
            return False
            
        # Use wrapper function to get the focal edge info
        focal_info = get_strong_connection_focal_info(
            agg_df=st.session_state.agg_df, 
            rank=params["rank"], 
            measure=params["measure"]
        )

        if not focal_info:
            st.error(f"No data available for rank {params['rank']}. Try a lower rank.")
            return False

        # Create a figure for capturing the visualisation
        fig = plt.figure(figsize=(14, 10))

        # Call the visualisation function
        visualize_strong_connections(
            agg_df=st.session_state.agg_df,
            rank=params["rank"],
            measure=params["measure"],
            degree=params["degree"],
            layout=params["layout"],
            spacing=1.0,  # Use default spacing
            iterations=params["iterations"],
            figsize=(14, 10),
        )

        # Create weighted graph for statistics
        G = create_weighted_graph(st.session_state.agg_df, params["measure"])

        # Get the current figure
        fig = plt.gcf()
        plt.close(fig)  # Close the figure to free memory
        
        # Update the visualization state with the new data
        update_visualization(
            viz_type="relationship",
            params=params,
            graph=G,
            plot=fig,
            metadata={"focal_info": focal_info}
        )
        
        # For backward compatibility during transition
        st.session_state.network_data = {
            "graph": G,
            "type": "Strong Connections",
            "params": params,
            "focal_info": focal_info,
        }
        
        st.session_state.viz_data = {
            "type": "relationship",
            "graph": G,
            "params": params,
            "focal_info": focal_info,
        }
        
        return True
    except Exception as e:
        st.error(f"Error generating Strong Connections visualisation: {str(e)}")
        return False


def generate_company_network_viz(params):
    """Generate the Company Network visualisation."""
    try:
        # Create a figure for capturing the visualisation
        fig = plt.figure(figsize=(14, 10))

        # Call the visualisation function
        visualize_company_network(
            agg_df=st.session_state.agg_df,
            company_id=params["company_id"],
            measure=params["measure"],
            degree=params["degree"],
            layout=params["layout"],
            spacing=1.5,  # Use default spacing
            iterations=params["iterations"],
            figsize=(14, 10),
        )

        # Create full graph for statistics
        full_G = create_weighted_graph(st.session_state.agg_df, params["measure"])

        # Create subgraph for visualisation
        subgraph_nodes = set()
        company_id = params["company_id"]

        # Only continue if company_id is in the graph
        if company_id in full_G:
            # Get neighbors up to degree
            subgraph_nodes = {company_id}
            current_frontier = {company_id}

            for d in range(params["degree"]):
                next_frontier = set()
                for node in current_frontier:
                    next_frontier.update(full_G.neighbors(node))
                subgraph_nodes.update(next_frontier)
                current_frontier = next_frontier

            # Create subgraph with only these nodes
            G = full_G.subgraph(subgraph_nodes).copy()
        else:
            G = full_G  # Fallback if company not found

        # Get the current figure
        fig = plt.gcf()
        plt.close(fig)  # Close the figure to free memory
        
        # Update the visualization state with the new data
        update_visualization(
            viz_type="company_network",
            params=params,
            graph=G,
            plot=fig,
            metadata={
                "full_graph": full_G,
                "focal_company": company_id
            }
        )
        
        # For backward compatibility during transition  
        st.session_state.network_data = {
            "graph": G,  # The visualisation subgraph
            "full_graph": full_G,  # The complete graph for statistics
            "type": "Company Network",
            "params": params,
        }
        
        st.session_state.viz_data = {
            "type": "company_network",
            "graph": G,
            "full_graph": full_G,
            "params": params,
            "focal_company": company_id
        }
        
        return True
    except Exception as e:
        st.error(f"Error generating Company Network visualisation: {str(e)}")
        return False


def generate_network_timelapse_viz(params):
    """Generate the Network Timelapse visualisation."""
    try:
        # Get the timelapse visualisation as HTML and all data
        html, animation, G, time_slices = create_network_timelapse(
            df=st.session_state.agg_df,
            measure=params["measure"],
            time_window=params["time_window"],
            min_weight=params["min_weight"],
            layout=params["layout"],
            figsize=(14, 10),
        )

        if html is not None:
            # Update the visualization state with the new data
            update_visualization(
                viz_type="network_timelapse",
                params=params,
                graph=G,
                html=html,
                metadata={"time_slices": time_slices}
            )
            
            # For backward compatibility during transition
            st.session_state.network_data = {
                "graph": G,  # The aggregated graph
                "type": "Network Timelapse",
                "params": params,
                "time_slices": time_slices,
            }
            
            st.session_state.viz_data = {
                "type": "network_timelapse",
                "graph": G,
                "params": params,
                "time_slices": time_slices,
            }
            
            return True
        else:
            st.error("Failed to generate animation.")
            return False
    except Exception as e:
        st.error(f"Error generating Network Timelapse: {str(e)}")
        return False


def generate_heatmap_viz(params):
    """Generate the Heatmap visualisation."""
    try:
        # Check if cg_df exists in session state and is not None
        if st.session_state.cg_df is None:
            st.error("No data available. Please load the data.")
            return False
            
        # Create a figure for capturing the visualisation
        fig, pivot_df = create_heatmap(
            df=st.session_state.cg_df,
            row_col=params["row_col"],
            col_col=params["col_col"],
            value_col=params["value_col"],
            significance_level=params["significance_level"],
            correct_multiple_tests=params["correct_multiple_tests"],
            min_effect_size=params["min_effect_size"],
        )

        # Close the figure to free memory after storing it
        plt.close(fig)
        
        # Update the visualization state with the new data
        update_visualization(
            viz_type="heatmap",
            params={
                "x_axis": params["col_col"],
                "y_axis": params["row_col"],
                "color_by": params["value_col"],
                "significance_level": params["significance_level"],
                "correct_multiple_tests": params["correct_multiple_tests"],
                "min_effect_size": params["min_effect_size"],
            },
            result_df=pivot_df,
            plot=fig
        )
        
        # For backward compatibility during transition
        st.session_state.network_data = {
            "graph": None,  # No graph for heatmap visualizations
            "type": "Heatmap",
            "params": params,
            "pivot_df": pivot_df,
        }
        
        st.session_state.viz_data = {
            "type": "heatmap",
            "data": pivot_df,
            "params": {
                "x_axis": params["col_col"],
                "y_axis": params["row_col"],
                "color_by": params["value_col"],
                "significance_level": params["significance_level"],
                "correct_multiple_tests": params["correct_multiple_tests"],
                "min_effect_size": params["min_effect_size"],
            }
        }
        
        return True
    except Exception as e:
        st.error(f"Error generating Heatmap: {str(e)}")
        return False


def generate_packing_week_heatmap_viz(params):
    """Generate the Packing Week Heatmap visualisation."""
    try:
        # Check if cg_df exists in session state and is not None
        if st.session_state.cg_df is None:
            st.error("No data available. Please load the data.")
            return False
            
        # Create a figure for capturing the visualisation
        fig, pivot_df = create_heatmap_packing_week(
            df=st.session_state.cg_df,
            col_col=params["col_col"],
            value_col=params["value_col"],
            significance_level=params["significance_level"],
            correct_multiple_tests=params["correct_multiple_tests"],
            min_effect_size=params["min_effect_size"],
        )

        # Close the figure to free memory after storing it
        plt.close(fig)
        
        # Update the visualization state with the new data
        update_visualization(
            viz_type="packing_week_heatmap",
            params={
                "x_axis": "packing_week",
                "y_axis": params["col_col"],
                "color_by": params["value_col"],
                "significance_level": params["significance_level"],
                "correct_multiple_tests": params["correct_multiple_tests"],
                "min_effect_size": params["min_effect_size"],
            },
            result_df=pivot_df,
            plot=fig
        )
        
        # For backward compatibility during transition
        st.session_state.network_data = {
            "graph": None,  # No graph for heatmap visualizations
            "type": "Packing Week Heatmap",
            "params": params,
            "pivot_df": pivot_df,
        }
        
        st.session_state.viz_data = {
            "type": "packing_week_heatmap",
            "data": pivot_df,
            "params": {
                "x_axis": "packing_week",
                "y_axis": params["col_col"],
                "color_by": params["value_col"],
                "significance_level": params["significance_level"],
                "correct_multiple_tests": params["correct_multiple_tests"],
                "min_effect_size": params["min_effect_size"],
            }
        }
        
        return True
    except Exception as e:
        st.error(f"Error generating Packing Week Heatmap: {str(e)}")
        return False


def generate_concentration_bubble_viz(params):
    """Generate the Concentration Bubble Plot visualisation."""
    try:
        # Check if cg_df exists in session state and is not None
        if st.session_state.cg_df is None:
            st.error("No data available. Please load the data.")
            return False
            
        # Calculate concentration metrics
        metrics_df = calculate_concentration_metrics(
            df=st.session_state.cg_df,
            entity1_col=params["entity1_col"],
            entity2_col=params["entity2_col"],
            value_col=params["value_col"],
        )
        
        # Create bubble plot
        fig = plot_concentration_bubble(
            metrics_df=metrics_df,
            title=params["title"],
            min_pallets=params["min_pallets"],
            label_threshold=params["label_threshold"],
        )

        # Close the figure to free memory after storing it
        plt.close(fig)
        
        # Update the visualization state with the new data
        update_visualization(
            viz_type="concentration_bubble",
            params={
                "x_axis": "herfindahl",
                "y_axis": "cv",
                "size_by": "volume",
                "color_by": "entity1_col",
                "entity1_col": params["entity1_col"],
                "entity2_col": params["entity2_col"],
                "value_col": params["value_col"],
                "min_pallets": params["min_pallets"],
                "label_threshold": params["label_threshold"],
            },
            result_df=metrics_df,
            plot=fig
        )
        
        # For backward compatibility during transition
        st.session_state.network_data = {
            "graph": None,  # No graph for bubble plot visualizations
            "type": "Concentration Bubble Plot",
            "params": params,
            "metrics_df": metrics_df,
        }
        
        st.session_state.viz_data = {
            "type": "concentration_bubble",
            "data": metrics_df,
            "params": {
                "x_axis": "herfindahl",
                "y_axis": "cv",
                "size_by": "volume",
                "color_by": "entity1_col",
                "entity1_col": params["entity1_col"],
                "entity2_col": params["entity2_col"],
                "value_col": params["value_col"],
                "min_pallets": params["min_pallets"],
                "label_threshold": params["label_threshold"],
            }
        }
        
        return True
    except Exception as e:
        st.error(f"Error generating Concentration Bubble Plot: {str(e)}")
        return False 