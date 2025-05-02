import streamlit as st
import matplotlib.pyplot as plt
from viz_utils import get_strong_connection_focal_info, create_weighted_graph
from modules.network import (
    visualize_strong_connections,
    visualize_company_network,
    create_network_timelapse,
)
from modules.heatmap import (
    create_heatmap,
    create_heatmap_packing_week,
)


# ================ STATE INITIALIZATION ================

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    # Data state
    if "df" not in st.session_state:
        st.session_state.df = None
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "company_ids" not in st.session_state:
        st.session_state.company_ids = []

    # Unified visualisation state
    if "visualisation" not in st.session_state:
        st.session_state.visualisation = {
            "type": None,  # Type of visualisation (e.g., "network", "heatmap")
            "params": {},  # Parameters used to generate the visualisation
            "graph": None,  # Network graph for network visualisations
            "plot": None,  # Matplotlib figure for static visualisations
            "html": None,  # HTML content for interactive visualisations
            "metadata": {},  # Additional information about the visualisation
        }

    # For backward compatibility
    if "current_plot" not in st.session_state:
        st.session_state.current_plot = None
    if "current_html" not in st.session_state:
        st.session_state.current_html = None
    if "last_viz_id" not in st.session_state:
        st.session_state.last_viz_id = 0


def clear_visualisation_state():
    """Clear all visualisation-related state variables."""
    st.session_state.visualisation = {
        "type": None,
        "params": {},
        "graph": None,
        "plot": None,
        "html": None,
        "metadata": {},
    }

    # For backward compatibility
    st.session_state.current_plot = None
    st.session_state.current_html = None


def update_visualisation_state(
    viz_type, params, graph=None, plot=None, html=None, metadata=None
):
    """
    Update the unified visualisation state with new data.

    Parameters:
    -----------
    viz_type : str
        Type of visualisation (e.g., "network", "heatmap")
    params : dict
        Parameters used to generate the visualisation
    graph : networkx.Graph, optional
        Network graph for network visualisations
    plot : matplotlib.figure.Figure, optional
        Matplotlib figure for static visualisations
    html : str, optional
        HTML content for interactive visualisations
    metadata : dict, optional
        Additional metadata for the visualisation
    """
    # Update the visualisation state
    st.session_state.visualisation = {
        "type": viz_type,
        "params": params,
        "graph": graph,
        "plot": plot,
        "html": html,
        "metadata": metadata or {},
    }

    # For backward compatibility
    st.session_state.current_plot = plot
    st.session_state.current_html = html

    # Increment visualisation ID for statistics updates
    if "last_viz_id" in st.session_state:
        st.session_state.last_viz_id += 1
    else:
        st.session_state.last_viz_id = 1


def update_data_state(df):
    """
    Update the data-related state variables.

    Parameters:
    -----------
    df : pandas.DataFrame
        The main dataframe with essential data
    """
    if df is not None:
        st.session_state.df = df
        st.session_state.data_loaded = True

        # Extract and store unique company IDs from the dataframe
        buyer_ids = df["buyer_id"].dropna().unique().tolist()
        seller_ids = df["seller_id"].dropna().unique().tolist()
        st.session_state.company_ids = sorted(list(set(buyer_ids + seller_ids)))
    else:
        st.session_state.df = None
        st.session_state.data_loaded = False
        st.session_state.company_ids = []
        clear_visualisation_state()


# ================ VISUALIZATION GENERATORS ================

# --- Network Visualizations ---

def create_network_visualisation(viz_type, df, params, visualisation_func, graph_processor=None):
    """
    Generic function to create network visualisations.
    
    Parameters:
    -----------
    viz_type : str
        Type of network visualisation
    df : pandas.DataFrame
        Data for visualisation
    params : dict
        Parameters for the visualisation
    visualisation_func : function
        The specific visualisation function to call
    graph_processor : function, optional
        Function to process the graph after creation
    
    Returns:
    --------
    bool
        Success or failure of visualisation creation
    """
    try:
        if df is None:
            st.error("No data available for visualisation.")
            return False
            
        # Standard figure size for network visualisations
        fig_size = (14, 10)
        fig = plt.figure(figsize=fig_size)
        
        # Call the specific visualisation function
        result = visualisation_func(df=df, **params, figsize=fig_size)
        
        # Create weighted graph for statistics
        measure = params.get("measure", "carton_count")
        G = create_weighted_graph(df, measure)
        
        # Process graph if needed
        if graph_processor and callable(graph_processor):
            G = graph_processor(G, params)
            
        # Get the current figure and close it to free memory
        fig = plt.gcf()
        plt.close(fig)
        
        # Get metadata if result is a tuple (some functions return additional data)
        metadata = {}
        if isinstance(result, tuple):
            metadata = {"additional_data": result[1:]}
        
        # Update visualisation state
        update_visualisation_state(
            viz_type=viz_type,
            params=params,
            graph=G,
            plot=fig,
            metadata=metadata,
        )
        
        return True
    except Exception as e:
        st.error(f"Error generating {viz_type} visualisation: {str(e)}")
        return False


def create_strong_connections_network(params):
    """Create the Strong Connections network visualisation."""
    df = st.session_state.df
    
    # Add focal info to metadata
    focal_info = get_strong_connection_focal_info(
        df=df, rank=params["rank"], measure=params["measure"]
    )
    
    if not focal_info:
        st.error(f"No data available for rank {params['rank']}. Try a lower rank.")
        return False
    
    # Create visualisation function with specific parameters
    def viz_func(df, **viz_params):
        return visualize_strong_connections(
            df=df,
            rank=viz_params["rank"],
            measure=viz_params["measure"],
            degree=viz_params["degree"],
            layout=viz_params["layout"],
            spacing=1.0,
            iterations=viz_params["iterations"],
            figsize=viz_params["figsize"],
        )
    
    success = create_network_visualisation(
        viz_type="strong_connections",
        df=df,
        params=params,
        visualisation_func=viz_func
    )
    
    if success:
        # Update metadata with focal info
        st.session_state.visualisation["metadata"]["focal_info"] = focal_info
    
    return success


def create_company_network(params):
    """Create the Company Network visualisation."""
    df = st.session_state.df
    
    # Create visualisation function with specific parameters
    def viz_func(df, **viz_params):
        return visualize_company_network(
            df=df,
            company_id=viz_params["company_id"],
            measure=viz_params["measure"],
            degree=viz_params["degree"],
            layout=viz_params["layout"],
            spacing=1.5,
            figsize=viz_params["figsize"],
        )
    
    # Create subgraph processor function
    def process_company_subgraph(full_G, params):
        company_id = params["company_id"]
        degree = params["degree"]
        
        # Only continue if company_id is in the graph
        if company_id not in full_G:
            return full_G
            
        # Get neighbors up to specified degree
        subgraph_nodes = {company_id}
        current_frontier = {company_id}

        for d in range(degree):
            next_frontier = set()
            for node in current_frontier:
                next_frontier.update(full_G.neighbors(node))
            subgraph_nodes.update(next_frontier)
            current_frontier = next_frontier

        # Create subgraph with only these nodes
        return full_G.subgraph(subgraph_nodes).copy()
    
    success = create_network_visualisation(
        viz_type="company_network",
        df=df,
        params=params,
        visualisation_func=viz_func,
        graph_processor=process_company_subgraph
    )
    
    if success:
        # Add company ID to metadata
        st.session_state.visualisation["metadata"]["focal_company"] = params["company_id"]
    
    return success


def create_network_timelapse(params):
    """Create the Network Timelapse visualisation."""
    try:
        df = st.session_state.df
        
        if df is None:
            st.error("No data available for visualisation.")
            return False
            
        # Get the timelapse visualisation as HTML and all data
        html, G, time_slices = create_network_timelapse(
            df=df,
            measure=params["measure"],
            time_window=params["time_window"],
            min_weight=params["min_weight"],
            layout=params["layout"],
            figsize=(14, 10),
        )

        if html is None:
            st.error("Failed to generate network timelapse.")
            return False
            
        # Update the visualisation state
        update_visualisation_state(
            viz_type="network_timelapse",
            params=params,
            graph=G,
            html=html,
            metadata={"time_slices": time_slices},
        )

        return True
    except Exception as e:
        st.error(f"Error generating Network Timelapse: {str(e)}")
        return False


# --- Heatmap Visualizations ---

def create_heatmap_visualisation(viz_type, df, params, heatmap_func):
    """
    Generic function to create heatmap visualisations.
    
    Parameters:
    -----------
    viz_type : str
        Type of heatmap visualisation (e.g., "standard_heatmap", "packing_week_heatmap")
    df : pandas.DataFrame
        Data for visualisation
    params : dict
        Parameters for the visualisation
    heatmap_func : function
        Function to generate the heatmap
        
    Returns:
    --------
    bool
        Success or failure of visualisation creation
    """
    try:
        if df is None:
            st.error("No data available for visualisation.")
            return False
            
        # Call the heatmap function
        fig = heatmap_func(df, **params)
        
        if fig is None:
            st.error("Failed to generate heatmap.")
            return False
        
        # Extract pivot_df and pvalue_df from figure metadata
        metadata = {}
        if hasattr(fig, 'pivot_df'):
            metadata['pivot_data'] = fig.pivot_df
        if hasattr(fig, 'pvalue_df'):
            metadata['pvalue_table'] = fig.pvalue_df
            
        # Add the heatmap to the state
        update_visualisation_state(
            viz_type=viz_type,
            params=params,  # Use original params for stats
            plot=fig,
            metadata=metadata,
        )
        
        return True
        
    except Exception as e:
        st.error(f"Error generating {viz_type} visualisation: {str(e)}")
        return False


def create_standard_heatmap(params):
    """Create a standard heatmap visualisation."""
    df = st.session_state.df
    
    # Filter out parameters not accepted by create_heatmap
    heatmap_params = {
        'row_col': params.get('row_col'),
        'col_col': params.get('col_col'),
        'value_col': params.get('value_col'),
        'significance_level': params.get('significance_level', 0.05),
        'correct_multiple_tests': params.get('correct_multiple_tests', True),
        'min_effect_size': params.get('min_effect_size', 5.0)
    }
    
    # Remove None values
    heatmap_params = {k: v for k, v in heatmap_params.items() if v is not None}
        
    return create_heatmap_visualisation(
        viz_type="standard_heatmap",
        df=df,
        params=heatmap_params,
        heatmap_func=create_heatmap
    )


def create_packing_week_heatmap(params):
    """Create a packing week heatmap visualisation."""
    df = st.session_state.df
    
    # Filter out parameters not accepted by create_heatmap_packing_week
    heatmap_params = {
        'row_col': 'packing_week',  # Explicitly add row_col for consistency
        'col_col': params.get('col_col'),
        'value_col': params.get('value_col'),
        'significance_level': params.get('significance_level', 0.05),
        'correct_multiple_tests': params.get('correct_multiple_tests', True),
        'min_effect_size': params.get('min_effect_size', 5.0)
    }
    
    # Remove None values
    heatmap_params = {k: v for k, v in heatmap_params.items() if v is not None}
    
    success = create_heatmap_visualisation(
        viz_type="packing_week_heatmap",
        df=df,
        params=heatmap_params,
        heatmap_func=create_heatmap_packing_week
    )
        
    return success
