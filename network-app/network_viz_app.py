import streamlit as st

# Import modules
from data_loader import load_data_from_database, create_aggregated_dataframe
from state_manager import (
    initialize_session_state, 
    clear_visualisation_state, 
    update_data_state
)
from ui_components import (
    setup_page_config,
    apply_custom_css,
    render_header,
    render_data_filters,
    render_visualisation_selector,
    render_relationship_params,
    render_company_network_params,
    render_network_timelapse_params,
    render_heatmap_params,
    render_packing_week_heatmap_params,
    render_concentration_bubble_params,
    render_current_visualisation,
    render_data_preview
)
from viz_handlers import (
    handle_relationship_viz,
    handle_company_network_viz,
    handle_network_timelapse_viz,
    handle_heatmap_viz,
    handle_packing_week_heatmap_viz,
    handle_concentration_bubble_viz
)
from viz_utils import (
    format_network_stats,
    display_network_stats,
    format_heatmap_stats,
    format_bubble_stats,
    display_visualization_stats,
    format_stats
)


def main():
    # Setup page configuration
    setup_page_config()
    
    # Apply custom CSS
    apply_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Render application header
    render_header()
    
    # Render data filters in sidebar
    season_year, commodity_group, load_data = render_data_filters()
    
    # Handle data loading
    if load_data:
        # Clear previous visualisation state first
        clear_visualisation_state()
        with st.spinner("Loading and processing data..."):
            # Load new data - now only returns cg_df
            cg_df = load_data_from_database(season_year, commodity_group)
            update_data_state(cg_df)
    
    # Use the data from session state
    data_loaded = st.session_state.data_loaded
    
    # Only show visualisation options if data is loaded
    if data_loaded:
        # Render visualisation type selector in sidebar
        viz_type = render_visualisation_selector()
        
        # Create the appropriate aggregated dataframe based on the visualisation type
        with st.spinner("Preparing data for visualisation..."):
            if viz_type in ["Buyer-Seller Relationship", "Individual Company Network", "Network Timelapse"]:
                agg_df = create_aggregated_dataframe(st.session_state.cg_df, "network")
            else:
                agg_df = create_aggregated_dataframe(st.session_state.cg_df, "heatmap")
            
            # Always update agg_df in session state to ensure it's refreshed for the current visualization
            st.session_state.agg_df = agg_df
            if agg_df is not None:
                # Update company IDs based on the new agg_df
                buyer_ids = agg_df["buyer_id"].dropna().unique().tolist()
                seller_ids = agg_df["seller_id"].dropna().unique().tolist()
                st.session_state.company_ids = sorted(list(set(buyer_ids + seller_ids)))
        
        # Handle different visualisation types
        if viz_type == "Buyer-Seller Relationship":
            # Render parameters UI and get values
            params, generate_viz = render_relationship_params()
            # Handle visualisation generation
            handle_relationship_viz(params, generate_viz)
        
        elif viz_type == "Individual Company Network":
            # Render parameters UI and get values
            params, generate_viz = render_company_network_params(st.session_state.company_ids)
            # Handle visualisation generation
            handle_company_network_viz(params, generate_viz)
        
        elif viz_type == "Network Timelapse":
            # Render parameters UI and get values
            params, generate_viz = render_network_timelapse_params()
            # Handle visualisation generation
            handle_network_timelapse_viz(params, generate_viz)
            
        elif viz_type == "Heatmap":
            # Render parameters UI and get values
            params, generate_viz = render_heatmap_params()
            # Handle visualisation generation
            handle_heatmap_viz(params, generate_viz)
            
        elif viz_type == "Packing Week Heatmap":
            # Render parameters UI and get values
            params, generate_viz = render_packing_week_heatmap_params()
            # Handle visualisation generation
            handle_packing_week_heatmap_viz(params, generate_viz)
            
        elif viz_type == "Concentration Bubble Plot":
            # Render parameters UI and get values
            params, generate_viz = render_concentration_bubble_params()
            # Handle visualisation generation
            handle_concentration_bubble_viz(params, generate_viz)
        
        # Display the current visualisation
        visualization = st.session_state.visualization
        render_current_visualisation(visualization["html"], visualization["plot"])
        
        # Display visualization statistics if available
        if visualization["type"] is not None:
            # Create an expander for statistics
            with st.expander("Statistics", expanded=True):
                # Get and display statistics using the unified format_stats function
                stats = format_stats(visualization)
                if stats:
                    display_visualization_stats(stats)
                else:
                    st.info("No statistics available for this visualization type.")
        
        # Use legacy approach as fallback during transition
        elif st.session_state.viz_data is not None:
            # Create an expander for statistics
            with st.expander("Statistics", expanded=True):
                viz_data = st.session_state.viz_data
                viz_type = viz_data.get("type")
                
                # Format statistics based on visualization type
                stats = None
                if viz_type in ["relationship", "company_network", "network_timelapse"]:
                    # For network visualizations
                    G = viz_data.get("graph")
                    params = viz_data.get("params")
                    focal_info = viz_data.get("focal_info")
                    full_graph = viz_data.get("full_graph")
                    stats = format_network_stats(G, viz_type, params, focal_info, full_graph)
                elif viz_type in ["heatmap", "packing_week_heatmap"]:
                    # For heatmap visualizations
                    data = viz_data.get("data")
                    params = viz_data.get("params")
                    stats = format_heatmap_stats(data, viz_type, params)
                elif viz_type == "concentration_bubble":
                    # For bubble plot visualizations
                    data = viz_data.get("data")
                    params = viz_data.get("params")
                    stats = format_bubble_stats(data, params)
                
                # Display the formatted statistics
                if stats:
                    display_visualization_stats(stats)
                else:
                    st.info("No statistics available for this visualization type.")
        
        # Fall back to the old network_data approach as last resort
        elif st.session_state.network_data is not None:
            with st.expander("Network Statistics", expanded=True):
                G = st.session_state.network_data["graph"]
                viz_type = st.session_state.network_data["type"]
                params = st.session_state.network_data["params"]
                focal_info = st.session_state.network_data.get("focal_info")
                full_graph = st.session_state.network_data.get("full_graph")
                
                # Format and display the network statistics
                stats = format_network_stats(G, viz_type, params, focal_info, full_graph)
                display_network_stats(stats)
    
    # Display data preview based on whether visualisation is active
    has_visualization = st.session_state.visualization["type"] is not None
    legacy_visualization = st.session_state.get("viz_data") is not None or st.session_state.network_data is not None
    data_preview_expanded = not (has_visualization or legacy_visualization)
    render_data_preview(st.session_state.cg_df, expanded=data_preview_expanded)


if __name__ == "__main__":
    main()
