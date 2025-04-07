import streamlit as st
from state_manager import (
    clear_visualisation_state, 
    generate_strong_connections_viz,
    generate_company_network_viz,
    generate_network_timelapse_viz,
    generate_heatmap_viz,
    generate_packing_week_heatmap_viz,
    generate_concentration_bubble_viz
)


def handle_relationship_viz(params, generate_viz):
    """Handle the Strong Connections visualization generation process."""
    if generate_viz:
        # Set the flag and clear visualization state
        clear_visualisation_state()
        st.session_state.generating_strong_connections = True
        st.rerun()  # Rerun to clear UI before generating

    # Check if we need to generate the visualization after a rerun
    if st.session_state.generating_strong_connections:
        # Reset the flag
        st.session_state.generating_strong_connections = False
        
        with st.spinner("Generating Strong Connections visualisation..."):
            generate_strong_connections_viz(params)


def handle_company_network_viz(params, generate_viz):
    """Handle the Company Network visualization generation process."""
    if generate_viz:
        # Set the flag and clear visualization state
        clear_visualisation_state()
        st.session_state.generating_company_network = True
        st.rerun()  # Rerun to clear UI before generating

    # Check if we need to generate the visualization after a rerun
    if st.session_state.generating_company_network:
        # Reset the flag
        st.session_state.generating_company_network = False
        
        with st.spinner("Generating Company Network visualisation..."):
            generate_company_network_viz(params)


def handle_network_timelapse_viz(params, generate_viz):
    """Handle the Network Timelapse visualization generation process."""
    if generate_viz:
        # Set the flag and clear visualization state
        clear_visualisation_state()
        st.session_state.generating_network_timelapse = True
        st.rerun()  # Rerun to clear UI before generating

    # Check if we need to generate the visualization after a rerun
    if st.session_state.generating_network_timelapse:
        # Reset the flag
        st.session_state.generating_network_timelapse = False
        
        with st.spinner("Generating Network Timelapse... this may take a moment"):
            generate_network_timelapse_viz(params)


def handle_heatmap_viz(params, generate_viz):
    """Handle the Heatmap visualization generation process."""
    if generate_viz:
        # Set the flag and clear visualization state
        clear_visualisation_state()
        st.session_state.generating_heatmap = True
        st.rerun()  # Rerun to clear UI before generating

    # Check if we need to generate the visualization after a rerun
    if st.session_state.generating_heatmap:
        # Reset the flag
        st.session_state.generating_heatmap = False
        
        with st.spinner("Generating Heatmap visualisation..."):
            generate_heatmap_viz(params)


def handle_packing_week_heatmap_viz(params, generate_viz):
    """Handle the Packing Week Heatmap visualization generation process."""
    if generate_viz:
        # Set the flag and clear visualization state
        clear_visualisation_state()
        st.session_state.generating_packing_week_heatmap = True
        st.rerun()  # Rerun to clear UI before generating

    # Check if we need to generate the visualization after a rerun
    if st.session_state.generating_packing_week_heatmap:
        # Reset the flag
        st.session_state.generating_packing_week_heatmap = False
        
        with st.spinner("Generating Packing Week Heatmap visualisation..."):
            generate_packing_week_heatmap_viz(params)


def handle_concentration_bubble_viz(params, generate_viz):
    """Handle the Concentration Bubble Plot visualization generation process."""
    if generate_viz:
        # Set the flag and clear visualization state
        clear_visualisation_state()
        st.session_state.generating_concentration_bubble = True
        st.rerun()  # Rerun to clear UI before generating

    # Check if we need to generate the visualization after a rerun
    if st.session_state.generating_concentration_bubble:
        # Reset the flag
        st.session_state.generating_concentration_bubble = False
        
        with st.spinner("Generating Concentration Bubble Plot visualisation..."):
            generate_concentration_bubble_viz(params) 