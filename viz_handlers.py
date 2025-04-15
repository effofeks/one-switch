import streamlit as st
from state_manager import (
    clear_visualisation_state,
    create_strong_connections_network,
    create_company_network,
    create_network_timelapse,
    create_standard_heatmap,
    create_packing_week_heatmap,
)


def handle_relationship_viz(params, generate_viz):
    """Handle the Strong Connections visualisation generation process."""
    if generate_viz:
        # Clear visualisation state
        clear_visualisation_state()
        # Generate visualisation
        with st.spinner("Generating Strong Connections visualisation..."):
            create_strong_connections_network(params)


def handle_company_network_viz(params, generate_viz):
    """Handle the Company Network visualisation generation process."""
    if generate_viz:
        # Clear visualisation state
        clear_visualisation_state()
        # Generate visualisation
        with st.spinner("Generating Company Network visualisation..."):
            create_company_network(params)


def handle_network_timelapse_viz(params, generate_viz):
    """Handle the Network Timelapse visualisation generation process."""
    if generate_viz:
        # Clear visualisation state
        clear_visualisation_state()
        # Generate visualisation
        with st.spinner("Generating Network Timelapse... this may take a moment"):
            create_network_timelapse(params)


def handle_heatmap_viz(params, generate_viz):
    """Handle the Heatmap visualisation generation process."""
    if generate_viz:
        # Clear visualisation state
        clear_visualisation_state()
        # Generate visualisation
        with st.spinner("Generating Heatmap visualisation..."):
            create_standard_heatmap(params)


def handle_packing_week_heatmap_viz(params, generate_viz):
    """Handle the Packing Week Heatmap visualisation generation process."""
    if generate_viz:
        # Clear visualisation state
        clear_visualisation_state()
        # Generate visualisation
        with st.spinner("Generating Packing Week Heatmap visualisation..."):
            create_packing_week_heatmap(params)
