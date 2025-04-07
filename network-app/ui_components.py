import streamlit as st
from pathlib import Path
import os


def setup_page_config():
    """Set up the page configuration and create necessary directories."""
    # Create .streamlit directory if it doesn't exist
    os.makedirs(".streamlit", exist_ok=True)

    # Create config.toml with custom theme
    config_path = Path(".streamlit/config.toml")
    if not config_path.exists():
        with open(config_path, "w") as f:
            f.write(
                """
    [theme]
    primaryColor = "#0066cc"
    backgroundColor = "#0e1117"
    secondaryBackgroundColor = "#1a2634"
    textColor = "#fafafa"
    font = "sans serif"
    """
            )

    # Set page configuration
    st.set_page_config(
        page_title="OneSwitch Data Explorer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown(
        """
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #0066cc;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #c2c2c2;
            margin-bottom: 2rem;
        }
        .logo-container {
            text-align: center;
            padding: 1rem 0;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_header():
    """Render the application header."""
    # Add logo to sidebar top
    st.sidebar.markdown(
        """
    <div style="text-align: center; padding: 1.5rem 0;">
        <img src="https://app.agrigateone.com/assets/icons/logo.png" alt="Logo" width="100px">
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Application title with custom styling
    st.markdown('<h1 class="main-header">OneSwitch Data Explorer</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Explore data through a set of visualisations</p>',
        unsafe_allow_html=True,
    )


def render_data_filters():
    """Render data filter options in the sidebar."""
    st.sidebar.header("Data Source")

    # Select season year
    season_year = st.sidebar.selectbox(
        "Season",
        [2022, 2023, 2024, 2025],
        index=2,  # Default to 2024
        help="Select the season",
    )

    # Select commodity group
    commodity_group = st.sidebar.selectbox(
        "Commodity Group",
        ["Citrus", "Grape", "Pome", "Stone"],
        index=0,  # Default to Citrus
        help="Select the commodity group",
    )

    # Load data button
    load_data = st.sidebar.button("Load Data")

    return season_year, commodity_group, load_data


def render_visualisation_selector():
    """Render visualisation type selector in the sidebar."""
    st.sidebar.header("Visualisation")

    # Select visualisation type
    viz_type = st.sidebar.selectbox(
        "Select Visualisation Type",
        [
            "Buyer-Seller Relationship", 
            "Individual Company Network", 
            "Network Timelapse", 
            "Heatmap", 
            "Packing Week Heatmap",
            "Concentration Bubble Plot"
        ],
    )

    return viz_type


def render_relationship_params():
    """Render parameters for visualisation."""
    st.sidebar.subheader("Visualisation parameters")

    # Parameters for visualize_strong_connections
    rank = st.sidebar.slider(
        "Rank", 1, 10, 1, help="Rank of the edge to focus on (1 = strongest)"
    )

    measure = st.sidebar.selectbox(
        "Measure",
        ["containers", "std_cartons", "revenue"],
        help="Measure to use for edge weights",
    )

    degree = st.sidebar.slider(
        "Degree", 1, 3, 1, help="Degree of neighbors to include"
    )

    layout = st.sidebar.selectbox(
        "Layout Algorithm",
        ["spring", "kamada_kawai", "circular"],
        help="Network layout algorithm",
    )

    # Default iterations value (not shown in UI)
    iterations = 100

    # Button to generate visualisation
    generate_viz = st.sidebar.button("Generate visualisation")

    return {
        "rank": rank,
        "measure": measure,
        "degree": degree,
        "layout": layout,
        "iterations": iterations,
    }, generate_viz


def render_company_network_params(company_ids):
    """Render parameters for visualisation."""
    st.sidebar.subheader("Visualisation parameters")

    # Get company IDs from the list
    all_companies = company_ids if company_ids else ["No companies found"]

    # Parameters for visualize_company_network
    company_id = st.sidebar.selectbox(
        "Company ID", all_companies, help="ID of the company to focus on"
    )

    measure = st.sidebar.selectbox(
        "Measure",
        ["containers", "std_cartons", "revenue"],
        help="Measure to use for edge weights",
    )

    degree = st.sidebar.slider(
        "Degree", 1, 3, 1, help="Degree of neighbors to include"
    )

    layout = st.sidebar.selectbox(
        "Layout Algorithm",
        ["spring", "kamada_kawai", "circular"],
        index=2,  # Default to circular
        help="Network layout algorithm",
    )

    # Default iterations value (not shown in UI)
    iterations = 100

    # Button to generate visualisation
    generate_viz = st.sidebar.button("Generate visualisation")

    return {
        "company_id": company_id,
        "measure": measure,
        "degree": degree,
        "layout": layout,
        "iterations": iterations,
    }, generate_viz


def render_network_timelapse_params():
    """Render parameters for Network Timelapse visualisation."""
    st.sidebar.subheader("Network Timelapse Parameters")

    # Parameters for create_network_timelapse
    interval = st.sidebar.slider(
        "Frame Interval (ms)",
        500,
        3000,
        1500,
        100,
        help="Milliseconds between frames",
    )

    fps = st.sidebar.slider(
        "Frames Per Second", 1, 10, 1, help="Frames per second for saved animation"
    )

    seed = st.sidebar.slider(
        "Random Seed", 1, 100, 42, help="Random seed for layout generation"
    )

    # Button to generate visualisation
    generate_viz = st.sidebar.button("Generate Network Timelapse")

    return {
        "interval": interval,
        "fps": fps,
        "seed": seed,
    }, generate_viz


def render_current_visualisation(current_html, current_plot):
    """Render the current visualisation."""
    if current_html:
        st.markdown(current_html, unsafe_allow_html=True)
    elif current_plot:
        st.pyplot(current_plot)


def render_heatmap_params():
    """Render parameters for Heatmap visualisation."""
    st.sidebar.subheader("Heatmap Parameters")

    # All available variables
    all_variables = [
        "commodity_name", "variety_name", "production_region", "seller_id", "buyer_id", 
        "local_market", "target_country", "jbin", "size_count"
    ]
    
    # Initialize session state tracking variables with different keys than the widget keys
    if "heatmap_row_var_value" not in st.session_state:
        st.session_state.heatmap_row_var_value = "commodity_name"
    if "heatmap_col_var_value" not in st.session_state:
        st.session_state.heatmap_col_var_value = "local_market"
    
    # Function to update the column variable when row changes
    def on_row_change():
        row_var = st.session_state.heatmap_row_var
        
        # If the selected row matches the current column, change the column
        if row_var == st.session_state.heatmap_col_var_value:
            # Find a variable that's not the selected row
            available_cols = [var for var in all_variables if var != row_var]
            if available_cols:
                st.session_state.heatmap_col_var_value = available_cols[0]
        
        # Always update our tracking variable
        st.session_state.heatmap_row_var_value = row_var
    
    # Function to update the row variable when column changes
    def on_col_change():
        col_var = st.session_state.heatmap_col_var
        
        # If the selected column matches the current row, change the row
        if col_var == st.session_state.heatmap_row_var_value:
            # Find a variable that's not the selected column
            available_rows = [var for var in all_variables if var != col_var]
            if available_rows:
                st.session_state.heatmap_row_var_value = available_rows[0]
        
        # Always update our tracking variable
        st.session_state.heatmap_col_var_value = col_var
    
    # Select row and column parameters
    row_col = st.sidebar.selectbox(
        "Row Variable", 
        all_variables,
        index=all_variables.index(st.session_state.heatmap_row_var_value),
        key="heatmap_row_var",
        on_change=on_row_change,
        help="Variable to use for rows"
    )
    
    # Create the list of available column variables (excluding the selected row)
    available_col_vars = [var for var in all_variables if var != row_col]
    
    # If the current column value is not in available_col_vars, reset it to the first available
    if st.session_state.heatmap_col_var_value not in available_col_vars and available_col_vars:
        st.session_state.heatmap_col_var_value = available_col_vars[0]
    
    col_col = st.sidebar.selectbox(
        "Column Variable", 
        available_col_vars,
        index=available_col_vars.index(st.session_state.heatmap_col_var_value) if st.session_state.heatmap_col_var_value in available_col_vars else 0,
        key="heatmap_col_var",
        on_change=on_col_change,
        help="Variable to use for columns"
    )
    
    # Add measure selection
    measure = st.sidebar.selectbox(
        "Measure",
        ["containers", "std_cartons", "revenue"],
        help="Measure to use for cell values"
    )
    
    # Map the selected measure to the appropriate value column
    if measure == "containers":
        value_col = "container_number"
    elif measure == "std_cartons":
        value_col = "std_cartons"
    else:  # revenue
        value_col = "income"

    # Statistical significance parameters
    significance_level = st.sidebar.slider(
        "Significance Level", 
        0.01, 0.1, 0.05, 0.01,
        help="Alpha level for statistical significance (lower = more conservative)"
    )
    
    correct_multiple_tests = st.sidebar.checkbox(
        "Correct for Multiple Tests", 
        value=True,
        help="Apply correction for multiple hypothesis testing"
    )
    
    min_effect_size = st.sidebar.slider(
        "Minimum Effect Size (%)", 
        0.5, 10.0, 2.5, 0.5,
        help="Minimum percentage point difference for significance"
    )
    
    # Button to generate visualisation
    generate_viz = st.sidebar.button("Generate Heatmap")

    return {
        "row_col": row_col,
        "col_col": col_col,
        "value_col": value_col,
        "significance_level": significance_level,
        "correct_multiple_tests": correct_multiple_tests,
        "min_effect_size": min_effect_size,
    }, generate_viz


def render_packing_week_heatmap_params():
    """Render parameters for Packing Week Heatmap visualisation."""
    st.sidebar.subheader("Packing Week Heatmap Parameters")

    # All available variables (excluding packing_week which is fixed as the row)
    available_variables = [
        "commodity_name", "variety_name", "production_region", "seller_id", "buyer_id", 
        "local_market", "target_country", "jbin", "size_count"
    ]
    
    # Initialize the session state tracking variable with a different key than the widget key
    if "pw_heatmap_col_var_value" not in st.session_state:
        st.session_state.pw_heatmap_col_var_value = "local_market"
    
    # Function to update our tracking variable when the column changes
    def on_col_change():
        # Update our tracking variable with the widget value
        st.session_state.pw_heatmap_col_var_value = st.session_state.pw_heatmap_col_var

    # Column parameter (row is always packing_week)
    col_col = st.sidebar.selectbox(
        "Column Variable", 
        available_variables,
        index=available_variables.index(st.session_state.pw_heatmap_col_var_value) if st.session_state.pw_heatmap_col_var_value in available_variables else 0,
        key="pw_heatmap_col_var",
        on_change=on_col_change,
        help="Variable to use for columns"
    )
    
    # Add measure selection
    measure = st.sidebar.selectbox(
        "Measure",
        ["containers", "std_cartons", "revenue"],
        help="Measure to use for cell values"
    )
    
    # Map the selected measure to the appropriate value column
    if measure == "containers":
        value_col = "container_number"
    elif measure == "std_cartons":
        value_col = "std_cartons"
    else:  # revenue
        value_col = "income"
    
    # Visualization parameters
    title = st.sidebar.text_input(
        "Title", 
        f"Heatmap of Packing Week by {col_col.replace('_', ' ').title()}",
        help="Title for the heatmap"
    )
    
    normalize_rows = st.sidebar.checkbox(
        "Normalize Rows", 
        value=True,
        help="Normalize each row (percentages sum to 1)"
    )
    
    cmap = st.sidebar.selectbox(
        "Color Map", 
        ["YlGnBu", "viridis", "plasma", "inferno", "magma", "cividis", "Blues", "Greens", "Reds"],
        help="Color map for the heatmap"
    )
    
    # Statistical significance parameters
    significance_level = st.sidebar.slider(
        "Significance Level", 
        0.01, 0.1, 0.05, 0.01,
        help="Alpha level for statistical significance (lower = more conservative)"
    )
    
    correct_multiple_tests = st.sidebar.checkbox(
        "Correct for Multiple Tests", 
        value=True,
        help="Apply correction for multiple hypothesis testing"
    )
    
    min_effect_size = st.sidebar.slider(
        "Minimum Effect Size (%)", 
        0.5, 10.0, 2.5, 0.5,
        help="Minimum percentage point difference for significance"
    )
    
    top_n = st.sidebar.slider(
        "Top N Columns", 
        5, 30, 15, 1,
        help="Number of top columns to show (others are grouped)"
    )
    
    # Button to generate visualisation
    generate_viz = st.sidebar.button("Generate Packing Week Heatmap")

    return {
        "col_col": col_col,
        "value_col": value_col,
        "significance_level": significance_level,
        "correct_multiple_tests": correct_multiple_tests,
        "min_effect_size": min_effect_size,
    }, generate_viz


def render_concentration_bubble_params():
    """Render parameters for Concentration Bubble Plot visualisation."""
    st.sidebar.subheader("Concentration Bubble Plot Parameters")

    # All available variables for grouping
    primary_variables = [
        "commodity_name", "variety_name", "production_region", "seller_id", "buyer_id", 
        "local_market", "target_country", "jbin", "size_count", "packing_week"
    ]
    
    # Variables for concentration measurement
    secondary_variables = [
        "buyer_id", "seller_id", "target_country", "local_market", "production_region"
    ]
    
    # Initialize session state tracking variables with different keys than the widget keys
    if "bubble_primary_var_value" not in st.session_state:
        st.session_state.bubble_primary_var_value = "commodity_name"
    if "bubble_secondary_var_value" not in st.session_state:
        st.session_state.bubble_secondary_var_value = "buyer_id"
    
    # Function to update the secondary variable when primary changes
    def on_primary_change():
        primary_var = st.session_state.bubble_primary_var
        
        # If the selected primary matches the current secondary, change the secondary
        if primary_var == st.session_state.bubble_secondary_var_value:
            # Find a variable that's not the selected primary
            available_secondary = [var for var in secondary_variables if var != primary_var]
            if available_secondary:
                st.session_state.bubble_secondary_var_value = available_secondary[0]
        
        # Always update our tracking variable
        st.session_state.bubble_primary_var_value = primary_var
    
    # Function to update the primary variable when secondary changes
    def on_secondary_change():
        secondary_var = st.session_state.bubble_secondary_var
        
        # If the selected secondary matches the current primary, change the primary
        if secondary_var == st.session_state.bubble_primary_var_value:
            # Find a variable that's not the selected secondary
            available_primary = [var for var in primary_variables if var != secondary_var]
            if available_primary:
                st.session_state.bubble_primary_var_value = available_primary[0]
        
        # Always update our tracking variable
        st.session_state.bubble_secondary_var_value = secondary_var

    # Select entity parameters
    entity1_col = st.sidebar.selectbox(
        "Primary Variable (Grouping)", 
        primary_variables,
        index=primary_variables.index(st.session_state.bubble_primary_var_value),
        key="bubble_primary_var",
        on_change=on_primary_change,
        help="Main grouping variable"
    )
    
    # Create filtered list of secondary variables (excluding the primary)
    available_secondary_vars = [var for var in secondary_variables if var != entity1_col]
    
    # If the current secondary value is not in available_secondary_vars, reset it to the first available
    if st.session_state.bubble_secondary_var_value not in available_secondary_vars and available_secondary_vars:
        st.session_state.bubble_secondary_var_value = available_secondary_vars[0]
    
    entity2_col = st.sidebar.selectbox(
        "Secondary Variable (Concentration)", 
        available_secondary_vars,
        index=available_secondary_vars.index(st.session_state.bubble_secondary_var_value) if st.session_state.bubble_secondary_var_value in available_secondary_vars else 0,
        key="bubble_secondary_var",
        on_change=on_secondary_change,
        help="Variable to measure concentration for"
    )
    
    # Add measure selection
    measure = st.sidebar.selectbox(
        "Measure",
        ["containers", "std_cartons", "revenue"],
        help="Measure to use for concentration calculation"
    )
    
    # Map the selected measure to the appropriate value column
    if measure == "containers":
        value_col = "container_number"
    elif measure == "std_cartons":
        value_col = "std_cartons"
    else:  # revenue
        value_col = "income"
    
    # Visualization parameters
    title = st.sidebar.text_input(
        "Title", 
        f"{entity2_col.replace('_', ' ').title()} Concentration by {entity1_col.replace('_', ' ').title()}",
        help="Title for the bubble plot"
    )
    
    min_pallets = st.sidebar.slider(
        "Minimum Pallets", 
        0, 50, 10, 5,
        help="Minimum number of pallets for inclusion"
    )
    
    label_threshold = st.sidebar.slider(
        "Label Threshold", 
        0.01, 0.2, 0.05, 0.01,
        help="Threshold for showing labels (% of max)"
    )
    
    # Button to generate visualisation
    generate_viz = st.sidebar.button("Generate Concentration Bubble Plot")

    return {
        "entity1_col": entity1_col,
        "entity2_col": entity2_col,
        "value_col": value_col,
        "measure": measure,
        "title": title,
        "min_pallets": min_pallets,
        "label_threshold": label_threshold,
    }, generate_viz


def render_data_preview(cg_df, expanded=True):
    """Render a preview of the loaded data."""
    with st.expander("Data preview", expanded=expanded):
        if cg_df is not None:
            # Display data stats from the aggregated dataframe
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Carton Groupings", len(cg_df))
            with col2:
                st.metric("Unique Buyers", cg_df["buyer_id"].nunique())
            with col3:
                st.metric("Unique Sellers", cg_df["seller_id"].nunique())

            # Show both aggregated data and raw preview
            st.subheader("Carton Grouping Data")
            st.dataframe(cg_df.head(5))
        else:
            st.error("Data not loaded. Please load the data to view the preview.") 