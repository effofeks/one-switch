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
        '<p class="sub-header">Visualise relationships and uncover statistically significant patterns in trade network data.</p>',
        unsafe_allow_html=True,
    )


def render_data_filters():
    """Render data filter options in the sidebar."""
    st.sidebar.header("Data Source")

    # Select season year
    season = st.sidebar.selectbox(
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

    # Load data flag
    load_data = st.sidebar.button("Load Data")

    return season, commodity_group, load_data


def render_visualisation_selector():
    """Render visualisation type selector in the sidebar."""
    st.sidebar.header("Visualisation")

    # Select main visualisation type
    viz_type = st.sidebar.selectbox(
        "Select Visualisation Type",
        ["Network", "Heatmap"],
        index=1,  # Default to Heatmap
    )
    
    # Initialize session state for network subtype if needed
    if "network_subtype" not in st.session_state:
        st.session_state.network_subtype = "by strongest relationship"
        
    # If Network is selected, show sub-type options
    if viz_type == "Network":
        network_subtype = st.sidebar.selectbox(
            "Network type",
            ["by strongest relationship", "by company", "timelapse"],
            key="network_subtype"
        )
        
        # Map network subtype to original visualisation type
        if network_subtype == "by strongest relationship":
            return "Buyer-Seller Relationship"
        elif network_subtype == "by company":
            return "Individual Company Network"
        elif network_subtype == "timelapse":
            return "Network Timelapse"
    
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
    # Check if the current visualization is a heatmap type
    is_heatmap = False
    if "visualisation" in st.session_state and st.session_state.visualisation["type"]:
        viz_type = st.session_state.visualisation["type"]
        if "heatmap" in viz_type.lower() or viz_type == "Heatmap":
            is_heatmap = True

    if current_html:
        # For HTML visualizations like network timelapse
        # Add interpretation expander for heatmaps if applicable (though unlikely with HTML)
        if is_heatmap:
            from viz_utils import get_heatmap_interpretation
            # with st.expander("📖 Interpretation Guide", expanded=False):
            #     st.markdown(get_heatmap_interpretation())
        
        st.markdown(current_html, unsafe_allow_html=True)
    elif current_plot:
        # Import the export function
        from viz_utils import export_plot_as_png, get_heatmap_interpretation
        
        # Get a descriptive filename based on session state if available
        if "visualisation" in st.session_state and st.session_state.visualisation["type"]:
            viz_type = st.session_state.visualisation["type"]
            if viz_type == "standard_heatmap":
                # For heatmap, use row_col and col_col in the filename
                params = st.session_state.visualisation["params"]
                row_col = params.get("row_col", "").replace("_", "-")
                col_col = params.get("col_col", "").replace("_", "-")
                filename = f"heatmap-{row_col}-by-{col_col}.png"
            elif "network" in viz_type.lower():
                # For network visualizations
                filename = f"{viz_type.lower().replace(' ', '-')}.png"
            else:
                # Default filename
                filename = f"plot-{viz_type.lower().replace(' ', '-')}.png"
        else:
            filename = "plot.png"
        
        # For heatmap visualizations, add interpretation guide
        if is_heatmap:
            # with st.expander("📖 Interpretation Guide", expanded=False):
            #     st.markdown(get_heatmap_interpretation())
                
            # Add usage example expander below the interpretation guide
            with st.expander("🎓 Usage Example", expanded=True):
                st.markdown("""
For 2024 citrus data, we explore the relationship between variety and target country for standard carton volumes. Consider the **Star Ruby** and **Japan** pair with a **cell value of 12%**:


What the cell value means:
- The 12% is a row-normalized proportion.
- This means 12% of all Star Ruby volume went to Japan.
- Each row (variety) sums to 100%, showing how that variety is distributed across countries.

What the statistical test examines:
- The Chi-squared test for independence asks: _Is the distribution of Star Ruby across countries different from what we’d expect if variety and country were unrelated?_
- The asterisk (*) means the observed value (12%) is significantly higher than expected under this assumption of independence.

In practical terms:
- The cell values show how a variety’s volume is split across target countries.
- Significance markers (* or †) flag when this pattern is unusual — either stronger or weaker than what the overall data trends would predict.
- A significant result suggests the variety-country pair has a distinctive relationship, not explained by general popularity alone.
- The chosen significance level of 5% means that cells are flagged as significant if the observed value is less than 5% likely to occur by chance.
- The chosen minimum effect size of 5% means that cells are flagged as significant if the observed value is at least 5% different from what would be expected by chance.
 - Together, the significance level and minimum effect size ensure that only substantial and reliable deviations are highlighted.

""")
        
        # Create a container for the export button and align it to the right
        _, right_col = st.columns([4, 1])
        with right_col:
            # Add the export button
            export_plot_as_png(current_plot, filename=filename)
            
        # Show the plot in a separate row
        st.pyplot(current_plot)


def render_heatmap_params():
    """Render parameters for Heatmap visualisation."""
    st.sidebar.subheader("Heatmap Parameters")

    # All available variables
    all_variables = [
        "commodity_name", "variety_name", "production_region", "seller_id", "buyer_id", 
        "local_market", "target_country", "jbin", "size_count", "size_categorization", "class",
        "packing_week"
    ]
    
    # Initialize session state tracking variables with different keys than the widget keys
    if "heatmap_row_var_value" not in st.session_state:
        st.session_state.heatmap_row_var_value = "variety_name"  # Default to variety_name
    if "heatmap_col_var_value" not in st.session_state:
        st.session_state.heatmap_col_var_value = "target_country"  # Default to target_country
    
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
        index=1,  # Default to std_cartons
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
        0.5, 10.0, 5.0, 0.5,
        help="Minimum percentage point difference for significance"
    )
    
    # Button to generate visualisation
    generate_viz = st.sidebar.button("Generate Heatmap")
    
    # Check if this is a packing week heatmap
    is_packing_week_heatmap = (row_col == "packing_week")
    
    # Return parameters based on whether this is a packing week heatmap
    return {
        "row_col": row_col,
        "col_col": col_col,
        "value_col": value_col,
        "significance_level": significance_level,
        "correct_multiple_tests": correct_multiple_tests,
        "min_effect_size": min_effect_size,
        "is_packing_week_heatmap": is_packing_week_heatmap,
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
    generate_viz = st.sidebar.button("Generate Packing Week Heatmap")

    return {
        "col_col": col_col,
        "value_col": value_col,
        "significance_level": significance_level,
        "correct_multiple_tests": correct_multiple_tests,
        "min_effect_size": min_effect_size,
    }, generate_viz


def render_data_preview(df, expanded=True):
    """Render a preview of the loaded data."""
    with st.expander("👀 Data Preview", expanded=expanded):
        if df is not None:
            st.dataframe(df.head(10))
        else:
            st.info("No data loaded. Use the filters in the sidebar to load data.") 