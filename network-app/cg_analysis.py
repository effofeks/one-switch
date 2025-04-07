import datetime
import pandas as pd
import numpy as np


# imports
import os
import sqlalchemy
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import URL

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from collections import Counter
import seaborn as sns

import pandas as pd
import numpy as np
import os
import sqlalchemy
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy.dialects import postgresql
from sqlalchemy import URL

import matplotlib.dates as mdates
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from matplotlib.backends.backend_pdf import PdfPages


load_dotenv()
# constants
DB_URL = URL.create(
    drivername="postgresql",
    username="warehouse",
    password="warehouse",
    host="localhost",
    port=8432,
    database="warehouse",
)

# DB_URL = f"postgresql://{os.environ['DBT_DEV_USER']}:{os.environ['DBT_DEV_PASSWORD']}@{os.environ['DBT_DEV_HOST']}:5432/postgres"

SCRIPTS_PATH = Path("scripts")
DATA_PATH = Path("data")
OUTPUT_CG_PATH = Path("output/cg-analysis")
OUTPUT_NET_PATH = Path("output/network-analysis")


CG_COL_TYPES = {
    "id": "Int64",
    "line_item_id": "Int64",
    "order_id": "Int64",
    "state": str,
    "pallet_number": str,
    "sequence_number": "Int64",
    "exporter_code": str,
    "farm_code": str,
    "packhouse_code": str,
    "production_region": str,
    "commodity_name": str,
    "variety_name": str,
    "cartons": "Int64",
    "pallet_stack": str,
    "pack": str,
    "size_count": str,
    "size_categorization": str,
    "grade": str,
    "orchard": str,
    "net_mass": "Float64",
    "container_number": str,
    "mark": str,
    "local_market": str,
    "jbin": str,
    "target_market": str,
    "target_region": str,
    "target_country": str,
    "pallet_gross_mass": "Float64",
    "seller_id": "Int64",
    "seller_types": str,
    "buyer_id": "Int64",
    "buyer_types": str,
    "packing_week": str,
    "batch_number": str,
    "inventory_code": str,
    "consignment_number": str,
    "pallet_rejected": "bool",
    "commercial_term_id": "Int64",
    "advance_price": "Float64",
    "final_price": "Float64",
    "currency": str,
    "transport_type": str,
}

LI_COL_TYPES = {
    "id": "Int64",
    "quantity": "Float64",
    "quantity_unit": str,
    "price_minor_unit": "Int64",
    "price_unit": str,
    "currency": str,
    "pack": str,
    "price_term": str,
    "additional_fields": str,
    "pallet_stack": str,
    "unlimited": "bool",
    "target_market": str,
    "target_region": str,
    "target_country": str,
    "packing_week": str,
    "incoterm": str,
    "deleted": "bool",
    "grade": str,
    "state": str,
    "line_item_grouping_id": "Int64",
    "rank": "Int64",
    "batch_number": str,
    "inventory_code": str,
    "planned_quantity": "Float64",
    "planned_quantity_unit": str,
    "size_counts": str,
}

PT_COL_TYPES = {
    "cg_id": "Int64",
    "line_item_id": "Int64",
    "container_id": "Int64",
    "state": str,
    "local_market": str,
    "steri_market": str,
    "jbin": str,
    "tradelane": str,
    "ls_tradelane": str,
    "ss_tradelane": str,
    "std_cartons": "Float64",
    "enter_packhouse": str,
    "packhouse_dwell": "Float64",
    "exit_packhouse": str,
    "dispatch_dwell": "Float64",
    "enter_coldstore": str,
    "cs_fbo_code": str,
    "cold_store_dwell": "Float64",
    "exit_coldstore": str,
    "current_coldstore": str,
    "cs_to_gi_dwell": "Float64",
    "origin_port_id": "Int64",
    "origin_port": str,
    "gi_to_load_dwell": "Float64",
    "load_vessel": str,
    "load_to_dpl_dwell": "Float64",
    "dpl_voyage": str,
    "dpl_to_adp_dwell": "Float64",
    "destination_port": str,
    "destination_port_country_code": str,
    "discharge_to_gate_out_dwell": "Float64",
}

FIN_COL_TYPES = {
    "invoice_line_item_id": "Int64",
    "cg_id": "Int64",
    "company_id": "Int64",
    "document_type": str,
    "invoice_id": "Int64",
    "cost_type": str,
    "ili_price_minor_unit": "Float64",
    "currency": str,
    "price_unit": str,
    "cgt_amount": "Float64",
    "is_actual": "bool",
    "order_price": "Float64",
    "order_price_per_carton": "Float64",
    "order_price_unit": str,
    "order_currency": str,
    "ct_advance_amount_per_carton": "Float64",
    "ct_advance_currency": str,
    "ct_advance_week": str,
    "ct_final_value": "Float64",
    "ct_final_currency": str,
    "ct_advance_credit_term": str,
    "ct_final_credit_term": str,
    "incoterm": str,
    "actual_advance_currency": str,
    "advances_exchange_rate": "Float64",
    "actual_advance": "Float64",
    "advance_transaction_week": str,
    "actual_advance_zar": "Float64",
    "actual_final_currency": str,
    "finals_exchange_rate": "Float64",
    "actual_final": "Float64",
    "actual_final_zar": "Float64",
    "actual_final_per_pallet": "Float64",
    "actual_final_per_carton": "Float64",
    "total_value_per_pallet": "Float64",
    "final_transaction_week": str,
    "total_value_per_carton": "Float64",
    "exchange_rate": "Float64",
    "payment_status": str,
    "account_type": str,
}

FIN_ADOPTION_COL_TYPES = {
    "packing_week": str,
    "commodity_group": str,
    "total_cgs": "Int64",
    "finance_cgs": "Int64",
    "prop_finance_cgs": "Float64",
}

CG_DATETYPE_COLS = [
    "advance_due_date",
    "final_due_date",
    "packed_datetime",
    "first_event_datetime",
]

PT_DATETYPE_COLS = [
    "packed_datetime",
    "departed_packhouse_date",
    "enter_cs_date",
    "exit_cs_date",
    "gate_in_date",
    "load_date",
    "dpl_date",
    "adp_date",
    "discharge_date",
    "gate_out_date",
]

FIN_DATETYPE_COLS = [
    "stuff_date",
    "final_due_date",
    "advance_due_date",
    "invoice_date",
]

# supply
# seller side - look at carton groupings data to see which orders they could fulfull
# future work - enquire about estimations or pre-season planning data for a more accurate data set on supply
ASK_COL_NAMES = [
    "seller_id",
    "seller_types",
    "variety_name",
    "size_count",
    "grade",
    "packing_week",
    "farm_code",
    "orchard",
]

BID_COL_NAMES = [
    "buyer_id",
    "buyer_types",
    "variety_name",
    "size_count",
    "grade",
    "packing_week",
    "target_market",
    "target_region",
    "target_country",
]


# Read SQL queries from files
with open(SCRIPTS_PATH / "carton_groupings.sql", "r") as f:
    cg_query = f.read()

with open(SCRIPTS_PATH / "line_items.sql", "r") as f:
    li_query = f.read()

with open(SCRIPTS_PATH / "pallet_timeline.sql", "r") as f:
    pt_query = f.read()

with open(SCRIPTS_PATH / "dso_finance.sql", "r") as f:
    fin_query = f.read()

with open(SCRIPTS_PATH / "finance_adoption.sql", "r") as f:
    fin_adoption_query = f.read()


def create_heatmap(
    df,
    row_col,
    col_col,
    value_col,
    figsize=(16, 10),
    significance_level=0.05,
    correct_multiple_tests=True,
    min_effect_size=2.5,
):
    """
    Create a heatmap from the DataFrame with improved statistical significance testing.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data
    row_col : str
        Column name to use for rows (entity1)
    col_col : str
        Column name to use for columns (entity2)
    value_col : str
        Column name to use for values/counts
    figsize : tuple, optional
        Figure size (width, height)
    cmap : str, optional
        Colormap for the heatmap
    annot : bool, optional
        Whether to annotate the heatmap with values
    significance_level : float, optional
        Alpha level for statistical significance testing (default 0.05)
    correct_multiple_tests : bool, optional
        Whether to apply correction for multiple testing (default True)
    min_effect_size : float, optional
        Minimum percentage point difference required for significance testing (default 2.5)

    Returns:
    --------
    tuple: (pivot_df, weighted_avg_series, significance_df)
        - pivot_df: The original pivot table with raw counts
        - weighted_avg_series: Series containing weighted average percentages for each entity2
        - significance_df: DataFrame indicating which cells are significantly above average
    """
    import numpy as np
    import pandas as pd
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.stats.multitest import multipletests

    # Convert min_effect_size from percentage points to proportion
    min_effect_size_prop = min_effect_size / 100

    filtered_df = df.dropna(subset=[row_col, col_col])
    filtered_df = filtered_df[filtered_df[row_col] != "None"]
    filtered_df = filtered_df[filtered_df[col_col] != "None"]

    # Validate measure parameter
    if value_col not in ["container_number", "std_cartons", "income"]:
        raise ValueError(
            "measure must be either 'container_number', 'std_cartons', or 'income'"
        )

    # Create a pivot table
    if value_col == "container_number":
        pivot_df = filtered_df.pivot_table(
            index=row_col,
            columns=col_col,
            values=value_col,
            aggfunc="nunique",
            fill_value=0,
        )
    elif value_col == "std_cartons":  # measure == 'std_cartons'
        pivot_df = filtered_df.pivot_table(
            index=row_col,
            columns=col_col,
            values=value_col,
            aggfunc="sum",
            fill_value=0,
        )
    else:  # value_col == 'income'
        pivot_df = filtered_df.pivot_table(
            index=row_col,
            columns=col_col,
            values=value_col,
            aggfunc="sum",
            fill_value=0,
        )

    # Sort by row totals
    row_totals = pivot_df.sum(axis=1).sort_values(ascending=False)
    pivot_df = pivot_df.loc[row_totals.index]

    # Sort by column totals
    col_totals = pivot_df.sum(axis=0).sort_values(ascending=False)
    pivot_df = pivot_df[col_totals.index]

    # Calculate weighted averages
    entity1_totals = pivot_df.sum(axis=1)
    entity1_percentages = pivot_df.div(entity1_totals, axis=0)

    weighted_avg_by_entity2 = {}
    for entity2 in pivot_df.columns:
        weighted_sum = (entity1_percentages[entity2] * entity1_totals).sum()
        weighted_avg = weighted_sum / entity1_totals.sum()
        weighted_avg_by_entity2[entity2] = weighted_avg

    weighted_avg_series = pd.Series(weighted_avg_by_entity2).sort_values(
        ascending=False
    )

    # Collect p-values for all tests
    pvalues = []
    test_indices = []

    # Perform statistical significance testing with improved approach
    min_sample_size = 30  # Minimum sample size for reliable testing

    for row_idx in pivot_df.index:
        for col_idx in pivot_df.columns:
            # Skip cells with zero counts or small sample sizes
            if (
                pivot_df.loc[row_idx, col_idx] == 0
                or entity1_totals[row_idx] < min_sample_size
            ):
                continue

            # Get counts for this cell
            count = pivot_df.loc[row_idx, col_idx]
            row_total = entity1_totals[row_idx]

            # Calculate observed proportion
            observed_prop = count / row_total

            # Get weighted average (expected proportion)
            expected_prop = weighted_avg_by_entity2[col_idx]

            # Handle NaN or Infinity values
            if pd.isna(observed_prop) or pd.isna(expected_prop) or np.isinf(observed_prop) or np.isinf(expected_prop):
                continue

            # Only test if observed is higher than expected
            if observed_prop <= expected_prop:
                continue

            # Calculate effect size - how substantial is the difference?
            effect_size = observed_prop - expected_prop

            # Skip very small effect sizes (meaningful difference threshold) or NaN values
            if pd.isna(effect_size) or effect_size < min_effect_size_prop:
                continue

            # Perform z-test for proportions
            # Standard error for the difference between two proportions
            se = np.sqrt(
                expected_prop
                * (1 - expected_prop)
                * (1 / row_total + 1 / entity1_totals.sum())
            )

            # Z-score
            z_score = (observed_prop - expected_prop) / se

            # P-value (one-tailed test)
            p_value = 1 - stats.norm.cdf(z_score)

            # Store p-value and indices for later correction
            pvalues.append(p_value)
            test_indices.append((row_idx, col_idx))

    # Create a DataFrame to store significance results
    significance_df = pd.DataFrame(
        index=pivot_df.index, columns=pivot_df.columns, data=False
    )

    # Apply multiple testing correction if requested and if we have any tests
    if pvalues and correct_multiple_tests:
        # Use Benjamini-Hochberg FDR correction for multiple testing
        reject, pvals_corrected, _, _ = multipletests(
            pvalues, alpha=significance_level, method="fdr_bh"
        )

        # Update significance based on corrected p-values
        for i, (row_idx, col_idx) in enumerate(test_indices):
            is_significant = reject[i]
            significance_df.loc[row_idx, col_idx] = is_significant
    else:
        # Use uncorrected p-values
        for i, (row_idx, col_idx) in enumerate(test_indices):
            is_significant = pvalues[i] < significance_level
            significance_df.loc[row_idx, col_idx] = is_significant

    # Group rows and columns if they exceed max_categories
    max_categories = 19
    working_pivot = pivot_df.copy()
    working_significance_df = significance_df.copy()

    # First, handle columns (entity2)
    if len(working_pivot.columns) > max_categories:
        top_columns = col_totals.nlargest(max_categories).index.tolist()
        other_columns = [col for col in working_pivot.columns if col not in top_columns]

        # Create new pivot table with top columns and "Other" category
        new_pivot = pd.DataFrame(index=working_pivot.index)

        # Add top columns
        for col in top_columns:
            new_pivot[col] = working_pivot[col]

        # Add "Other" column
        if other_columns:
            new_pivot["Other"] = working_pivot[other_columns].sum(axis=1)

            # Calculate weighted average for "Other" category
            other_weighted_avg = sum(
                weighted_avg_by_entity2[col] for col in other_columns
            )
            weighted_avg_by_entity2["Other"] = other_weighted_avg

        # Create updated significance DataFrame for columns
        new_significance_df = pd.DataFrame(
            False, index=working_pivot.index, columns=new_pivot.columns
        )

        # Copy significance information for top columns
        for row_idx in working_pivot.index:
            for col_idx in top_columns:
                if (
                    row_idx in working_significance_df.index
                    and col_idx in working_significance_df.columns
                ):
                    new_significance_df.loc[row_idx, col_idx] = (
                        working_significance_df.loc[row_idx, col_idx]
                    )

        # "Other" column is marked as significant if any of its components were significant
        if "Other" in new_pivot.columns and other_columns:
            for row_idx in working_pivot.index:
                if row_idx in working_significance_df.index:
                    # Check if any of the grouped columns were significant for this row
                    for col in other_columns:
                        if (
                            col in working_significance_df.columns
                            and working_significance_df.loc[row_idx, col]
                        ):
                            new_significance_df.loc[row_idx, "Other"] = True
                            break

        working_pivot = new_pivot
        working_significance_df = new_significance_df

    # Now handle rows (entity1) independently
    if len(working_pivot.index) > max_categories:
        top_rows = row_totals.nlargest(max_categories).index.tolist()
        other_rows = [row for row in working_pivot.index if row not in top_rows]

        # Create new pivot with only top rows
        row_reduced_pivot = working_pivot.loc[top_rows].copy()

        # Add "Other" row if needed
        if other_rows:
            other_row_values = {}
            for col in working_pivot.columns:
                # Sum values for all other rows for this column
                other_row_values[col] = working_pivot.loc[other_rows, col].sum()

            # Add the "Other" row to the pivot
            row_reduced_pivot.loc["Other"] = pd.Series(other_row_values)

        # Create updated significance DataFrame for rows
        new_row_significance_df = pd.DataFrame(
            False, index=row_reduced_pivot.index, columns=working_pivot.columns
        )

        # Copy significance information for top rows
        for row_idx in top_rows:
            for col_idx in working_pivot.columns:
                if (
                    row_idx in working_significance_df.index
                    and col_idx in working_significance_df.columns
                ):
                    new_row_significance_df.loc[row_idx, col_idx] = (
                        working_significance_df.loc[row_idx, col_idx]
                    )

        # "Other" row is marked as significant if any of its components were significant
        if "Other" in row_reduced_pivot.index and other_rows:
            for col_idx in working_pivot.columns:
                if col_idx in working_significance_df.columns:
                    # Check if any of the grouped rows were significant for this column
                    for row in other_rows:
                        if (
                            row in working_significance_df.index
                            and working_significance_df.loc[row, col_idx]
                        ):
                            new_row_significance_df.loc["Other", col_idx] = True
                            break

        working_pivot = row_reduced_pivot
        working_significance_df = new_row_significance_df

    # Update our working DataFrames for visualization
    pivot_df_top = working_pivot
    significance_df_top = working_significance_df

    # Recalculate entity1_totals for the new pivot
    new_entity1_totals = pivot_df_top.sum(axis=1)

    # Normalise rows
    viz_df = pivot_df_top.copy()
    row_sums = viz_df.sum(axis=1)
    viz_df = viz_df.div(row_sums, axis=0)
    fmt = ".2f" 


    # Calculate appropriate figure height based on number of rows (minimum 12, increased from 10)
    # Add a multiplier to increase the overall height
    adjusted_height = max(12, len(viz_df) * 0.6)  # Increased from 0.5 to 0.6
    # Calculate appropriate figure width to accommodate the row totals column
    adjusted_width = figsize[0]

    # Create the plot with adjusted size
    fig = plt.figure(figsize=(adjusted_width, adjusted_height))

    # Calculate appropriate margins for the annotation text
    footnote_size = 9
    title_size = 16

    # Create the subplot with extra space for annotations
    # Use gridspec to create more flexible layout
    # Changed height ratios to give more space to the main heatmap and padding between sections
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(2, 1, height_ratios=[0.9, 0.1], figure=fig)
    ax = plt.subplot(gs[0])

    # Create the heatmap base
    heatmap = sns.heatmap(viz_df, cmap="YlGnBu", annot=False, linewidths=0.5, ax=ax)

    # Get the colormap normalized data for determining text color
    norm = plt.Normalize(viz_df.min().min(), viz_df.max().max())
    cmap_obj = plt.colormaps["YlGnBu"]

    # Add custom annotations with significance indicators
    for i, row_idx in enumerate(viz_df.index):
        for j, col_idx in enumerate(viz_df.columns):
            val = viz_df.loc[row_idx, col_idx]
            is_significant = significance_df_top.loc[row_idx, col_idx]

            # Get the actual color of the cell
            color_val = norm(val)
            rgb = cmap_obj(color_val)

            # Calculate luminance (brightness) of the background color
            # Using standard formula: 0.299*R + 0.587*G + 0.114*B
            luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

            # Choose text color based on background brightness
            # Threshold of 0.5 for a good contrast
            text_color = "white" if luminance < 0.5 else "black"
            # Only mark significant cells with *
            if is_significant:
                text = ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{val:.2f}*",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color=text_color,
                )
            else:
                text = ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                )

    # Add row totals to the right of the heatmap
    # Calculate position for row totals (just past the end of the heatmap)
    right_pos = len(viz_df.columns)

    # Add a title for the totals column    
    if value_col == "container_number":
        s = "Total Containers"
    elif value_col == "std_cartons":
        s = "Total Standard Cartons"
    else:
        s = "Total Revenue"
    ax.text(
        right_pos + 0.5,
        -0.5,
        s,
        ha="center",
        va="center",
        fontweight="bold",
        color="black",
    )

    # Add dividing line
    for i in range(-1, len(viz_df.index) + 1):
        ax.plot([right_pos, right_pos], [i, i + 1], "k-", lw=1)

    # Add row totals for each row
    for i, row_idx in enumerate(viz_df.index):
        row_total = new_entity1_totals[row_idx]
        ax.text(
            right_pos + 0.5,
            i + 0.5,
            f"{int(row_total):,}",
            ha="center",
            va="center",
            color="black",
        )

    # Create a formatted string of weighted averages for visible columns
    weighted_avg_info = "Weighted averages by column:\n"
    for col in viz_df.columns:
        avg_val = weighted_avg_by_entity2[col]
        weighted_avg_info += f"{col}: {avg_val:.3f}  "

        # Add line breaks to avoid too long lines
        if (list(viz_df.columns).index(col) + 1) % 3 == 0:
            weighted_avg_info += "\n"

    # Add note about the statistical testing in the footnote
    test_method = (
        "with FDR correction for multiple testing"
        if correct_multiple_tests
        else "without correction for multiple testing"
    )

    footnote_text = (
        f"* Statistically significant higher proportion at α={significance_level} {test_method}\n"
        f"  (Requires >{min_effect_size}% difference and minimum sample size of {min_sample_size})\n\n"
    )
    footnote_text += weighted_avg_info

    # Create a separate axis for the footnote
    footnote_ax = plt.subplot(gs[1])
    footnote_ax.axis("off")  # Hide the axis
    footnote_ax.text(
        0.01,
        0.99,
        footnote_text,
        fontsize=footnote_size,
        color="black",
        verticalalignment="top",
        transform=footnote_ax.transAxes,
    )

    # Set title and labels for the main heatmap
    row_label = row_col.replace("_", " ")
    col_label = col_col.replace("_", " ")
    title = f"Heatmap of {row_label} by {col_label}"

    ax.set_title(title, fontsize=title_size, pad=20)
    plt.sca(ax)  # Set ax as the current axis
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Adjust the spacing between subplots to increase the gap between heatmap and annotations
    plt.subplots_adjust(hspace=0.2)  # Increased from 0.05 to 0.2

    # Return the original pivot table, weighted averages, and significance results
    return fig, pivot_df


def create_heatmap_packing_week(
    df,
    col_col,
    value_col,
    figsize=(16, 20),
    significance_level=0.05,
    correct_multiple_tests=True,
    min_effect_size=2.5,
):
    """
    Create a heatmap from the DataFrame with packing_week as rows, ordered chronologically.
    This function combines create_heatmap and display_time_ordered_heatmap functionality.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data
    col_col : str
        Column name to use for columns (entity)
    value_col : str
        Column name to use for values/counts
    figsize : tuple, optional
        Figure size (width, height)
    significance_level : float, optional
        Alpha level for statistical significance testing (default 0.05)
    correct_multiple_tests : bool, optional
        Whether to apply correction for multiple testing (default True)
    min_effect_size : float, optional
        Minimum percentage point difference required for significance testing (default 2.5)
    top_n : int
        Number of top entity categories to show individually. The rest will be grouped as "Other".

    Returns:
    --------
    tuple: (pivot_df, ordered_pivot, weighted_avg_series, significance_df)
        - pivot_df: The original pivot table with raw counts
        - ordered_pivot: The pivot table ordered chronologically by packing_week and limited to top_n columns
        - weighted_avg_series: Series containing weighted average percentages for each entity
        - significance_df: DataFrame indicating which cells are significantly above average
    """
    import numpy as np
    import pandas as pd
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.stats.multitest import multipletests
    import matplotlib.gridspec as gridspec

    plt.ioff()  # Turn off interactive mode

    # Format entity name for display
    entity_name = col_col.replace("_", " ").title()

    # Create title if not provided
    title = f"Heatmap of Packing Week by {entity_name}"

    # Start with create_heatmap functionality
    # Convert min_effect_size from percentage points to proportion
    min_effect_size_prop = min_effect_size / 100

    row_col = "packing_week"
    filtered_df = df.dropna(subset=[row_col, col_col])
    filtered_df = filtered_df[filtered_df[row_col] != "None"]
    filtered_df = filtered_df[filtered_df[col_col] != "None"]
    
    
    # Validate measure parameter
    if value_col not in ["container_number", "std_cartons", "income"]:
        raise ValueError(
            "measure must be either 'container_number', 'std_cartons', or 'income'"
        )

    # Create a pivot table
    if value_col == "container_number":
        pivot_df = filtered_df.pivot_table(
            index=row_col,
            columns=col_col,
            values=value_col,
            aggfunc="nunique",
            fill_value=0,
        )
    elif value_col == "std_cartons":  # measure == 'std_cartons'
        pivot_df = filtered_df.pivot_table(
            index=row_col,
            columns=col_col,
            values=value_col,
            aggfunc="sum",
            fill_value=0,
        )
    else:  # value_col == 'income'
        pivot_df = filtered_df.pivot_table(
            index=row_col,
            columns=col_col,
            values=value_col,
            aggfunc="sum",
            fill_value=0,
        )

    # Calculate weighted averages
    entity1_totals = pivot_df.sum(axis=1)
    entity1_percentages = pivot_df.div(entity1_totals, axis=0)

    weighted_avg_by_entity2 = {}
    for entity2 in pivot_df.columns:
        weighted_sum = (entity1_percentages[entity2] * entity1_totals).sum()
        weighted_avg = weighted_sum / entity1_totals.sum()
        weighted_avg_by_entity2[entity2] = weighted_avg

    weighted_avg_series = pd.Series(weighted_avg_by_entity2).sort_values(
        ascending=False
    )

    # Perform statistical significance testing
    # Collect p-values for all tests
    pvalues = []
    test_indices = []

    # Minimum sample size for reliable testing
    min_sample_size = 30

    for row_idx in pivot_df.index:
        for col_idx in pivot_df.columns:
            # Skip cells with zero counts or small sample sizes
            if (
                pivot_df.loc[row_idx, col_idx] == 0
                or entity1_totals[row_idx] < min_sample_size
            ):
                continue

            # Get counts for this cell
            count = pivot_df.loc[row_idx, col_idx]
            row_total = entity1_totals[row_idx]

            # Calculate observed proportion
            observed_prop = count / row_total

            # Get weighted average (expected proportion)
            expected_prop = weighted_avg_by_entity2[col_idx]

            # Handle NaN or Infinity values
            if pd.isna(observed_prop) or pd.isna(expected_prop) or np.isinf(observed_prop) or np.isinf(expected_prop):
                continue

            # Only test if observed is higher than expected
            if observed_prop <= expected_prop:
                continue

            # Calculate effect size - how substantial is the difference?
            effect_size = observed_prop - expected_prop

            # Skip very small effect sizes (meaningful difference threshold) or NaN values
            if pd.isna(effect_size) or effect_size < min_effect_size_prop:
                continue

            # Perform z-test for proportions
            # Standard error for the difference between two proportions
            se = np.sqrt(
                expected_prop
                * (1 - expected_prop)
                * (1 / row_total + 1 / entity1_totals.sum())
            )

            # Z-score
            z_score = (observed_prop - expected_prop) / se

            # P-value (one-tailed test)
            p_value = 1 - stats.norm.cdf(z_score)

            # Store p-value and indices for later correction
            pvalues.append(p_value)
            test_indices.append((row_idx, col_idx))

    # Create a DataFrame to store significance results
    significance_df = pd.DataFrame(
        index=pivot_df.index, columns=pivot_df.columns, data=False
    )

    # Apply multiple testing correction if requested and if we have any tests
    if pvalues and correct_multiple_tests:
        # Use Benjamini-Hochberg FDR correction for multiple testing
        reject, pvals_corrected, _, _ = multipletests(
            pvalues, alpha=significance_level, method="fdr_bh"
        )

        # Update significance based on corrected p-values
        for i, (row_idx, col_idx) in enumerate(test_indices):
            is_significant = reject[i]
            significance_df.loc[row_idx, col_idx] = is_significant
    else:
        # Use uncorrected p-values
        for i, (row_idx, col_idx) in enumerate(test_indices):
            is_significant = pvalues[i] < significance_level
            significance_df.loc[row_idx, col_idx] = is_significant

    # Now implement the display_time_ordered_heatmap functionality
    # Sort the index (rows) by packing week chronologically
    ordered_pivot = pivot_df.copy().sort_index()

    # Get column totals to identify top entities
    col_totals = ordered_pivot.sum(axis=0)
    top_entities = col_totals.nlargest(15).index.tolist()

    # Create a new pivot with top N entities and "Other" category
    new_pivot = pd.DataFrame(index=ordered_pivot.index)

    # Add top entity columns
    for entity in top_entities:
        new_pivot[entity] = ordered_pivot[entity]

    # Create "Other" column if needed
    other_entities = [e for e in ordered_pivot.columns if e not in top_entities]
    if other_entities:
        new_pivot["Other"] = ordered_pivot[other_entities].sum(axis=1)

    # Replace our working pivot with the new one
    ordered_pivot = new_pivot

    # Calculate row totals for use in normalization and row counts
    new_entity1_totals = ordered_pivot.sum(axis=1)

    # Create normalized version for visualization
    viz_df = ordered_pivot.copy()
    row_sums = viz_df.sum(axis=1)
    viz_df = viz_df.div(row_sums, axis=0)

    # Calculate appropriate figure height based on number of rows (minimum height from figsize[1])
    adjusted_height = max(figsize[1], len(viz_df) * 0.5)
    adjusted_width = figsize[0]

    # Create the plot with improved layout
    fig = plt.figure(figsize=(adjusted_width, adjusted_height))

    # Define constants for layout
    footnote_size = 9
    title_size = 16

    # Use gridspec for more flexible layout
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.9, 0.1], figure=fig)
    ax = plt.subplot(gs[0])

    # Create the heatmap base
    heatmap = sns.heatmap(viz_df, cmap="YlGnBu", annot=False, linewidths=0.5, ax=ax)

    # Get the colormap normalized data for determining text color
    norm = plt.Normalize(viz_df.min().min(), viz_df.max().max())
    cmap_obj = plt.colormaps["YlGnBu"]

    # Create updated significance DataFrame
    # First, initialize with all False values matching our new pivot
    new_significance_df = pd.DataFrame(
        False, index=ordered_pivot.index, columns=ordered_pivot.columns
    )

    # Copy significance information for top entities
    for entity in top_entities:
        if entity in significance_df.columns:
            common_indices = set(new_significance_df.index) & set(significance_df.index)
            for idx in common_indices:
                new_significance_df.loc[idx, entity] = significance_df.loc[idx, entity]

    # "Other" category is marked as significant if any of its components were significant
    if "Other" in new_significance_df.columns and other_entities:
        for idx in new_significance_df.index:
            if idx in significance_df.index:
                # Check if any of the grouped entities were significant
                for entity in other_entities:
                    if (
                        entity in significance_df.columns
                        and significance_df.loc[idx, entity]
                    ):
                        new_significance_df.loc[idx, "Other"] = True
                        break

    # Use the updated significance DataFrame
    significance_df_top = new_significance_df

    # Add custom annotations with significance indicators
    for i, row_idx in enumerate(viz_df.index):
        for j, col_idx in enumerate(viz_df.columns):
            val = viz_df.loc[row_idx, col_idx]

            # Check if this cell should be marked as significant
            is_significant = False
            if (
                row_idx in significance_df_top.index
                and col_idx in significance_df_top.columns
            ):
                is_significant = significance_df_top.loc[row_idx, col_idx]

            # Get the actual color of the cell
            color_val = norm(val)
            rgb = cmap_obj(color_val)

            # Calculate luminance (brightness) of the background color
            luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

            # Choose text color based on background brightness
            text_color = "white" if luminance < 0.5 else "black"

            # Add the annotation with appropriate formatting
            if is_significant:
                text = ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{val:.2f}*",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color=text_color,
                )
            else:
                text = ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                )

    # Add row totals to the right of the heatmap
    right_pos = len(viz_df.columns)

    # Add a title for the totals column
    if value_col == "container_number":
        s = "Total Containers"
    elif value_col == "std_cartons":
        s = "Total Standard Cartons"
    else:
        s = "Total Revenue"

    ax.text(
        right_pos + 0.5,
        -0.5,
        s,
        ha="center",
        va="center",
        fontweight="bold",
        color="black",
    )

    # Add dividing line
    for i in range(-1, len(viz_df.index) + 1):
        ax.plot([right_pos, right_pos], [i, i + 1], "k-", lw=1)

    # Add row totals for each row
    for i, row_idx in enumerate(viz_df.index):
        row_total = new_entity1_totals[row_idx]
        ax.text(
            right_pos + 0.5,
            i + 0.5,
            f"{int(row_total):,}",
            ha="center",
            va="center",
            color="black",
        )

    # Create footnote text about statistical testing
    test_method = (
        "with FDR correction for multiple testing"
        if correct_multiple_tests
        else "without correction for multiple testing"
    )

    footnote_text = (
        f"* Statistically significant higher proportion at α={significance_level} {test_method}\n"
        f"  (Requires >{min_effect_size}% difference and minimum sample size of {min_sample_size})"
    )

    # Create a formatted string of weighted averages for top entities
    weighted_avg_info = "\n\nWeighted averages by column:\n"
    for col in viz_df.columns:
        if col in weighted_avg_by_entity2:
            weighted_avg_info += f"{col}: {weighted_avg_by_entity2[col]:.3f}  "
            # Add line breaks to avoid too long lines
            if (list(viz_df.columns).index(col) + 1) % 3 == 0:
                weighted_avg_info += "\n"

    footnote_text += weighted_avg_info

    # Create a separate axis for the footnote
    footnote_ax = plt.subplot(gs[1])
    footnote_ax.axis("off")  # Hide the axis
    footnote_ax.text(
        0.01,
        0.99,
        footnote_text,
        fontsize=footnote_size,
        color="black",
        verticalalignment="top",
        transform=footnote_ax.transAxes,
    )

    # Set title and labels
    ax.set_title(title, fontsize=title_size, pad=20)
    plt.sca(ax)  # Set ax as the current axis
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Adjust the spacing between subplots to avoid overlap
    plt.subplots_adjust(hspace=0.05)

    return fig, ordered_pivot


def calculate_concentration_metrics(
    df, entity1_col, entity2_col, value_col="container_number"
):
    """
    Calculate concentration metrics between two columns in a dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    entity1_col : str
        The column name for the first entity (grouping variable)
    entity2_col : str
        The column name for the second entity (whose concentration will be measured)
    value_col : str, default='container_number'
        The column to use for measurement (container_number, std_cartons, income)

    Returns:
    --------
    pandas.DataFrame
        A dataframe with the following columns for each entity1:
        - count of unique pallets
        - raw HHI (formatted to 2 decimal places)
        - normalized HHI (formatted to 2 decimal places)
        - count of distinct entity2 values
        - entity2 with the highest pallet count
        - percentage of pallets going to the largest entity2 (formatted to 2 decimal places)

        The dataframe is sorted by unique_pallets in descending order.
    """
    # Create a copy to avoid modifying the original dataframe
    data = df.dropna(subset=[entity1_col, entity2_col])
    data = data[data[entity1_col] != "None"]
    data = data[data[entity2_col] != "None"]

    # Determine the measurement method based on value_col
    if value_col == "container_number":
        # Step 1: Calculate entity preferences (distribution of entity2 for each entity1)
        entity_preferences = (
            data.groupby([entity1_col, entity2_col])["pallet_number"]
            .nunique()
            .reset_index(name="pallet_count")
        )

        # Calculate the total pallet count for each entity1
        entity1_totals = (
            data.groupby(entity1_col)["pallet_number"]
            .nunique()
            .reset_index(name="total_pallets")
        )

        measurement_name = "pallets"
    elif value_col == "std_cartons":
        # Use standard cartons instead of pallet counts
        entity_preferences = (
            data.groupby([entity1_col, entity2_col])["std_cartons"]
            .sum()
            .reset_index(name="pallet_count")  # keep column name for consistency
        )

        # Calculate the total cartons for each entity1
        entity1_totals = (
            data.groupby(entity1_col)["std_cartons"]
            .sum()
            .reset_index(name="total_pallets")  # keep column name for consistency
        )

        measurement_name = "cartons"
    else:  # value_col == "income"
        # Use revenue instead of pallet counts
        entity_preferences = (
            data.groupby([entity1_col, entity2_col])["income"]
            .sum()
            .reset_index(name="pallet_count")  # keep column name for consistency
        )

        # Calculate the total revenue for each entity1
        entity1_totals = (
            data.groupby(entity1_col)["income"]
            .sum()
            .reset_index(name="total_pallets")  # keep column name for consistency
        )

        measurement_name = "revenue"

    # Merge the total counts back to get percentages
    entity_preferences = entity_preferences.merge(entity1_totals, on=entity1_col)
    entity_preferences["percentage"] = (
        entity_preferences["pallet_count"] / entity_preferences["total_pallets"]
    ) * 100
    entity_preferences["percentage_prop"] = entity_preferences["percentage"] / 100

    # Step 2: Calculate metrics for each entity1
    results = []
    for name, group in entity_preferences.groupby(entity1_col):
        # Count unique pallets for this entity1
        total_pallets = group["total_pallets"].iloc[0]

        # Calculate raw HHI
        hhi = sum(group["percentage_prop"] ** 2)

        # Count distinct entity2 values
        distinct_entity2_count = len(group)

        # Calculate normalized HHI
        normalized_hhi = (
            1.0
            if distinct_entity2_count == 1
            else (hhi - 1 / distinct_entity2_count) / (1 - 1 / distinct_entity2_count)
        )

        # Find entity2 with the highest pallet count
        top_entity2_row = group.loc[group["pallet_count"].idxmax()]
        top_entity2 = top_entity2_row[entity2_col]
        top_entity2_pallet_count = top_entity2_row["pallet_count"]

        # Calculate percentage of total pallets for the top entity2
        top_entity2_percentage = (top_entity2_pallet_count / total_pallets) * 100

        results.append(
            {
                entity1_col: name,
                "unique_pallets": total_pallets,
                "hhi": round(hhi, 2),
                "normalized_hhi": round(normalized_hhi, 2),
                f"distinct_{entity2_col}_count": distinct_entity2_count,
                f"top_{entity2_col}": top_entity2,
                f"top_{entity2_col}_percentage": round(top_entity2_percentage, 2),
                "measurement": measurement_name,
            }
        )

    # Convert results to dataframe
    result_df = pd.DataFrame(results)

    # Sort by pallets in descending order
    result_df = result_df.sort_values(by="unique_pallets", ascending=False)

    return result_df


def plot_concentration_bubble(
    metrics_df, figsize=(12, 8), min_pallets=0, title=None, label_threshold=0.05
):
    """
    Create a simple bubble chart visualization from concentration metrics data.

    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        The dataframe output from calculate_concentration_metrics function
    figsize : tuple
        Figure size for the plot (width, height)
    min_pallets : int
        Minimum number of pallets for inclusion in the chart (0 to include all)
    title : str, optional
        Custom title for the plot
    label_threshold : float
        Minimum percentage of max pallet count to show labels (0.05 = 5%)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the visualization
    """

    df = metrics_df.copy()

    # Extract entity column name (first column)
    entity_col = df.columns[0]

    # Get top entity column name and percentage column
    top_entity_col = [
        col
        for col in df.columns
        if col.startswith("top_") and not col.endswith("percentage")
    ][0]
    top_pct_col = f"{top_entity_col}_percentage"

    # Extract entity2_col from top_entity_col by removing "top_" prefix
    entity2_col = top_entity_col.replace("top_", "")

    # Get measurement type from the metrics dataframe
    measurement = (
        df["measurement"].iloc[0] if "measurement" in df.columns else "pallets"
    )

    # Define axis label based on measurement
    y_axis_label = {
        "pallets": "Number of Pallets",
        "cartons": "Number of Standard Cartons",
        "revenue": "Total Revenue (ZAR)",
    }.get(measurement, "Value")

    # Filter data if threshold is provided
    plot_data = df[df["unique_pallets"] >= min_pallets]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Create color map from blue (low concentration) to red (high concentration)
    colors = ["#4575b4", "#91bfdb", "#e0f3f8", "#fee090", "#fc8d59", "#d73027"]
    cmap = LinearSegmentedColormap.from_list("concentration", colors)

    # Create bubble chart
    scatter = ax.scatter(
        plot_data["normalized_hhi"],
        plot_data["unique_pallets"],
        s=plot_data["unique_pallets"]
        / plot_data["unique_pallets"].max()
        * 500,  # Size based on pallets
        c=plot_data[top_pct_col],  # Color based on percentage of top entity
        cmap=cmap,
        alpha=0.7,
        edgecolors="k",
    )

    # Add labels to significant points
    for i, row in plot_data.iterrows():
        if row["unique_pallets"] > plot_data["unique_pallets"].max() * label_threshold:
            ax.annotate(
                row[entity_col],
                (row["normalized_hhi"], row["unique_pallets"]),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f"% of {measurement} to top {entity2_col}")

    # Set titles and labels
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(
            f"{entity2_col} Concentration Analysis by {entity_col}", fontsize=14
        )

    ax.set_xlabel(
        "Normalized HHI (0 = Even Distribution, 1 = Complete Concentration)",
        fontsize=12,
    )
    ax.set_ylabel(y_axis_label, fontsize=12)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save the figure
    # filename = f"{entity_col}_by_{entity2_col}_concentration_bubble.png"
    # plt.savefig(OUTPUT_CG_PATH / filename, dpi=300, bbox_inches="tight")

    return fig


def save_figures_to_pdf(figures, output_path, title):
    """
    Save multiple matplotlib figures to a single PDF file with proper layout preservation.

    Parameters:
    -----------
    figures : list
        List of matplotlib figure objects to save
    output_path : str or Path
        Path where the PDF will be saved
    title : str
        Title for the PDF metadata
    """
    with PdfPages(output_path) as pdf:
        for i, fig in enumerate(figures):
            # Do NOT call tight_layout again as it will override the careful
            # layout settings we've already applied in the figure creation functions

            # Save the figure to the PDF with proper bounding box to include all elements
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.5)

            # Close the figure to free memory
            plt.close(fig)

        # Add metadata to the PDF
        d = pdf.infodict()
        d["Title"] = title
        d["Author"] = "Data Science Team"
        d["Subject"] = "Market Concentration Analysis"
        d["Keywords"] = "concentration, market analysis, HHI"
        d["CreationDate"] = datetime.datetime.today()

    print(f"PDF saved to {output_path}")
