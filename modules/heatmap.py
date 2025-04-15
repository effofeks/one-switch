import datetime
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap


def create_heatmap(
    df,
    row_col,
    col_col,
    value_col,
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
    significance_level : float, optional
        Alpha level for statistical significance testing (default 0.05)
    correct_multiple_tests : bool, optional
        Whether to apply correction for multiple testing (default True)
    min_effect_size : float, optional
        Minimum percentage point difference required for significance testing (default 2.5)

    Returns:
    --------
    tuple: (fig, pivot_df, pvalue_df)
        - fig: The heatmap figure
        - pivot_df: The original pivot table with raw counts
        - pvalue_df: DataFrame containing p-values and cell values for all row/column combinations
    """
    import numpy as np
    import pandas as pd
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.stats.multitest import multipletests

    # Validate input parameters
    if df is None or len(df) == 0:
        raise ValueError("Input dataframe is empty or None")
        
    if row_col not in df.columns:
        raise ValueError(f"Row column '{row_col}' not found in dataframe")
        
    if col_col not in df.columns:
        raise ValueError(f"Column column '{col_col}' not found in dataframe")
        
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in dataframe")
    
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    # initial_rows = len(df_copy)
    # print(f"DEBUG: Initial row count: {initial_rows}")
    # print(f"DEBUG: Sample data for {row_col}: {df_copy[row_col].head().tolist()}")
    # print(f"DEBUG: Sample data for {col_col}: {df_copy[col_col].head().tolist()}")
    # print(f"DEBUG: Sample data for {value_col}: {df_copy[value_col].head().tolist()}")

    # Convert min_effect_size from percentage points to proportion
    min_effect_size_prop = min_effect_size / 100

    filtered_df = df_copy.dropna(subset=[row_col, col_col])
    # print(f"DEBUG: Row count after dropping NA in row/col: {len(filtered_df)}")
    
    filtered_df = filtered_df[filtered_df[row_col] != "None"]
    filtered_df = filtered_df[filtered_df[col_col] != "None"]
    # print(f"DEBUG: Row count after removing 'None' values: {len(filtered_df)}")

    # Convert row and column values to strings for safety
    filtered_df[row_col] = filtered_df[row_col].astype(str)
    filtered_df[col_col] = filtered_df[col_col].astype(str)
    
    # Print unique values to help diagnose empty pivot issues
    # print(f"DEBUG: Unique values in {row_col}: {filtered_df[row_col].nunique()}")
    # print(f"DEBUG: Unique values in {col_col}: {filtered_df[col_col].nunique()}")
    
    # Relaxed validation for value_col - don't restrict to just three types
    if value_col not in ["container_number", "std_cartons", "income"]:
        raise ValueError(
            "measure must be either 'containers', 'std_cartons', or 'revenue'"
        )

    if value_col == "income":
        filtered_df = filtered_df.dropna(subset=["income"])
        # print(f"DEBUG: Row count after dropping NA in income: {len(filtered_df)}")

    # Create a pivot table
    try:
        if value_col == "container_number":
            try:
                pivot_df = filtered_df.pivot_table(
                    index=row_col,
                    columns=col_col,
                    values=value_col,
                    aggfunc="nunique",
                    fill_value=0,
                )
            except Exception as e:
                print(f"DEBUG: Error with nunique: {str(e)}, falling back to count")
                # Fall back to count if nunique fails
                pivot_df = filtered_df.pivot_table(
                    index=row_col,
                    columns=col_col,
                    values=value_col,
                    aggfunc="count",
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
        else:  # value_col == 'income' or any other numeric
            pivot_df = filtered_df.pivot_table(
                index=row_col,
                columns=col_col,
                values=value_col,
                aggfunc="sum",
                fill_value=0,
            )
    except Exception as e:
        print(f"DEBUG: Pivot table creation failed: {str(e)}")
        # Last resort: try a very basic pivot using count
        try:
            print("DEBUG: Attempting fallback to simple count pivot")
            pivot_df = pd.pivot_table(
                filtered_df,
                index=row_col,
                columns=col_col,
                values=value_col,
                aggfunc="count",
                fill_value=0
            )
        except Exception as e2:
            print(f"DEBUG: Fallback pivot also failed: {str(e2)}")
            raise ValueError(f"Could not create pivot table: {str(e)}")
    
    # print(f"DEBUG: Pivot table shape: {pivot_df.shape if not pivot_df.empty else 'Empty'}")
    # if not pivot_df.empty:
    #     print(f"DEBUG: Pivot table index preview: {pivot_df.index[:5].tolist()}")
    #     print(f"DEBUG: Pivot table columns preview: {pivot_df.columns[:5].tolist()}")
        
    # Check if pivot_df is empty
    # if pivot_df.empty:
    #     print(f"DEBUG: Empty pivot table. Last filtered dataframe had {len(filtered_df)} rows.")
    #     print(f"DEBUG: Last filtered dataframe sample: {filtered_df.head(3).to_dict()}")


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

    # Collect p-values for all tests
    pvalues = []
    test_indices = []
    
    # Create p-value table - one table for all combinations of row and column
    # with p-values and original counts
    pvalue_table = pd.DataFrame(index=pivot_df.index, columns=pivot_df.columns)
    count_table = pivot_df.copy()

    # Perform statistical significance testing with improved approach
    min_sample_size = 30  # Minimum sample size for reliable testing
    
    # print(f"DEBUG: Check point 1")

    for row_idx in pivot_df.index:
        for col_idx in pivot_df.columns:
            # print(f"DEBUG: row_idx {row_idx} and col_idx {col_idx}")
            # Skip cells with zero counts or small sample sizes
            count = pivot_df.loc[row_idx, col_idx]
            row_total = entity1_totals[row_idx]
            
            # Default p-value is 1.0 (not significant)
            p_value = 1.0
            
            if count == 0 or row_total < min_sample_size:
                pvalue_table.loc[row_idx, col_idx] = f"{p_value:.4f} ({int(count)})"
                continue

            # Calculate observed proportion
            observed_prop = count / row_total
            # print(f"DEBUG: observed_prop {observed_prop}")
            # Get weighted average (expected proportion)
            expected_prop = weighted_avg_by_entity2[col_idx]
            # print(f"DEBUG: expected_prop {expected_prop}")

            # Handle NaN or Infinity values
            if (
                pd.isna(observed_prop)
                or pd.isna(expected_prop)
                or np.isinf(observed_prop)
                or np.isinf(expected_prop)
            ):
                pvalue_table.loc[row_idx, col_idx] = f"{p_value:.4f} ({int(count)})"
                continue

            # Only test if observed is higher than expected
            if observed_prop <= expected_prop:
                pvalue_table.loc[row_idx, col_idx] = f"{p_value:.4f} ({int(count)})"
                continue

            # Calculate effect size - how substantial is the difference?
            effect_size = observed_prop - expected_prop

            # Skip very small effect sizes (meaningful difference threshold) or NaN values
            if pd.isna(effect_size) or effect_size < min_effect_size_prop:
                pvalue_table.loc[row_idx, col_idx] = f"{p_value:.4f} ({int(count)})"
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
            
            # Store the p-value in the table with the count
            pvalue_table.loc[row_idx, col_idx] = f"{p_value:.4f} ({int(count)})"

            # Store p-value and indices for later correction
            pvalues.append(p_value)
            test_indices.append((row_idx, col_idx))

    # Create a DataFrame to store significance results
    significance_df = pd.DataFrame(
        index=pivot_df.index, columns=pivot_df.columns, data=False
    )

    # print(f"DEBUG: Check point 2")
    # print(significance_df)

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
            
            # Update the p-value in the table with the corrected p-value
            count = int(pivot_df.loc[row_idx, col_idx])
            pvalue_table.loc[row_idx, col_idx] = f"{pvals_corrected[i]:.4f} ({count})"
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

    # Update our working DataFrames for visualisation
    pivot_df_top = working_pivot
    significance_df_top = working_significance_df

    # Recalculate entity1_totals for the new pivot
    new_entity1_totals = pivot_df_top.sum(axis=1)

    # Normalise rows
    viz_df = pivot_df_top.copy()
    row_sums = viz_df.sum(axis=1)
    viz_df = viz_df.div(row_sums, axis=0)
    # Ensure all values are float64 (not Float64/nullable type)
    viz_df = viz_df.astype('float64')
    fmt = ".2f"

    figsize=(16, 10)
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

    # Return the figure, DataFrame, and p-value table
    return fig, pivot_df, pvalue_table


def create_heatmap_packing_week(
    df,
    col_col,
    value_col,
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
    significance_level : float, optional
        Alpha level for statistical significance testing (default 0.05)
    correct_multiple_tests : bool, optional
        Whether to apply correction for multiple testing (default True)
    min_effect_size : float, optional
        Minimum percentage point difference required for significance testing (default 2.5)

    Returns:
    --------
    tuple: (fig, pivot_df, pvalue_df)
        - fig: The heatmap figure
        - pivot_df: The original pivot table with raw counts
        - pvalue_df: DataFrame containing p-values and cell values for all row/column combinations
    """
    import numpy as np
    import pandas as pd
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.stats.multitest import multipletests
    import matplotlib.gridspec as gridspec

    row_col = 'packing_week'
    # Validate input parameters
    if df is None or len(df) == 0:
        raise ValueError("Input dataframe is empty or None")
        
    if row_col not in df.columns:
        raise ValueError(f"Row column '{row_col}' not found in dataframe")
        
    if col_col not in df.columns:
        raise ValueError(f"Column column '{col_col}' not found in dataframe")
        
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in dataframe")
    
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()

    # Start with create_heatmap functionality
    # Convert min_effect_size from percentage points to proportion
    min_effect_size_prop = min_effect_size / 100

    row_col = "packing_week"
    filtered_df = df_copy.dropna(subset=[row_col, col_col])
    filtered_df = filtered_df[filtered_df[row_col] != "None"]
    filtered_df = filtered_df[filtered_df[col_col] != "None"]
        
    # Convert row and column values to strings for safety
    filtered_df[row_col] = filtered_df[row_col].astype(str)
    filtered_df[col_col] = filtered_df[col_col].astype(str)

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
        
    # Check if pivot_df is empty
    if pivot_df.empty:
        raise ValueError("Pivot table is empty. Check your input data and column selections.")

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

    # Create p-value table - one table for all combinations of row and column
    # with p-values and original counts
    pvalue_table = pd.DataFrame(index=pivot_df.index, columns=pivot_df.columns)

    # Perform statistical significance testing
    # Collect p-values for all tests
    pvalues = []
    test_indices = []

    # Minimum sample size for reliable testing
    min_sample_size = 30

    for row_idx in pivot_df.index:
        for col_idx in pivot_df.columns:
            # Get counts for this cell
            count = pivot_df.loc[row_idx, col_idx]
            row_total = entity1_totals[row_idx]
            
            # Default p-value is 1.0 (not significant)
            p_value = 1.0
            
            # Skip cells with zero counts or small sample sizes
            if count == 0 or row_total < min_sample_size:
                pvalue_table.loc[row_idx, col_idx] = f"{p_value:.4f} ({int(count)})"
                continue

            # Calculate observed proportion
            observed_prop = count / row_total

            # Get weighted average (expected proportion)
            expected_prop = weighted_avg_by_entity2[col_idx]

            # Handle NaN or Infinity values
            if (
                pd.isna(observed_prop)
                or pd.isna(expected_prop)
                or np.isinf(observed_prop)
                or np.isinf(expected_prop)
            ):
                pvalue_table.loc[row_idx, col_idx] = f"{p_value:.4f} ({int(count)})"
                continue

            # Only test if observed is higher than expected
            if observed_prop <= expected_prop:
                pvalue_table.loc[row_idx, col_idx] = f"{p_value:.4f} ({int(count)})"
                continue

            # Calculate effect size - how substantial is the difference?
            effect_size = observed_prop - expected_prop

            # Skip very small effect sizes (meaningful difference threshold) or NaN values
            if pd.isna(effect_size) or effect_size < min_effect_size_prop:
                pvalue_table.loc[row_idx, col_idx] = f"{p_value:.4f} ({int(count)})"
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
            
            # Store the p-value in the table with the count
            pvalue_table.loc[row_idx, col_idx] = f"{p_value:.4f} ({int(count)})"

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
            
            # Update the p-value in the table with the corrected p-value
            count = int(pivot_df.loc[row_idx, col_idx])
            pvalue_table.loc[row_idx, col_idx] = f"{pvals_corrected[i]:.4f} ({count})"
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

    # Create normalized version for visualisation
    viz_df = ordered_pivot.copy()
    row_sums = viz_df.sum(axis=1)
    viz_df = viz_df.div(row_sums, axis=0)
    # Ensure all values are float64 (not Float64/nullable type)
    viz_df = viz_df.astype('float64')

    # Calculate appropriate figure height based on number of rows (minimum height from figsize[1])
    figsize=(16, 20)
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

    # Add a footnote with statistical test information
    test_method = (
        "with FDR correction for multiple tests"
        if correct_multiple_tests
        else "without correction for multiple tests"
    )
    footnote_text = (
        f"* Statistically significant higher proportion at α={significance_level} {test_method}\n"
        f"  (Requires >{min_effect_size}% difference and minimum sample size of {min_sample_size})\n\n"
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
        1.0,
        footnote_text,
        fontsize=footnote_size,
        transform=footnote_ax.transAxes,
        verticalalignment="top",
    )

    entity_name = col_col.replace("_", " ").title()
    title = f"Heatmap of Packing Week by {entity_name}"
    
    ax.set_title(title, fontsize=title_size, pad=20)
    plt.sca(ax)  # Set ax as the current axis
    plt.xticks(rotation=45, ha="right")
    
    # Return the figure, DataFrame, and p-value table
    return fig, pivot_df, pvalue_table
