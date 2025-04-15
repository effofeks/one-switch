import streamlit as st
import pandas as pd
import sqlalchemy

import utils

# Function to load data from database
@st.cache_data(show_spinner=False)
def load_data_from_database(season, commodity_group):
    """
    Load data from the database based on the specified season year and commodity group.

    Parameters:
    -----------
    season_year : int
        The season filter
    commodity_group : str
        The commodity group filter (e.g., 'Citrus')

    Returns:
    --------
    pandas.DataFrame
        A summarised dataframe of carton grouping data with finance data
    """
    try:
        # Create database engine
        engine = sqlalchemy.create_engine(utils.DB_URL)

        # Replace the filter parameters in the query
        cg_query = utils.cg_query.replace(
            "map_season_year = 2024", f"map_season_year = {season}"
        )
        cg_query = cg_query.replace(
            "commodities.commodity_group = 'Citrus'",
            f"commodities.commodity_group = '{commodity_group}'",
        )

        fin_query = utils.fin_query.replace(
            "carton_groupings.map_season_year = 2024",
            f"carton_groupings.map_season_year = {season}",
        )
        fin_query = fin_query.replace(
            "commodities.commodity_group = 'Citrus'",
            f"commodities.commodity_group = '{commodity_group}'",
        )

        # Load only the essential columns
        cg_df = pd.read_sql_query(sql=cg_query, con=engine, dtype=utils.CG_COL_TYPES)

        fin_df = pd.read_sql_query(
            sql=fin_query,
            con=engine,
            dtype=utils.FIN_COL_TYPES,
        )

        # Process finance data
        fin_df["cgt_amount_zar"] = fin_df["cgt_amount"] * fin_df["exchange_rate"]
        revenue_df = fin_df[
            (fin_df["payment_status"] != "voided")
            & (fin_df["document_type"] != "commercial_invoice")
        ]

        # Aggregate income by cg_id
        income_df = (
            revenue_df[revenue_df["cost_type"] == "income"]
            .groupby("cg_id")["cgt_amount_zar"]
            .sum()
            .reset_index()
        )
        income_df.rename(columns={"cgt_amount_zar": "income"}, inplace=True)

        # Merge income into main dataframe
        cg_df = cg_df.merge(income_df, left_on="id", right_on="cg_id", how="left")

        # Explicitly convert income to float
        cg_df["income"] = pd.to_numeric(cg_df["income"], errors="coerce")

        # Fill NA values
        cg_df = cg_df.fillna({"std_cartons": 0})

        # Remove self-loops (where buyer_id equals seller_id)
        cg_df = cg_df[cg_df["buyer_id"] != cg_df["seller_id"]]

        df = (
            cg_df.groupby(
                [
                    "seller_id",
                    "buyer_id",
                    "packing_week",
                    "container_number",
                    "commodity_name",
                    "variety_name",
                    "production_region",
                    "local_market",
                    "target_country",
                    "jbin",
                    "size_count",
                    "size_categorization",
                    "class",
                ]
            )
            .agg(
                {
                    "std_cartons": "sum",
                    "income": "sum",
                    "id": "count",  # Count the number of rows per group
                }
            )
            .reset_index()
        )

        # Rename the count column to be more descriptive
        df.rename(columns={"id": "n_cg"}, inplace=True)

        # Clean up memory
        del fin_df, revenue_df, income_df, cg_df
        import gc

        gc.collect()

        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


@st.cache_data(show_spinner=False)
def create_viz_df(df, viz_type="network"):
    """
    Create an aggregated dataframe based on the visualisation type.

    Parameters:
    -----------
    df : pandas.DataFrame
        The carton groupings dataframe
    viz_type : str
        The visualisation type to determine how to aggregate the data

    Returns:
    --------
    pandas.DataFrame
        The aggregated dataframe suitable for the specified visualisation
    """
    if df is None:
        return None

    if viz_type == "network":
        # Network visualisation aggregation
        viz_df = (
            df.groupby(["seller_id", "buyer_id", "packing_week", "container_number"])
            .agg(
                {
                    "std_cartons": "sum",
                    "income": "sum",
                    "n_cg": "sum",
                }
            )
            .reset_index()
        )

        return viz_df

    elif viz_type == "heatmap":
        # Group by grower decision variables and pallet spec
        viz_df = (
            df.groupby(
                [
                    "seller_id",
                    "buyer_id",
                    "packing_week",
                    "container_number",
                    "commodity_name",
                    "variety_name",
                    "production_region",
                    "local_market",
                    "target_country",
                    "jbin",
                    "size_count",
                    "size_categorization",
                    "class",
                ]
            )
            .agg(
                {
                    "std_cartons": "sum",
                    "income": "sum",
                    "n_cg": "sum",  # Count the number of rows per group
                }
            )
            .reset_index()
        )

        return viz_df

    else:
        st.warning(
            f"Visualization type '{viz_type}' not recognized. Using default heatmap visualisation."
        )
        return create_viz_df(df, "heatmap")


def get_unique_company_ids(df):
    """Extract unique company IDs from the network dataframe."""
    if df is None:
        return []

    buyer_ids = df["buyer_id"].dropna().unique().tolist()
    seller_ids = df["seller_id"].dropna().unique().tolist()
    return sorted(list(set(buyer_ids + seller_ids)))
