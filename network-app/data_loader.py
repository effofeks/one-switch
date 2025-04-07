import streamlit as st
import pandas as pd
import sqlalchemy
from sqlalchemy import URL
from pathlib import Path
import os

# Import column types and constants from network_analysis
from network_analysis import (
    CG_COL_TYPES,
    FIN_COL_TYPES,
    DB_URL,
    SCRIPTS_PATH,
)

# Function to read SQL query from file
def read_sql_file(file_path):
    """Read SQL query from a file."""
    with open(file_path, "r") as f:
        return f.read()


# Function to load data from database
@st.cache_data(show_spinner=False)
def load_data_from_database(season_year, commodity_group):
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
        The carton groupings dataframe with essential data
    """
    try:
        # Create database engine
        engine = sqlalchemy.create_engine(DB_URL)

        # Read the SQL query file
        cg_query = read_sql_file(SCRIPTS_PATH / "carton_groupings.sql")
        fin_query = read_sql_file(SCRIPTS_PATH / "dso_finance.sql")

        # Replace the filter parameters in the query
        cg_query = cg_query.replace(
            "map_season_year = 2024", f"map_season_year = {season_year}"
        )
        cg_query = cg_query.replace(
            "commodities.commodity_group = 'Citrus'",
            f"commodities.commodity_group = '{commodity_group}'",
        )

        fin_query = fin_query.replace(
            "carton_groupings.map_season_year = 2024",
            f"carton_groupings.map_season_year = {season_year}",
        )
        fin_query = fin_query.replace(
            "commodities.commodity_group = 'Citrus'",
            f"commodities.commodity_group = '{commodity_group}'",
        )

        # Load only the essential columns
        cg_df = pd.read_sql_query(sql=cg_query, con=engine, dtype=CG_COL_TYPES)

        fin_df = pd.read_sql_query(
            sql=fin_query,
            con=engine,
            dtype=FIN_COL_TYPES,
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

        # Fill NA values
        cg_df = cg_df.fillna({"income": 0, "std_cartons": 0})

        # Remove self-loops (where buyer_id equals seller_id)
        cg_df = cg_df[cg_df["buyer_id"] != cg_df["seller_id"]]

        # Clean up memory
        del fin_df, revenue_df, income_df
        import gc
        gc.collect()

        return cg_df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


@st.cache_data(show_spinner=False)
def create_aggregated_dataframe(cg_df, viz_type="network"):
    """
    Create an aggregated dataframe based on the visualization type.
    
    Parameters:
    -----------
    cg_df : pandas.DataFrame
        The carton groupings dataframe
    viz_type : str
        The visualization type to determine how to aggregate the data
        
    Returns:
    --------
    pandas.DataFrame
        The aggregated dataframe suitable for the specified visualization
    """
    if cg_df is None:
        return None
    
    if viz_type == "network":
        # Network visualization aggregation (original logic)
        # Group by seller_id, buyer_id, packing_week, container_number
        aggregated_df = (
            cg_df.groupby(["seller_id", "buyer_id", "packing_week", "container_number"])
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
        aggregated_df.rename(columns={"id": "n_cg"}, inplace=True)
        
        return aggregated_df
    
    elif viz_type == "heatmap":
        # Group by seller_id, buyer_id, packing_week, container_number
        aggregated_df = (
            cg_df.groupby(["seller_id", "buyer_id", "packing_week", "container_number", "commodity_name", "variety_name", "production_region", "local_market", "target_country", "jbin", "size_count", "size_categorization"])
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
        aggregated_df.rename(columns={"id": "n_cg"}, inplace=True)
        
        return aggregated_df
    
    else:
        st.warning(f"Visualization type '{viz_type}' not recognized. Using default network visualization.")
        return create_aggregated_dataframe(cg_df, "network")


def get_unique_company_ids(net_df):
    """Extract unique company IDs from the network dataframe."""
    if net_df is None:
        return []
        
    buyer_ids = net_df["buyer_id"].dropna().unique().tolist()
    seller_ids = net_df["seller_id"].dropna().unique().tolist()
    return sorted(list(set(buyer_ids + seller_ids))) 