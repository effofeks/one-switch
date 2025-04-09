from pathlib import Path
from sqlalchemy import URL

DB_URL = URL.create(
    drivername="postgresql",
    username="warehouse",
    password="warehouse",
    host="localhost",
    port=8432,
    database="warehouse",
)
SCRIPTS_PATH = Path("scripts")
OUTPUT_PATH = Path("output")

CG_COL_TYPES = {
    "id": "Int64",
    "pallet_number": str,
    "container_number": str,
    "seller_id": "Int64",
    "buyer_id": "Int64",
    "packing_week": str,
    "production_region": str,
    "orchard": str,
    "commodity_name": str,
    "variety_name": str,
    "size_count": str,
    "size_categorization": str,
    "class": str,
    "pallet_stack": str,
    "pack": str,
    "local_market": str,
    "jbin": str,
    "target_market": str,
    "target_region": str,
    "target_country": str,
    "cartons": "Int64",
    "std_cartons": "Float64",
}

FIN_COL_TYPES = {
    "cg_id": "Int64",
    "document_type": str,
    "cost_type": str,
    "currency": str,
    "cgt_amount": "Float64",
    "exchange_rate": "Float64",
    "payment_status": str,
    "price_unit": str,
    "incoterm": str,
}

# Read SQL queries from files
with open(SCRIPTS_PATH / "carton_groupings.sql", "r") as f:
    cg_query = f.read()

with open(SCRIPTS_PATH / "dso_finance.sql", "r") as f:
    fin_query = f.read()
