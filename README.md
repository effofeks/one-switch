# Network Visualization Tool

A lightweight web application for visualizing network graphs using different visualization methods based on parameter inputs.

## Overview

This application provides a user-friendly interface to generate various network visualizations:

1. **Strong Connections** - Visualize nodes with strong edge connections and their neighbors
2. **Company Network** - Visualize a network centered around a specific company
3. **Network Timelapse** - Create an animation showing the network evolution over time

## Installation

1. Make sure you have Python 3.8+ installed
2. Install the required packages:

```
pip install -r requirements.txt
```

3. Set up database connection (optional):
   * Create a `.env` file in the root directory with your database credentials
   * Use the following format:
   ```
   DBT_DEV_HOST=your_host
   DBT_DEV_USER=your_username
   DBT_DEV_PASSWORD=your_password
   ```
   * Alternatively, edit the DB_URL in cg_analysis.py directly

## Running the Application

To start the application, run:

```
streamlit run network_viz_app.py
```

This will start a local web server and open the application in your default web browser.

## Features

- Interactive parameter selection through sliders and dropdowns
- Three different network visualization types
- Dynamic graph generation based on parameter values
- Animated network visualization (timelapse)
- Data source selection (sample data or database connection)
- Data filtering options for season year and commodity group
- Detailed visualization of trade networks between buyers and sellers

## Usage

### Using Sample Data
1. Select "Sample Data" as the data source
2. Select the type of visualization from the sidebar
3. Adjust the parameters for the selected visualization
4. Click the "Generate" button to create the visualization

### Using Database Connection
1. Select "Database Connection" as the data source
2. Choose the season year and commodity group from the dropdown menus
3. Click "Load Data" to fetch the filtered data from the database
4. Select the visualization type and adjust parameters
5. Click the "Generate" button to create the visualization

## Data Processing

When using the database connection, the application:
1. Loads data from three main SQL queries:
   - Carton groupings data (main trade transactions)
   - Pallet timeline data (for standardized carton measurements)
   - Finance data (for revenue and expense information)
2. Applies filters for season year and commodity group
3. Processes and merges the data to create a comprehensive dataset for visualization
4. Calculates derived metrics like revenue in a standardized currency

## Future Work

- Add more visualization types and parameters
- Add export options for generated visualizations
- Implement caching for improved performance with large datasets
- Add user authentication for secure database access
- Add more filtering options (by region, date range, etc.)
- Support for custom SQL queries

## Dependencies

- Streamlit
- NetworkX
- Matplotlib
- Pandas
- NumPy
- SQLAlchemy
- python-dotenv 