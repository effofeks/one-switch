# Data Analysis & visualisation Tool

A comprehensive web application for analyzing and visualizing trade data through network graphs and statistical heatmaps.

## Overview

This application provides a user-friendly interface to generate various data visualisations:

1. **Network visualisations**
   - **Buyer-Seller Relationship** - Visualize nodes with strong edge connections and their neighbors
   - **Individual Company Network** - Visualize a network centered around a specific company
   - **Network Timelapse** - Create an animation showing the network evolution over time

2. **Statistical Heatmaps**
   - **Standard Heatmap** - Visualize relationships between two categorical variables with statistical significance testing
   - **Packing Week Heatmap** - Time-ordered heatmap showing trends across packing weeks

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
   DB_USERNAME = 'warehouse'
   DB_PASSWORD = 'warehouse'
   DB_HOST = 'localhost'
   DB_PORT = '5432'
   DB_NAME = 'warehouse'
   ```

## Running the Application

To start the application, run:

```
streamlit run data_explorer_app.py
```

This will start a local web server and open the application in your default web browser.

## Features

- Interactive parameter selection through sliders and dropdowns
- Multiple visualisation types for different analytical needs
- Statistical significance testing in heatmaps with p-value tables
- Dynamic graph generation based on parameter values
- Animated network visualisation (timelapse)
- Data source selection (sample data or database connection)
- Data filtering options for season year and commodity group
- Detailed visualisation of trade networks between buyers and sellers

## Usage

### Data Loading and Filtering
1. Select your data source
2. Apply filters as needed
3. Click "Load Data" to prepare the dataset for visualisation

### Creating visualisations
1. Select the visualisation type from the sidebar
2. Configure the parameters specific to that visualisation
3. Click "Generate" to create the visualisation
4. View statistics and insights in the expandable "Statistics" section

### Heatmap Statistical Analysis
The heatmap visualisations include:
- Statistical significance testing with configurable parameters
- Correction for multiple testing using Benjamini-Hochberg FDR method
- Adjustable significance level and minimum effect size
- Complete p-value table with formatted values and counts

## Data Processing

The application processes data through:
1. Loading from selected data sources
2. Filtering based on user-selected criteria
3. Applying statistical methods for significance testing
4. Calculating weighted averages and normalized proportions for heatmaps
5. Generating network metrics for graph-based visualisations
