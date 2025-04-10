import pandas as pd
import os
from pathlib import Path

def process_raw_data():
    """
    Load the Raw.csv file and drop all columns after 'price'.
    
    Returns:
    --------
    pandas.DataFrame
        The processed dataframe with all columns after 'price' dropped
    """
    # Define the file path
    file_path = Path('data/Estimate Volumes and Introductions.xlsx - Raw.csv')
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    # Load the CSV file
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    # Print original shape
    print(f"Original shape: {df.shape}")
    
    # Find the index of the 'price' column
    try:
        price_index = df.columns.get_loc('price')
        
        # Get all columns after 'price'
        columns_to_drop = df.columns[price_index+1:].tolist()
        print(f"Dropping columns: {columns_to_drop}")
        
        # Drop all columns after 'price'
        df_processed = df.drop(columns=columns_to_drop)
        
        # Print new shape
        print(f"New shape: {df_processed.shape}")
        
        return df_processed
    except KeyError:
        print("Error: 'price' column not found in the dataframe.")
        print(f"Available columns: {', '.join(df.columns)}")
        raise

def process_introductions_data():
    """
    Load the Introductions CSV file with the following specifications:
    - Read from row 2 to row 18
    - Row 2 contains header names
    - First column should have column name 'grower'
    
    Returns:
    --------
    pandas.DataFrame
        The processed dataframe with rows 2 to 18 and first column named 'grower'
    """
    # Define the file path
    file_path = Path('data/Estimate Volumes and Introductions.xlsx - Introductions.csv')
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    # Load the CSV file
    print(f"Loading {file_path}...")
    
    # Read the entire file first to get all rows
    df = pd.read_csv(file_path, header=None)
    
    # Print original shape
    print(f"Original shape: {df.shape}")
    
    # Get the header row (row 2, which is index 1)
    header = df.iloc[1]
    
    # Get the data rows (rows 3-18, which are indices 2-17)
    data = df.iloc[2:18]
    
    # Create the dataframe with the header row and data rows
    df_processed = pd.DataFrame(data.values, columns=header)
    
    # Rename the first column to 'grower'
    if len(df_processed.columns) > 0:
        first_col = df_processed.columns[0]
        df_processed = df_processed.rename(columns={first_col: 'grower'})
        print(f"Renamed first column '{first_col}' to 'grower'")
    
    # Print new shape
    print(f"New shape: {df_processed.shape}")
    
    return df_processed

def process_estimate_volumes():
    """
    Load the Estimate Volumes CSV file and create two separate dataframes:
    1. A citrus dataframe from row 3 to row 20 (row 3 includes header names)
    2. A pome dataframe from row 24 to row 27 (row 24 has header names)
    
    For both dataframes:
    - Drop columns named 'Total' in citrus and 'Total volumes' in pome
    - Unpivot columns from index 4 onwards into a 'packing_week_number' column
    
    Returns:
    --------
    tuple
        (citrus_df, pome_df) - Two dataframes containing the processed data
    """
    # Define the file path
    file_path = Path('data/Estimate Volumes and Introductions.xlsx - Estimate Volumes.csv')
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    # Load the CSV file
    print(f"Loading {file_path}...")
    
    # Read the entire file first to get all rows
    df = pd.read_csv(file_path, header=None)
    
    # Print original shape
    print(f"Original shape: {df.shape}")
    
    # Create citrus dataframe (rows 3-20, with row 3 as header)
    # In pandas, row 3 is index 2, and row 20 is index 19
    citrus_header = df.iloc[2]  # Get the header row (row 3, index 2)
    citrus_data = df.iloc[2:20]  # Get rows 3-20 (index 2-19)
    citrus_df = pd.DataFrame(citrus_data.values, columns=citrus_header)
    
    # Drop columns named 'Total' in citrus dataframe
    total_cols = [col for col in citrus_df.columns if 'Total' in str(col)]
    if total_cols:
        citrus_df = citrus_df.drop(columns=total_cols)
        print(f"Dropped columns from citrus dataframe: {total_cols}")
    
    # Create pome dataframe (rows 24-27, with row 24 as header)
    # In pandas, row 24 is index 23, and row 27 is index 26
    pome_header = df.iloc[23]  # Get the header row (row 24, index 23)
    pome_data = df.iloc[23:27]  # Get rows 24-27 (index 23-26)
    pome_df = pd.DataFrame(pome_data.values, columns=pome_header)
    
    # Drop columns named 'Total volumes' in pome dataframe
    total_volumes_cols = [col for col in pome_df.columns if 'Total volumes' in str(col)]
    if total_volumes_cols:
        pome_df = pome_df.drop(columns=total_volumes_cols)
        print(f"Dropped columns from pome dataframe: {total_volumes_cols}")
    
    # Function to unpivot dataframe
    def unpivot_dataframe(df, start_col_index=4):
        """
        Unpivot columns from start_col_index onwards into a 'packing_week_number' column.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe to unpivot
        start_col_index : int
            The index of the first column to unpivot (0-based)
            
        Returns:
        --------
        pandas.DataFrame
            The unpivoted dataframe
        """
        # Get the columns to keep as identifiers
        id_cols = df.columns[:start_col_index].tolist()
        
        # Get the columns to unpivot
        value_cols = df.columns[start_col_index:].tolist()
        
        # Unpivot the dataframe
        df_unpivoted = pd.melt(
            df,
            id_vars=id_cols,
            value_vars=value_cols,
            var_name='packing_week_number',
            value_name='volume'
        )
        
        return df_unpivoted
    
    # Unpivot both dataframes
    citrus_df_unpivoted = unpivot_dataframe(citrus_df)
    pome_df_unpivoted = unpivot_dataframe(pome_df)
    
    # Print shapes of the new dataframes
    print(f"\nCitrus dataframe shape: {citrus_df_unpivoted.shape}")
    print(f"Pome dataframe shape: {pome_df_unpivoted.shape}")
    
    return citrus_df_unpivoted, pome_df_unpivoted

def process_demand_volumes():
    """
    Load the Demand Volumes CSV file and create separate dataframes for different fruit categories:
    - Apples: row 3 to row 5, with header names
    - Lemons: row 10 to row 13, with header names
    - Oranges: row 18 to row 21, with header names (row 17 is variety, row 18 is packing week)
    - Grapefruit: row 24 to row 28, with header names
    - Soft citrus: row 32 to row 35, with header names
    
    For all dataframes:
    - First column is renamed to 'receiver'
    - Second column (index 1) is dropped
    - For dataframes with compound column structure (like lemons and oranges), create a MultiIndex
    - Numeric columns are unpivoted into a 'variety' column
    
    Returns:
    --------
    dict
        Dictionary containing the processed dataframes for each fruit category
    """
    # Define the file path
    file_path = Path('data/Estimate Volumes and Introductions.xlsx - Demand Volumes.csv')
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    # Load the CSV file
    print(f"Loading {file_path}...")
    
    # Read the entire file first to get all rows
    df = pd.read_csv(file_path, header=None)
    
    # Print original shape
    print(f"Original shape: {df.shape}")
    
    # Dictionary to store all dataframes
    fruit_dfs = {}
    
    # Define the fruit categories and their row ranges
    fruit_ranges = {
        'apples': {'header_row': 2, 'data_start': 2, 'data_end': 4},  # rows 3-5
        'lemons': {'header_row': 9, 'data_start': 9, 'data_end': 12},  # rows 10-13
        'oranges': {'header_row': 16, 'data_start': 17, 'data_end': 20},  # rows 18-21 (row 17 is variety, row 18 is packing week)
        'grapefruit': {'header_row': 23, 'data_start': 23, 'data_end': 27},  # rows 24-28
        'soft_citrus': {'header_row': 31, 'data_start': 31, 'data_end': 34}  # rows 32-35
    }
    
    # Function to unpivot dataframe
    def unpivot_dataframe(df, start_col_index=1):
        """
        Unpivot columns from start_col_index onwards into a 'variety' column.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe to unpivot
        start_col_index : int
            The index of the first column to unpivot (0-based)
            
        Returns:
        --------
        pandas.DataFrame
            The unpivoted dataframe
        """
        # Get the columns to keep as identifiers
        id_cols = df.columns[:start_col_index].tolist()
        
        # Get the columns to unpivot
        value_cols = df.columns[start_col_index:].tolist()
        
        # Unpivot the dataframe
        df_unpivoted = pd.melt(
            df,
            id_vars=id_cols,
            value_vars=value_cols,
            var_name='variety',
            value_name='volume'
        )
        
        return df_unpivoted
    
    # Create dataframes for each fruit category
    for fruit, ranges in fruit_ranges.items():
        print(f"\nProcessing {fruit} data...")
        
        # Special handling for oranges data
        if fruit in ['oranges', 'lemons', 'grapefruit', 'soft_citrus']:
            # Get the header rows for oranges
            df.columns
            print(df.head())
            print(df.columns)
            variety_header = df.iloc[16, 2]  # Row 17 (index 16) is variety
            week_header = df.iloc[17, 2]     # Row 18 (index 17) is packing week
            print(variety_header)
            print(week_header)  
            # Drop the second column (index 1)
            if len(fruit_df.columns) > 1:
                second_col = fruit_df.columns[1]
                fruit_df = fruit_df.drop(columns=[second_col])
                print(f"Dropped second column '{second_col}' from {fruit} dataframe")
            
            # Get the data rows
            data = df.iloc[ranges['data_start']:ranges['data_end']+1]
            
            # Create a MultiIndex for columns
            multi_index = pd.MultiIndex.from_arrays([variety_header, week_header], names=['Variety', 'Week'])
            
            # Create the dataframe with the MultiIndex columns
            fruit_df = pd.DataFrame(data.values, columns=multi_index)
            
            # Rename the first column to 'receiver'
            if len(fruit_df.columns) > 0:
                first_col = fruit_df.columns[0]
                fruit_df = fruit_df.rename(columns={first_col: 'receiver'})
                print(f"Renamed first column '{first_col}' to 'receiver' in {fruit} dataframe")
            
            # Create a new dataframe with the same data but with a simple column index
            # This avoids the MultiIndex issues during unpivoting
            simple_df = pd.DataFrame()
            
            # Copy the 'receiver' column
            simple_df['receiver'] = fruit_df['receiver']
            
            # Copy the remaining columns with their values
            for col in fruit_df.columns[1:]:
                if isinstance(col, tuple):
                    # For MultiIndex columns, use the first level as the column name
                    simple_df[col[0]] = fruit_df[col]
            
            # Unpivot the dataframe
            fruit_df_unpivoted = unpivot_dataframe(simple_df, start_col_index=1)
            
            # Add a 'week' column based on the original MultiIndex
            # We need to extract the week information from the original column names
            fruit_df_unpivoted['week'] = None
            
            # For each variety in the unpivoted dataframe, find the corresponding week
            for idx, row in fruit_df_unpivoted.iterrows():
                variety = row['variety']
                # Find the original column that contains this variety
                for col in fruit_df.columns[1:]:
                    if isinstance(col, tuple) and col[0] == variety:
                        fruit_df_unpivoted.at[idx, 'week'] = col[1]
                        break
            
        else:
            # Get the header rows
            header_row1 = df.iloc[ranges['header_row']]  # First level of headers
            header_row2 = df.iloc[ranges['header_row'] + 1]  # Second level of headers (if applicable)
            
            # Get the data rows
            data = df.iloc[ranges['data_start']:ranges['data_end']+1]
            
            # Check if this dataframe has a compound column structure
            # We'll assume it does if the first two header rows have different values
            has_compound_structure = False
            if ranges['header_row'] + 1 < ranges['data_start']:
                # Check if the first two header rows have different values in the same columns
                for i in range(1, len(header_row1)):
                    if header_row1[i] != header_row2[i]:
                        has_compound_structure = True
                        break
            
            if has_compound_structure:
                print(f"{fruit.capitalize()} data has a compound column structure")
                
                # Create a MultiIndex for columns
                multi_index = pd.MultiIndex.from_arrays([header_row1, header_row2], names=['Variety', 'Week'])
                
                # Create the dataframe with the MultiIndex columns
                fruit_df = pd.DataFrame(data.values, columns=multi_index)
                
                # Rename the first column to 'receiver'
                if len(fruit_df.columns) > 0:
                    first_col = fruit_df.columns[0]
                    fruit_df = fruit_df.rename(columns={first_col: 'receiver'})
                    print(f"Renamed first column '{first_col}' to 'receiver' in {fruit} dataframe")
                
                # Drop the second column (index 1)
                if len(fruit_df.columns) > 1:
                    second_col = fruit_df.columns[1]
                    fruit_df = fruit_df.drop(columns=[second_col])
                    print(f"Dropped second column '{second_col}' from {fruit} dataframe")
                
                # Create a new dataframe with the same data but with a simple column index
                # This avoids the MultiIndex issues during unpivoting
                simple_df = pd.DataFrame()
                
                # Copy the 'receiver' column
                simple_df['receiver'] = fruit_df['receiver']
                
                # Copy the remaining columns with their values
                for col in fruit_df.columns[1:]:
                    if isinstance(col, tuple):
                        # For MultiIndex columns, use the first level as the column name
                        simple_df[col[0]] = fruit_df[col]
                
                # Unpivot the dataframe
                fruit_df_unpivoted = unpivot_dataframe(simple_df, start_col_index=1)
                
                # Add a 'week' column based on the original MultiIndex
                # We need to extract the week information from the original column names
                fruit_df_unpivoted['week'] = None
                
                # For each variety in the unpivoted dataframe, find the corresponding week
                for idx, row in fruit_df_unpivoted.iterrows():
                    variety = row['variety']
                    # Find the original column that contains this variety
                    for col in fruit_df.columns[1:]:
                        if isinstance(col, tuple) and col[0] == variety:
                            fruit_df_unpivoted.at[idx, 'week'] = col[1]
                            break
                
            else:
                # Create the dataframe with a simple column index
                fruit_df = pd.DataFrame(data.values, columns=header_row1)
                
                # Check if the first column header is NaN and handle it
                if len(fruit_df.columns) > 0:
                    first_col = fruit_df.columns[0]
                    
                    # If the first column header is NaN, use a default name
                    if pd.isna(first_col) or first_col == 'nan':
                        print(f"First column in {fruit} dataframe has no header name. Using default name 'column_0'.")
                        fruit_df.columns.values[0] = 'column_0'
                        first_col = 'column_0'
                    
                    # Rename the first column to 'receiver'
                    fruit_df = fruit_df.rename(columns={first_col: 'receiver'})
                    print(f"Renamed first column '{first_col}' to 'receiver' in {fruit} dataframe")
                
                # Drop the second column (index 1)
                if len(fruit_df.columns) > 1:
                    second_col = fruit_df.columns[1]
                    fruit_df = fruit_df.drop(columns=[second_col])
                    print(f"Dropped second column '{second_col}' from {fruit} dataframe")
                
                # Unpivot numeric columns from column 3 onwards
                fruit_df_unpivoted = unpivot_dataframe(fruit_df, start_col_index=1)
        
        # Store in the dictionary
        fruit_dfs[fruit] = fruit_df_unpivoted
        
        # Print shape
        print(f"{fruit.capitalize()} dataframe shape: {fruit_df_unpivoted.shape}")
    
    return fruit_dfs

def save_dataframes(dataframes_dict, output_dir='data/processed'):
    """
    Save all dataframes to CSV files in the specified output directory.
    
    Parameters:
    -----------
    dataframes_dict : dict
        Dictionary with names as keys and pandas DataFrames as values
    output_dir : str
        Directory to save the CSV files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save each dataframe to a CSV file
    for name, df in dataframes_dict.items():
        file_path = output_path / f"{name}.csv"
        df.to_csv(file_path, index=False)
        print(f"Saved {name} to {file_path}")

def main():
    """
    Main function to process all data files and create the required dataframes.
    """
    print("Processing all data files...")
    
    # Dictionary to store all dataframes
    all_dataframes = {}
    
    try:
        # Process Raw data
        print("\n=== Processing Raw Data ===")
        raw_df = process_raw_data()
        all_dataframes['raw'] = raw_df
        
        # Process Introductions data
        print("\n=== Processing Introductions Data ===")
        introductions_df = process_introductions_data()
        all_dataframes['introductions'] = introductions_df
        
        # Process Estimate Volumes data
        print("\n=== Processing Estimate Volumes Data ===")
        citrus_df, pome_df = process_estimate_volumes()
        all_dataframes['citrus_estimate'] = citrus_df
        all_dataframes['pome_estimate'] = pome_df
        
        # Process Demand Volumes data
        print("\n=== Processing Demand Volumes Data ===")
        fruit_dfs = process_demand_volumes()
        all_dataframes.update(fruit_dfs)
        
        # Save all dataframes to CSV files
        print("\n=== Saving All Dataframes ===")
        save_dataframes(all_dataframes)
        
        print("\nAll data processing completed successfully!")
        
        # Return the dictionary of all dataframes
        return all_dataframes
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

if __name__ == "__main__":
    # Process all data files
    all_dataframes = main()
    
    if all_dataframes:
        # Display information about each dataframe
        print("\n=== Dataframe Summary ===")
        for name, df in all_dataframes.items():
            print(f"\n{name.capitalize()}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {', '.join(df.columns)}") 