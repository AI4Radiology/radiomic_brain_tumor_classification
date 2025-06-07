import ast
import os
import numpy as np
import pandas as pd
import re

class FlairReformatter:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.processed_data_temp_dir_name = "processedData"
        self.inter_dir = os.path.join(data_dir, self.processed_data_temp_dir_name)
        os.makedirs(self.inter_dir, exist_ok=True)

    def tuple_extraction(self, df):
        """Extracts and reformat tuple-like string data from DataFrame columns."""
        for col in df.columns:
            try: 
                parsed = ast.literal_eval(df[col].iloc[0])
                if isinstance(parsed, tuple):
                    if (re.search('BoundingBox', col) or re.search('CenterOfMassIndex', col)): 
                        vols = []
                        x_vals = []
                        y_vals = []
                        z_vals = []
                        for _, row_val in df[col].items():
                            parsed_row = ast.literal_eval(row_val)
                            coordinates = parsed_row[:3]
                            
                            volTuple = parsed_row[-3:]
                            vol = 1
                            for num in volTuple:
                                vol *= num
                            vols.append(vol)
                            
                            vals = re.sub(r'[()]', '', str(coordinates))
                            vals = vals.split(',')
                            vals = [float(x.strip()) for x in vals]
                            x_vals.append(vals[0])
                            y_vals.append(vals[1])
                            z_vals.append(vals[2])
                        
                        df.drop(columns=[col], inplace=True)
                        df[f"x.{col}"] = x_vals
                        df[f"y.{col}"] = y_vals  
                        df[f"z.{col}"] = z_vals

                        if (re.search('BoundingBox', col)):
                            df[col] = vols
                    
                    else: # Other tuple columns are 3-measure volumes
                        vols = []             
                        for _, row_val in df[col].items():
                            parsed_row = ast.literal_eval(row_val)
                            volTuple = parsed_row[-3:]
                            vol = 1
                            for num in volTuple:
                                vol *= num
                            vols.append(vol)
                        df[col] = vols 
            
            except (ValueError, SyntaxError, IndexError):
                pass

        return df

    def reformat_single_file(self, input_file_path, output_file_path):
        """Reformats a single FLAIR CSV file by restructuring columns."""
        df = pd.read_csv(input_file_path)

        df['Feature Name'] = df['Feature Name']+"/"+df['Feature Class']+"/"+df['Image type']
        df = df.set_index('Feature Name')
        df.drop(columns=['Image type', 'Feature Class'], inplace=True)

        df_transposed = df.T 
        
        id_col = [col for col in df_transposed.columns if col.startswith("Id/")][0]
        df_transposed.rename(columns={id_col: "Id"}, inplace=True)

        df_transposed.to_csv(output_file_path)

    def prepare_data(self, df):
        """Prepares the DataFrame for model training by cleaning and selecting features."""
        # Column ordering
        sorted_columns = sorted(df.columns)
        df = df[sorted_columns].copy()  # Create a copy to avoid warnings
        
        # Object columns exclusion (only contains meta data)
        df_meta = df.select_dtypes(include=['object'])
        columns_to_drop = list(df_meta.columns) + ['Minimum/Image-original/diagnostics']
        df = df.drop(columns=columns_to_drop)  # Use drop() instead of inplace
        
        # Remove columns with only one unique value
        df = df.loc[:, df.nunique() > 1]

        # Extract 15 most correlated columns with target variable 'highGrade'    
        df_cp = df.copy()
        correlation_with_target = df.corr(numeric_only=True)['highGrade'].drop('highGrade').sort_values(ascending=False)
        top = correlation_with_target.abs().sort_values(ascending=False).head(15)
        cols_to_drop = [col for col in df.columns if col not in top.index and col != 'highGrade']
        df = df.drop(columns=cols_to_drop)  # Use drop() instead of inplace
        df['highGrade'] = df_cp['highGrade']
        
        # Outlier Handling (capping)
        for col in df.drop(columns=['highGrade']).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 8 * IQR
            upper_bound = Q3 + 8 * IQR

            df[col] = df[col].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))
        return df

    def process_data(self):
        """Main method to process all FLAIR data files."""
        all_processed_dfs = []

        files = os.listdir(self.data_dir)
        for file in files:
            if file.endswith(".csv"):
                print(f"---Processing {file}...")
                input_file_path = os.path.join(self.data_dir, file)
                
                # Step 1: Reformat and transpose
                temp_output_file = os.path.join(self.inter_dir, file)
                self.reformat_single_file(input_file_path, temp_output_file)
                
                df_reformatted = pd.read_csv(temp_output_file)
                
                # Step 2: Tuple extraction
                df_extracted = self.tuple_extraction(df_reformatted.copy())

                # Step 3: Set up labels based on filename (tumor grade)
                match = re.search(r'\d', file)
                if match:
                    first_digit = match.group()
                    if first_digit == '1' or first_digit == '2':
                        df_extracted['highGrade'] = False
                    else:
                        df_extracted['highGrade'] = True
                    all_processed_dfs.append(df_extracted)
                else:
                    print(f"Warning: Could not determine highGrade for file {file}. Skipping.")

        if not all_processed_dfs:
            print("No valid FLAIR CSV files processed.")
            return None

        final_df = pd.concat(all_processed_dfs, ignore_index=True)
        final_df.set_index('Id', inplace=True)

        # Step 4: Prepare final data (column selection, outlier handling)
        prepared_df = self.prepare_data(final_df.copy())

        # Save prepared_data.csv
        output_path = os.path.join(self.output_dir, "prepared_data.csv")
        prepared_df.to_csv(output_path)
        print(f"Prepared data saved to: {output_path}")

        # Return column names for later use
        return prepared_df.drop(columns=['highGrade']).columns.tolist() 