import ast
import os
import numpy as np
import pandas as pd
import re
import argparse
import pickle

# Example: override display settings
pd.set_option('display.max_rows', None)    # Show all rows
pd.set_option('display.max_columns', None) # Show all columns
pd.set_option('display.width', None)       # Don't wrap columns
pd.set_option('display.max_colwidth', None) # Show full column content

inter_dir='processedData'


# ---------------------------- UTILITIES ----------------------------

def tupleExtraction(df):
    for col in df.columns:
        try: 
            #print(f"Processing column: {col}")
            parsed = ast.literal_eval(df[col][0])  # [0] because its a dataframe with 1 row
            if isinstance(parsed, tuple):
                if (re.search('BoundingBox', col) or re.search('CenterOfMassIndex', col)): 
                    
                    vols=[]
                    all_coordinates = []
                    x_vals=[]
                    y_vals=[]
                    z_vals=[]
                    for row in df[col]:
                        parsed=ast.literal_eval(row)
                        coordinates=parsed[:3]
                        all_coordinates.append(coordinates)
                        
                        # Extract tuple values and clean them
                        volTuple=parsed[-3:]
                        vol=1
                        for num in volTuple:
                            vol *= num
                        vols.append(vol)
                        
                        vals = re.sub(r'[()]', '', str(coordinates))   # Convert to strings
                        print('VALS 1: ', vals)
                        vals = vals.split(',')
                        vals=[float(x.strip()) for x in vals]  # Convert to float
                        print('VALS 2: ', vals)
                        x_vals.append(vals[0])
                        y_vals.append(vals[1])
                        z_vals.append(vals[2])
                        # Create new DataFrame with the extracted values
                    
                    df.drop(columns=[col], inplace=True)  # Drop the original column
                    df[f"x.{col}"]= x_vals
                    df[f"y.{col}"]= y_vals  
                    df[f"z.{col}"]= z_vals

                    if (re.search('BoundingBox', col)):
                        df[col] = vols
                
                else: # El resto de columnas de tupla son tuplas de 3 medidas para el volumen
                    vols = []             
                    for row in df[col]:
                        volTuple=parsed[-3:]
                        vol=1
                        for num in volTuple:
                            vol *= num
                        vols.append(vol)
                    df[col] = vols 
        
        except (ValueError, SyntaxError) as e:
            #print(f"Error caught: {e}")
            pass
    print("TUPLE EXTRACTED SUCCESSFULLY")
    return df


def reformating(input_file):


    if not os.path.exists(inter_dir):
        os.makedirs(inter_dir)

    df = pd.read_csv(input_file.file)
    print("-----BEFORE REFORMATTING:\n", df)
    df['Feature Name']=df['Feature Name']+"/"+df['Feature Class']+"/"+df['Image type']
    
    # only considering columns 'Feature Name' and 'Segment...' as columns and values respectively
    df = df.set_index('Feature Name')
    df.drop(columns=['Image type', 'Feature Class'], inplace=True)
    
    df_transposed = df.T 
    
    id_col=[col for col in df_transposed.columns if col.startswith("Id/")][0]
    df_transposed.rename(columns={id_col: "Id"}, inplace=True)

    print("-----AFTER REFORMATTING:\n", df_transposed)
    
    output_file = inter_dir+'/'+input_file.filename

    df_transposed.to_csv(output_file) #writes processed file



def prepareData(df):

    sorted_columns = sorted(df.columns)
    #df = df[sorted_columns]  
    with open("resources/columns.pkl", "rb") as f:
        columns_list = pickle.load(f)

    df=df[columns_list]
    print("------AFTER EXCLUDING COLUMNS:\n", df)
    return df



# ---------------------------- MAIN ----------------------------
def process(input_file):
    pd.set_option('display.max_rows', None)    # Show all rows
    pd.set_option('display.max_columns', None) # Show all columns
    pd.set_option('display.width', None)       # Don't wrap columns
    pd.set_option('display.max_colwidth', None) # Show full column content

    if input_file.filename.endswith(".csv"):
        
        print(f"---Procesando {input_file.filename}...")
        
        reformating(input_file) 
        df = pd.read_csv(inter_dir+'/'+input_file.filename) #reads processed file
        print("-----BEFORE TUPLE EXTRACTION:\n", df)
        
        df=tupleExtraction(df)
        print(df)
        finalDf=df
    #finalDf.to_csv(args.outputDir+'/base_data.csv')
    finalDf.set_index('Id', inplace=True)
    finalDf = prepareData(finalDf)

    return finalDf