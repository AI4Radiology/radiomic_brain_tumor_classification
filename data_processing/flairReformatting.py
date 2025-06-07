import ast
import os
import numpy as np
import pandas as pd
import re
import argparse
import pickle

parser = argparse.ArgumentParser()

parser.add_argument("dataDir", help="ruta al directorio de los .csv FLAIR")
parser.add_argument("outputDir", help="ruta al directorio de salida")

args = parser.parse_args()

inter_dir=args.dataDir+'/processedData'


# ---------------------------- UTILITIES ----------------------------

def tupleExtraction(df):
    for col in df.columns:
        try: 
            print(f"Processing column: {col}")
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

    return df


def reformating(input_file):

    file_name = os.path.basename(input_file)

    if not os.path.exists(inter_dir):
        os.makedirs(inter_dir)

    df = pd.read_csv(input_file)

    df['Feature Name']=df['Feature Name']+"/"+df['Feature Class']+"/"+df['Image type']
    print(df.shape)
    # only considering columns 'Feature Name' and 'Segment...' as columns and values respectively
    df = df.set_index('Feature Name')
    df.drop(columns=['Image type', 'Feature Class'], inplace=True)

    df_transposed = df.T 
    
    id_col=[col for col in df_transposed.columns if col.startswith("Id/")][0]
    df_transposed.rename(columns={id_col: "Id"}, inplace=True)

    output_file = inter_dir+'/'+file_name

    df_transposed.to_csv(output_file) #writes processed file



def prepareData(df):
    #print('CHECKPOINT 0: ', df.index) 
    #column ordering
    sorted_columns = sorted(df.columns)
    df=df[sorted_columns]
    
    # Object columns exclusion (only contains meta data)
    df_meta=df.select_dtypes(include=['object'])
    df.drop(columns = list(df_meta.columns) + ['Minimum/Image-original/diagnostics'], inplace=True)
    
    # remove columns with only one unique value
    df = df.loc[:, df.nunique() > 1]

    # Extract 15 most correlated columns with target variable 'highGrade'    
    df_cp=df.copy()
    print(df.shape)
    correlation_with_target = df.corr()['highGrade'].drop('highGrade').sort_values(ascending=False)
    top = correlation_with_target.abs().sort_values(ascending=False).head(15)
    cols_to_drop=[col for col in df.columns if col not in top.index]
    df.drop(columns=cols_to_drop, inplace=True)
    df['highGrade']=df_cp['highGrade']
    print(df.shape)
    print('Columns selected: ', df.columns)
    # OUTLIER HANDLING
    print('CHECKPOINT 1: ', df.index)
    for col in df.drop(columns=['highGrade']).columns:
        print(f"// dtype: {df[col].dtype}")
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 8* IQR
        upper_bound = Q3 + 8* IQR

        # Reeplace outliers by limits (capping)
        df[col] = df[col].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))
    print('CHECKPOINT 2: ', df.shape)
    return df



# ---------------------------- MAIN ----------------------------

files = os.listdir(args.dataDir)
finalDf = pd.DataFrame()

for file in files:

    if file.endswith(".csv"):
        
        print(f"---Procesando {file}...")
        df = pd.read_csv(f"{args.dataDir}/{file}")
        
        reformating(args.dataDir+'/'+file) 
        df = pd.read_csv(inter_dir+'/'+file) #reads processed file
        
        df=tupleExtraction(df)
        
        # Setting up labels based for tumor grade (high or low) implied by the file name 
        match = re.search(r'\d', file)
        first_digit = match.group() # There is no error handlidng because if there is no match, file name is not valid.
        if first_digit == '1' or first_digit == '2':
            df['highGrade'] = False
        else:
            df['highGrade'] = True 
        finalDf = pd.concat([finalDf, df])

finalDf.to_csv(args.outputDir+'/base_data.csv')
finalDf.set_index('Id', inplace=True)
finalDf = prepareData(finalDf)
finalDf.to_csv(args.outputDir+'/prepared_data.csv')


# Save the list
with open("columns.pkl", "wb") as f:
    pickle.dump(finalDf.drop(columns=['highGrade']).columns, f)

'''# Load it later
with open("my_list.pkl", "rb") as f:
    loaded_list = pickle.load(f)'''
