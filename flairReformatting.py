import os
import numpy as np
import pandas as pd
import re
from scipy import stats
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("dataDir", help="ruta al directorio de los .csv FLAIR")
parser.add_argument("outputDir", help="ruta al directorio de salida")

args = parser.parse_args()

inter_dir=args.dataDir+'/processedData'


# ---------------------------- UTILIDADES ----------------------------

def tupleExtraction(df, tuplas):
    tuple_df=pd.DataFrame()
    for col in tuplas: 
        # Extract tuple values and clean them
        vals = [re.sub(r'[()]', '', str(x)) for x in df[col]]  # Convert to strings
        vals = [x.split(',') for x in vals]  

        # Create new DataFrame with the extracted values
        asciiNum = 97  # ASCII 'a'
        temp_df = pd.DataFrame()

        for row_idx, tpl in enumerate(vals):  
            for i in range(len(tpl)):
                newAscii = chr(asciiNum + i)  # Convert to letter
                newCol = f"{col}.{newAscii}"  
                
                if newCol not in temp_df.columns:
                    temp_df[newCol] = None  
                
                temp_df.loc[row_idx, newCol] = tpl[i]
                temp_df[newCol]=pd.to_numeric(temp_df[newCol])
                
        tuple_df = pd.concat([tuple_df, temp_df], axis=1)
    return tuple_df


def reformating(input_file):

    file_name = os.path.basename(input_file)

    if not os.path.exists(inter_dir):
        os.makedirs(inter_dir)

    df = pd.read_csv(input_file)

    # only considering columns feature name and segment as columns and values respectively
    df = df.set_index('Feature Name')
    df.drop(columns=['Image type', 'Feature Class'], inplace=True)

    df_transposed = df.T # trasnpose 

    output_file = inter_dir+'/'+file_name

    df_transposed.to_csv(output_file)

# ---------------------------- MAIN ----------------------------

files = os.listdir(args.dataDir)
finalDf = pd.DataFrame()

for file in files:

    if file.endswith(".csv"):

        print(f"---Procesando {file}...")
        df = pd.read_csv(f"{args.dataDir}/{file}")
        
        reformating(args.dataDir+'/'+file)
        df = pd.read_csv(inter_dir+'/'+file)
        
        tuplas = ['Spacing', 'Size', 'Spacing.1', 'Size.1', 'BoundingBox', 'CenterOfMassIndex', 'CenterOfMass']
        df_tuple=tupleExtraction(df, tuplas)

        df_tuple['Id']=df['Id']
        df.set_index('Id', inplace=True)
        df_tuple.set_index('Id', inplace=True)

        df=pd.merge(df_tuple, df.drop(columns=tuplas),  on='Id', how='inner')
        finalDf = pd.concat([finalDf, df])

finalDf.to_csv(args.outputDir+'/flair_df.csv')