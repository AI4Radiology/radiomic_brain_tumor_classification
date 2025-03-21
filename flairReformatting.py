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

    output_dir=args.dataDir+'/processedData'
    file_name = os.path.basename(input_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(input_file)

    df_transposed = df.T

    # dejar solo las columnas feature name y segment
    df_filtered = df_transposed.iloc[-2:]
    
    output_file = output_dir+'/'+file_name

    # Guardar el CSV transpuesto
    df_filtered.to_csv(output_file, header=False, index=False)

# ---------------------------- MAIN ----------------------------

files = os.listdir(args.dataDir)
finalDf = pd.DataFrame()

for file in files:

    if file.endswith(".csv"):

        print(f"---Procesando {file}...")
        df = pd.read_csv(f"{args.dataDir}/{file}")
        
        reformating(args.dataDir+'/'+file)
        df = pd.read_csv(f"/home/luis/Desktop/PDG/data/processedData/processed-{file}")
        
        tuplas = ['Spacing', 'Size', 'Spacing.1', 'Size.1', 'BoundingBox', 'CenterOfMassIndex', 'CenterOfMass']
        df_tuple=tupleExtraction(df, tuplas)

        df_tuple['Id']=df['Id']
        df.set_index('Id', inplace=True)
        df_tuple.set_index('Id', inplace=True)

        df=pd.merge(df_tuple, df.drop(columns=tuplas),  on='Id', how='inner')
    finalDf = pd.concat([finalDf, df], axis=0)

finalDf.to_csv(args.outputDir+'/flair_df.csv', header=False)