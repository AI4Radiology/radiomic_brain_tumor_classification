import ast
import os
import numpy as np
import pandas as pd
import re
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("dataDir", help="ruta al directorio de los .csv FLAIR")
parser.add_argument("outputDir", help="ruta al directorio de salida")

args = parser.parse_args()

inter_dir=args.dataDir+'/processedData'


# ---------------------------- UTILITIES ----------------------------

def tupleExtraction(df):
    for col in df.columns:
        print(f"Processing column: {col}")
        try: 
            parsed = ast.literal_eval(df[col][0])  # [0] because its a dataframe with 1 row
            if isinstance(parsed, tuple):
                if(len(parsed) == 3): 
                    vol=1
                    for num in parsed:
                        vol *= num
                    df[col] = vol  
                elif (len(parsed) == 6): #if the tuple has 6 dimensions, its the bounding box
                    coordenates = parsed[:3]
                    volTuple=parsed[-3:]
                    vol=1
                    for num in volTuple:
                        vol *= num
                    df[col] = vol
                    # Extract tuple values and clean them
                    vals = [re.sub(r'[()]', '', str(coordenates)) ]  # Convert to strings
                    print('VALS 1: ', vals)
                    vals = [vals[0].split(',')] #vals[0] because re.sub returns a list but we only need the one element inside.
                    vals=vals[0] # same thing as above
                    print('VALS 2: ', vals)
                    # Create new DataFrame with the extracted values
                    asciiNum = 97  # ASCII 'a'
                    temp_df = pd.DataFrame()

                    i=0
                    for val in enumerate(vals):  
                        newAscii = chr(asciiNum + i)  # Convert to letter
                        newCol = f"{newAscii}.{col}"  
                        i+=1
                        if newCol not in temp_df.columns:
                            temp_df[newCol] = None  
                        
                        temp_df[newCol] = float(val[0])
                        temp_df[newCol]=pd.to_numeric(temp_df[newCol])
                        print(temp_df[newCol].dtype)
                    print('TEMP DF: ', temp_df)
                    df = pd.concat([df, temp_df], axis=1)
        except (ValueError, SyntaxError) as e:
            print(f"Error caught: {e}")
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

finalDf.to_csv(args.outputDir+'/flair_df.csv')