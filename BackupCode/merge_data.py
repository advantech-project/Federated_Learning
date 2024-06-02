import pandas as pd
import os

def normalize_to_actual(normalized_value, min_value, max_value):
    return normalized_value * (max_value - min_value) + min_value


csv_names = ["2호남관", "2호북관", "5호동관", "하이테크센터"]
en_names = ["South of No.2", "North of No.2", "East of No.5", "HighTech Center"]


base_dir = "/home/ec2-user/environment"

for i in range(len(csv_names)):
    origin_csv = os.path.join(base_dir, "OriginCSV", f"{csv_names[i]}.csv")
    prediction_csv = os.path.join(base_dir, f"{csv_names[i]}_prediction.csv")
    output_csv_path = os.path.join(base_dir, f"{csv_names[i]}_data.csv")
    
    origin_csv = pd.read_csv(origin_csv)
    prediction_csv = pd.read_csv(prediction_csv)

    min_value = origin_csv['OriginalPowerUsage'].min()
    max_value = origin_csv['OriginalPowerUsage'].max()
    
    prediction_csv['OriginPrediction'] = prediction_csv['Prediction'].apply(
        lambda x: round(normalize_to_actual(x, min_value, max_value), 9)
    )
        
    merge = pd.merge(origin_csv, prediction_csv[['DateTime', 'Prediction', 'OriginPrediction']], on='DateTime', how='left')

    merge.to_csv(output_csv_path, index=False)

print("Merge!!")