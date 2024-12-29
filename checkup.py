import json
from variables import DATASET_FILE_PATH

datasets = {}
with open(DATASET_FILE_PATH, "r+") as f:
    datasets = json.loads(f.read())
    
print("-------- DATASETS INFORMATIONS --------")
total = 0
for key, value in datasets.items():
    print(f"{key}\t: {len(value)} data")
    total += len(value)
print(f"Total\t: {total} data")
