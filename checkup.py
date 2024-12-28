import json

CURRENT_DIRECTORY = "\\".join(__file__.split("\\")[:-1])+"\\"                           # Current directory (used for creating dataset file path variable)
DATASET_FILE_PATH = CURRENT_DIRECTORY+"dataset.json"                                    # Dataset file path

datasets = {}
with open(DATASET_FILE_PATH, "r+") as f:
    datasets = json.loads(f.read())
    
print("-------- DATASETS INFORMATIONS --------")
for key, value in datasets.items():
    print(f"{key}: {len(value)} datas")
    