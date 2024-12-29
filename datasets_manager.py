import json
from variables import DATASET_FILE_PATH
from typing import Dict, List

def load_dataset() -> Dict[str, List[List[float]]]:
    current_datasets = {"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[],"9":[],"0":[]}
    with open(DATASET_FILE_PATH, "r+") as f:
        current_datasets = json.loads(f.read())
    return current_datasets