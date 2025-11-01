import os
import pickle
import pandas as pd
from tqdm import tqdm
import re

DATA_PATH="/mnt/sdd/rag/ragdata_pa"
all_metadata = []

def get_batch_number(f):
    m = re.search(r"chunks_batch_(\d+)\.pkl", f)
    if m:
        return int(m.group(1))
    return -1

chunk_files = [
    f for f in os.listdir(DATA_PATH)
    if f.startswith("chunks_batch_") and f.endswith(".pkl")
]

chunk_files.sort(key=get_batch_number)

print(f"FOUND {len(chunk_files)} METADATA BATCH FILES ---------COMPILING NOW")

for f in tqdm(chunk_files, desc="LOADING METADATA"):
    filepath=os.path.join(DATA_PATH, f)
    try:
        with open(filepath, "rb") as file:
            data=pickle.load(file)
            all_metadata.extend(data["metadata"])
    except Exception as e:
        print(f"ERROR LOADING {f}:{e}")

    metadata_df = pd.DataFrame(all_metadata)

    output_file=os.path.join(DATA_PATH, "pa_metadata.pkl")
    metadata_df.to_pickle(output_file)

    print("---------COMPILING METADATA COMPLETED")
    print(f"total chunks/vectors processed: {len(metadata_df)}")
    print(f"METADATA SAVED TO: {output_file}")

