import os
import re
import pickle
import faiss
from tqdm import tqdm

DATA_PATH="/mnt/sdd/rag/ragdata"
FINAL_FAISS_PATH = os.path.join(DATA_PATH, "faiss_index_832737.idx")
FINAL_META_PATH = os.path.join(DATA_PATH, "metadata_832737.pkl")

total_faiss_size=0
total_meta_size=0

print(f"looking for partial indexes and metadata in {DATA_PATH}")

def format_size(num_bytes):
    for unit in ["B", 'KB', 'MB', 'GB']:
        if num_bytes<1024:
            return f"{num_bytes:.2f}{unit}"
        num_bytes/=1024
    return f"{num_bytes:.2f}TB"

faiss_files = sorted(
    [f for f in os.listdir(DATA_PATH) if re.match(r"faiss_partial_\d+\.idx", f)],
    key = lambda x: int(re.findall(r"\d+", x)[0])
)

total_faiss_size=sum(os.path.getsize(os.path.join(DATA_PATH, f)) for f in faiss_files)
print(f"total faiss partial sizes: {format_size(total_faiss_size)}")

meta_files = sorted(
    [f for f in os.listdir(DATA_PATH) if re.match(r"metadata_partial_\d+\.pkl", f)],
    key=lambda x: int(re.findall(r"\d+", x)[0])
)

total_meta_size=sum(os.path.getsize(os.path.join(DATA_PATH, f)) for f in meta_files)
print(f"total metadata partial sizes: {format_size(total_meta_size)}")

#####idx
print(f"FOUND {len(faiss_files)} FAISS PARTIAL INDEXES. MERGING NOW")

grouped = [faiss_files[i:i+3] for i in range(0,len(faiss_files), 3)]
merged_grps=[]

for gi, grp in enumerate(grouped, 1):
    print(f"\n merging faiss group {gi}/{len(grouped)} ({len(grp)} files)")
    base_index = faiss.read_index(os.path.join(DATA_PATH, grp[0]))

    for f in tqdm(grp[1:], desc="MERGING FAISS FILES GROUP {gi}"):
        tmp_index = faiss.read_index(os.path.join(DATA_PATH, f))
        xb=tmp_index.reconstruct_n(0, tmp_index.ntotal)
        base_index.add(xb)
        del tmp_index

    outpath=os.path.join(DATA_PATH, f"group_{gi}.idx")
    faiss.write_index(base_index, outpath)
    merged_grps.append(outpath)
    del base_index

print("\n created faiss groups")
for g in merged_grps:
    print(" ", g)

print("\n merging all groups into one index")

final_index = faiss.read_index(merged_grps[0])
for g in tqdm(merged_grps[1:], desc="merging merged groups"):
    tmp=faiss.read_index(g)
    xb = tmp.reconstruct_n(0,tmp.ntotal)
    final_index.add(xb)
    del tmp

faiss.write_index(final_index, FINAL_FAISS_PATH)
print(f"MERGED FAISS INDEX TO {FINAL_FAISS_PATH}")

#####meta
print(f"FOUND {len(meta_files)} PARTIAL METADATA. MERGING NOW")

all_meta=[]
with open(FINAL_META_PATH, "wb") as outfile:
    for f in tqdm(meta_files, desc="MERGING METADATA"):
        with open(os.path.join(DATA_PATH, f), "rb") as infile:
            data=pickle.load(infile)
            pickle.dump(data, outfile)
        del data

print(f"MERGED METADATA TO {FINAL_META_PATH}")

print("----------------------")

merged_faiss_size = os.path.getsize(FINAL_FAISS_PATH)
print(f"final faiss size: {merged_faiss_size}")

merged_meta_size = os.path.getsize(FINAL_META_PATH)
print(f"final metadata size: {merged_meta_size}")