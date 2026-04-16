#%%
from pathlib import Path
from tqdm import tqdm
import json
import os

#%%

path = "/lc/data/parsed_doc/mmlongdoc"
unique_types = set()
for root, _, files in tqdm(os.walk(path)):
    for file in files:
        # Filter strings directly (faster than creating Path objects)
        if file.endswith(".json") and "content_list_v2.json" in file:
            file_path = os.path.join(root, file)
            with open(file_path, "r") as jf:
                data = json.load(jf)
                for item in data[0]:
                    unique_types.add(item["type"])
# %%
path = "/lc/data/parsed_doc/mmlongdoc"
unique_types = set()
for root, _, files in tqdm(os.walk(path)):
    for file in files:
        # Filter strings directly (faster than creating Path objects)
        if file.endswith(".json") and "content_list.json" in file:
            file_path = os.path.join(root, file)
            with open(file_path, "r") as jf:
                data = json.load(jf)
                for item in data:
                    unique_types.add(item["type"])

#%%
path = "/lc/data/parsed_doc/mmlongdoc"
unique_types = set()
for root, _, files in tqdm(os.walk(path)):
    for file in files:
        # Filter strings directly (faster than creating Path objects)
        if file.endswith(".json") and "model.json" in file:
            file_path = os.path.join(root, file)
            with open(file_path, "r") as jf:
                data = json.load(jf)
                for item in data[0]:
                    unique_types.add(item["type"])

# %%
def test():
    path = "/lc/data/parsed_doc/mmlongdoc"
    unique_types = set()
    for root, _, files in tqdm(os.walk(path)):
        for file in files:
            # Filter strings directly (faster than creating Path objects)
            if file.endswith(".json") and "content_list_v2.json" in file:
                file_path = os.path.join(root, file)
                with open(file_path, "r") as jf:
                    data = json.load(jf)
                    for item in data[0]:
                        if item["type"] == "page_header":
                            return data

data = test()
# %%
