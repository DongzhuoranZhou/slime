#%%
import os
import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.data_utils import PathManager
from tqdm import tqdm

dataset = "longdocurl"
embed_model_name = "jinav4_dense"

# connect to database server
os.environ["NO_PROXY"] = "localhost"
client = QdrantClient(url="http://localhost:6333")

# get total points in the collection
collection_info = client.get_collection(collection_name=dataset)
total_points = collection_info.points_count
print(f"Total points in collection '{dataset}': {total_points}")

next_page_offset = None
document_names = set()
num_points_processed = 0
while True:
    # 1. Scroll to get points with their 'full_img_path'
    # We skip vectors to keep the response small and fast
    scroll_result, next_page_offset = client.scroll(
        collection_name=dataset,
        limit=100,
        with_payload=["document_name"],
        with_vectors=False,
        offset=next_page_offset
    )

    for point in scroll_result:
        document_names.add(point.payload["document_name"])
    
    num_points_processed += len(scroll_result)
    print(f"Processed {num_points_processed}/{total_points} points...", end='\r')

    if next_page_offset is None:
        break

for doc_name in tqdm(document_names, total=len(document_names)):
    print(doc_name)
    query_filter = Filter(
        must=[
            FieldCondition(
                key="document_name",
                match=MatchValue(value=doc_name)
            )
        ]
    )

    points, _ = client.scroll(
        collection_name=dataset,
        scroll_filter=query_filter,
        limit=1000,
        with_payload=True,
        with_vectors=False  # Recommended to keep False for faster retrieval
    )

    for point in points:
        if "page_num" not in point.payload:
            print(f"Point ID {point.id} missing 'page_num' payload.")
            continue

        new_page_num = int(point.payload["page_num"]) + 1
        client.set_payload(
            collection_name=dataset,
            payload={"page_num": new_page_num},
            points=[point.id]
        )

#%%
total_doc_nums = len(document_names)
successful_docs = 0
for doc_name in tqdm(document_names, total=len(document_names)):
    print(doc_name)
    query_filter = Filter(
        must=[
            FieldCondition(
                key="document_name",
                match=MatchValue(value=doc_name)
            )
        ]
    )

    points, _ = client.scroll(
        collection_name=dataset,
        scroll_filter=query_filter,
        limit=1000,
        with_payload=True,
        with_vectors=False  # Recommended to keep False for faster retrieval
    )

    page_nums = [point.payload["page_num"] for point in points if "page_num" in point.payload]
    if 0 in page_nums:
        print(f"Error: Found page_num 0 in document '{doc_name}'")
    else:
        successful_docs += 1

print(f"Successfully updated page numbers for {successful_docs}/{total_doc_nums} documents.")

# %%
import os
import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.data_utils import PathManager
from tqdm import tqdm

dataset = "longdocurl"
embed_model_name = "jinav4_dense"

# connect to database server
os.environ["NO_PROXY"] = "localhost"
client = QdrantClient(url="http://localhost:6333")

# get total points in the collection
collection_info = client.get_collection(collection_name=dataset)
total_points = collection_info.points_count
print(f"Total points in collection '{dataset}': {total_points}")

next_page_offset = None
document_names = set()
num_points_processed = 0
while True:
    # 1. Scroll to get points with their 'full_img_path'
    # We skip vectors to keep the response small and fast
    scroll_result, next_page_offset = client.scroll(
        collection_name=dataset,
        limit=100,
        with_payload=["document_name"],
        with_vectors=False,
        offset=next_page_offset
    )

    for point in scroll_result:
        document_names.add(point.payload["document_name"])
    
    num_points_processed += len(scroll_result)
    print(f"Processed {num_points_processed}/{total_points} points...", end='\r')

    if next_page_offset is None:
        break


# %%
