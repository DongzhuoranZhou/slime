#%%
import os
from pathlib import Path
from qdrant_client import QdrantClient, models

os.environ["NO_PROXY"] = "localhost"
client = QdrantClient("http://localhost:6333")
collection_name = "longdocurl"

next_page_offset = None

while True:
    # 1. Scroll to get points with their 'full_img_path'
    # We skip vectors to keep the response small and fast
    scroll_result, next_page_offset = client.scroll(
        collection_name=collection_name,
        limit=100,
        with_payload=["full_img_path"],
        with_vectors=False,
        offset=next_page_offset
    )

    for point in scroll_result:
        full_path = point.payload.get("full_img_path")
        if not full_path:
            print(f"Point ID {point.id} has no 'full_img_path' payload. Skipping.")
            continue
            
        # 2. Extract the CORRECT document_name
        # Logic: If path is /lc/data/DocA/page_1.jpg -> doc name is "DocA"
        path_obj = Path(full_path)
        
        # ADJUST THIS LOGIC based on your actual folder structure:
        # e.g., correct_name = path_obj.parent.name 
        # or    correct_name = path_obj.stem.split('_')[0]
        correct_name = path_obj.parent.name 

        # 3. Update the point with the corrected name
        client.set_payload(
            collection_name=collection_name,
            payload={"document_name": correct_name},
            points=[point.id]
        )

    if next_page_offset is None:
        break

print("All document names have been corrected.")
# %%
