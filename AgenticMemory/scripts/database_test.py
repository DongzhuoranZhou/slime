#%%
import os
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.data_utils import PathManager
from search_models import JinaV4Model, ColPaliModel

dataset = "longdocurl"
embed_model_name = "jinav4_dense"
device = "cuda:2"

# connect to database server
os.environ["NO_PROXY"] = "localhost"
client = QdrantClient(url="http://localhost:6333")

jsonl_path = PathManager.get_dataset_jsonl(dataset)

data = []
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# load embedding model
if embed_model_name == "jinav4_dense":
    embed_model = JinaV4Model(
        device=device,
        multivector=False
    )
elif embed_model_name == "jinav4_multivector":
    embed_model = JinaV4Model(
        device=device,
        multivector=True
    )
elif embed_model_name == "colpali_multivector":
    embed_model = ColPaliModel(
        device=device
    )

entry = data[0]
doc_name = entry['doc_name']
question = entry['question']

# Create query vector (you'll need to embed your query text)
query_vector = embed_model.embed_text(question)

# Search with document_name filter
search_results = client.query_points(
    collection_name=dataset,
    query=query_vector.tolist(),  # Your embedded query
    using=embed_model_name,
    limit=3,  # Number of results to return
    with_payload=True,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="document_name",
                match=MatchValue(value=doc_name)
            )
        ]
    )
)
# %%
all_document_names = []
next_page_offset = None

while True:
    # Scroll through points, fetching only the 'document_name' field
    # We set with_vectors=False to avoid downloading heavy multivectors
    scroll_result, next_page_offset = client.scroll(
        collection_name="mmlongdoc",
        limit=1000, 
        with_payload=["document_name", "full_img_path"], 
        with_vectors=False,
        offset=next_page_offset
    )

    for point in scroll_result:
        doc_name = point.payload.get("document_name")
        if doc_name:
            all_document_names.append(doc_name)

    # If next_page_offset is None, we've reached the end of the collection
    if next_page_offset is None:
        break

# Optional: Convert to a set if you only want unique document names
unique_docs = list(set(all_document_names))
print(f"Found {len(unique_docs)} unique documents.")
# %% test whether payload has been stored correctly
count = 0
for entry in data:
    doc_name = entry['doc_name']
    search_filter = Filter(
        must=[
            FieldCondition(
                key="document_name",
                match=MatchValue(value=doc_name)
            )
        ]
    )
    points, next_page_offset = client.scroll(
        collection_name="mmlongdoc",
        scroll_filter=search_filter,
        limit=3,
        with_payload=True,
        with_vectors=False  # Recommended to keep False for faster retrieval
    )

    # if there are points then count as successful retrieval
    if len(points) > 0:
        print(f"Retrieved {len(points)} points for document: {doc_name}")
        count += 1
    else:
        print(f"No points retrieved for document: {doc_name}")
    
print(f"Successful retrievals: {count} out of {len(data)}")
# %%
