#%%
import re
import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_utils import PathManager

def count_words(doc_path):
    parsed_text_path = PathManager.get_parsed_text(doc_path)
    if parsed_text_path.exists():
        text = parsed_text_path.read_text(encoding='utf-8')
        # delete titles in markdown
        text = re.sub(r"^#+\s.*$", "", text, flags=re.MULTILINE) 
        # delete image in markdown
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        word_count = len(text.split())
        return word_count
    return 0


preview_lines = ["--- DOCUMENT STRUCTURE SUMMARY ---"]

raw_paths = [
    "mmlongbench-doc/2021-Apple-Catalog/2021-Apple-Catalog_page0.jpg",
    "mmlongbench-doc/2021-Apple-Catalog/2021-Apple-Catalog_page2.jpg",
    "mmlongbench-doc/2021-Apple-Catalog/2021-Apple-Catalog_page4.jpg",
]

doc_paths = [PathManager.encode_doc_path(raw_path) for raw_path in raw_paths]

for doc_path in doc_paths:
    parsed_json_path = PathManager.get_parsed_json(doc_path)
    page_num = int(Path(doc_path).stem.split("page")[-1])
    if parsed_json_path.exists():
        with open(parsed_json_path, "r", encoding="utf-8") as jf:
            # Handle different JSON structures (list of lists vs flat list)
            raw_data = json.load(jf)
            data = raw_data[0] if isinstance(raw_data[0], list) else raw_data

            summary = {
                "titles": [],
                "tables": 0,
                "images": 0,
                "equations": 0
            }
            
            for d in data:
                d_type = d.get("type", "")
                if d_type == "title":
                    # Extract text safely
                    content = d.get("content", {}).get("title_content", [{}])[0].get("content", "")
                    if content: summary["titles"].append(content)
                elif d_type == "table":
                    summary["tables"] += 1
                elif d_type == "image":
                    summary["images"] += 1
                elif d_type == "equation_interline":
                    summary["equations"] += 1
    
    content_desc = []
    if summary["titles"]: content_desc.append(f"Titles: {summary['titles']}")
    content_desc.append(f"Contains {count_words(doc_path)} words of text")
    if summary["tables"] > 0: content_desc.append(f"Contains {summary['tables']} Tables")
    if summary["images"] > 0: content_desc.append(f"Contains {summary['images']} Images/Figures/Charts")
    if summary["equations"] > 0: content_desc.append(f"Contains {summary['equations']} Equations")

    preview_lines.append(f"Page {page_num}: {' | '.join(content_desc)}")

print("\n".join(preview_lines))
# %%

