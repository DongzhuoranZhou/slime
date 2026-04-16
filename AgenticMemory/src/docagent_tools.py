from smolagents import Tool
from data_utils import PathManager
import PIL

class DocumentPreviewTool(Tool):
    name = "preview_document"
    description = (
        "MANDATORY FIRST STEP. "
        "Scans the document structure to generate a 'Table of Contents' summary. "
        "Returns titles, page numbers, and counts of words, images, tables, and equations for each page. "
        "Use this to locate WHICH page contains the information you need before reading text or loading images."
    )
    inputs = {
        "doc_path_list": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "The list of original document paths, "
                "formatted like ['path/to/doc_page4.jpg', 'path/to/path/to/doc_page2.jpg']."
                )
        }
    }
    output_type = "string"

    def forward(self, doc_path_list: list):
        import json
        from pathlib import Path
        try:
            doc_paths = PathManager.check_path_list(doc_path_list)
            preview_lines = ["--- DOCUMENT STRUCTURE SUMMARY ---"]

            for doc_path in doc_paths:
                # Robust page number extraction
                try:
                    page_num = int(Path(doc_path).stem.split("page")[-1])
                except ValueError:
                    page_num = "Unknown"
            
                # Initialize counters
                summary = {
                    "titles": [], 
                    "paragraphs": 0, 
                    "tables": 0, 
                    "images": 0,
                    "equations": 0
                }

                # Load JSON metadata
                try:
                    parsed_json_path = PathManager.get_parsed_json(doc_path)
                    if parsed_json_path.exists():
                        with open(parsed_json_path, "r", encoding="utf-8") as jf:
                            # Handle different JSON structures (list of lists vs flat list)
                            raw_data = json.load(jf)
                            data = raw_data[0] if isinstance(raw_data[0], list) else raw_data
                            
                            for d in data:
                                d_type = d.get("type", "")
                                if d_type == "title":
                                    # Extract text safely
                                    content = d.get("content", {}).get("title_content", [{}])[0].get("content", "")
                                    if content: summary["titles"].append(content)
                                elif d_type == "paragraph":
                                    summary["paragraphs"] += 1
                                elif d_type == "table":
                                    summary["tables"] += 1
                                elif d_type == "image":
                                    summary["images"] += 1
                                elif d_type == "equation_interline":
                                    summary["equations"] += 1
                except Exception as e:
                    preview_lines.append(f"Page {page_num}: [Error reading metadata: {e}]")
                    continue

                # Format the output for the LLM
                # We emphasize Visuals (Tables/Images) because they usually require the ImageLoader
                content_desc = []
                if summary["titles"]: content_desc.append(f"Titles: {summary['titles']}")
                content_desc.append(f"Contains {self.count_words(doc_path)} words of text")
                if summary["tables"] > 0: content_desc.append(f"Contains {summary['tables']} Tables")
                if summary["images"] > 0: content_desc.append(f"Contains {summary['images']} Images/Figures/Charts")
                if summary["equations"] > 0: content_desc.append(f"Contains {summary['equations']} Equations")
                preview_lines.append(f"Page {page_num}: {' | '.join(content_desc)}")
            
            return "\n".join(preview_lines)
                
                # load low resolution images
                # with PIL.Image.open(doc_path) as img_ref:
                #     img_ref.load()

        except Exception as e:
            return f"Error reading document: {str(e)}"
    
    def count_words(self, doc_path: str) -> int:
        import re
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
        
class DocumentReaderTool(Tool):
    name = "read_document_text"
    description = (
        "Retrieves the parsed text content for specific pages. "
        "DECISION GUIDE: Prioritize this tool if the answer likely resides in the text (e.g., summaries, terms, specific clauses), "
        "or if the 'preview_document' indicates the page is TEXT-DOMINANT (High Word Count, few Figures/Charts)."
        "STRATEGY: Start here for efficiency. Use this tool if the question does not indicate a strict need for visual information."
        "Exception: If you already tried this tool and the text was insufficient, do not repeat it—switch to 'load_specific_pages'."
    )
    inputs = {
        "doc_path_list": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "The list of original document paths, "
                "formatted like ['path/to/doc_page4.jpg', 'path/to/path/to/doc_page2.jpg']."
            )
        },
        "page_numbers": {
            "type": "array",
            "items": {"type": "integer"},
            "description": (
                "The list of EXPLICIT Page IDs to load. "
                "RULE: Do not use the list index (0, 1). You must use the actual number found in the filename. "
                "Example: If you want to view '.../doc_page8.jpg', you must pass [8]. "
                "Values must strictly be a subset of the numbers present in 'doc_path_list'."
            )
        }
    }
    output_type = "string"

    def forward(self, doc_path_list: list, page_numbers: list) -> str:
        from pathlib import Path
        page_nums = [int(Path(doc_path).stem.split("page")[-1]) for doc_path in doc_path_list]
        if not all(page in page_nums for page in page_numbers):
            return f"Error: page_indeces must be selected from {page_nums}."
        try:
            doc_paths = PathManager.check_path_list(doc_path_list)

            content = "Document Content:\n"
            for doc_path in doc_paths:
                page_num = int(Path(doc_path).stem.split("page")[-1])
                if page_num in page_numbers:
                    page_text = PathManager.get_parsed_text(doc_path).read_text(encoding='utf-8')
                    content += f"\n--- Page {page_num} ---\n{page_text}\n"
                
            return content

        except Exception as e:
            return f"Error reading document: {str(e)}"

class PageLoaderTool(Tool):
    name = "load_specific_pages"
    description = (
        "Visual Inspection Tool. Loads high-resolution images for a specific subset of pages. "
        "DECISION GUIDE: Use this tool if the question explicitly requires visual interpretation "
        "(e.g., 'describe the chart') OR if the 'preview_document' "
        "shows the page is VISUAL-DOMINANT (e.g., Slides). "
        "STRATEGY: This is also your 'Escalation' tool. If you previously called 'read_document_text' "
        "and found the output insufficient, switch to this tool."
    )
    inputs = {
        "doc_path_list": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "The list of original document paths, "
                "formatted like ['path/to/doc_page4.jpg', 'path/to/doc_page2.jpg']."
                )
        },
        "page_numbers": {
            "type": "array",
            "items": {"type": "integer"},
            "description": (
                "The list of EXPLICIT Page IDs to load. "
                "RULE: Do not use the list index (0, 1). You must use the actual number found in the filename. "
                "Example: If you want to view '.../doc_page8.jpg', you must pass [8]. "
                "Values must strictly be a subset of the numbers present in 'doc_path_list'."
            )
        }
    }
    output_type = "image_list"

    def forward(self, doc_path_list: list, page_numbers: list):
        from pathlib import Path
        page_nums = [int(Path(doc_path).stem.split("page")[-1]) for doc_path in doc_path_list]
        if not all(page in page_nums for page in page_numbers):
            return f"Error: page_indeces must be selected from {page_nums}."
        try:
            doc_paths = PathManager.check_path_list(doc_path_list)

            img_files = []
            for doc_path in doc_paths:
                page_num = int(Path(doc_path).stem.split("page")[-1])
                if page_num in page_numbers:
                    img_path = PathManager.get_img_path(doc_path)
                    with PIL.Image.open(str(img_path)) as img_ref:
                        img_ref.load()
                        img_files.append(img_ref)
                        
            if img_files:
                return img_files
            else:
                return "No images found for the provided document paths and page indices."

        except Exception as e:
            return f"Error reading document: {str(e)}"
        
# class ImageLoaderTool(Tool):
#     name = "load_document_images"
#     description = (
#         "Visual Perception Tool. Loads high-resolution images into memory. "
#         "Use this ONLY when the 'preview_document' tool indicates that a specific page "
#         "contains a table, chart, or visual element needed to answer the question. "
#     )
#     inputs = {
#         "doc_path_list": {
#             "type": "array",
#             "items": {"type": "string"},
#             "description": (
#                 "The list of original document paths, "
#                 "formatted like ['path/to/image1.jpg', 'path/to/image2.jpg']."
#                 )
#         },
#         "page_indices": {
#             "type": "array",
#             "items": {"type": "integer"},
#             "description": "Optional list of page numbers to load (e.g., [0, 5]). "
#         }
#     }
#     output_type = "image_list"

#     def forward(self, doc_path_list: list, page_indices: list):
#         try:
#             doc_paths = PathManager.check_path_list(doc_path_list)

#             img_files = []
#             for doc_path in doc_paths:
#                 parsed_img_paths = PathManager.get_parsed_img(doc_path)
#                 for img_path in parsed_img_paths:
#                     with PIL.Image.open(str(img_path)) as img_ref:
#                         img_ref.load()
#                         img_files.append(img_ref)

#             if img_files:
#                 return img_files
#             else:
#                 return "No images found for the provided document paths."

#         except Exception as e:
#             return f"Error reading document: {str(e)}"
