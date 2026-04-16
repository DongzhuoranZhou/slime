from typing import Optional, Any, Dict, List
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from smolagents.tools import Tool
from smolagents.agent_types import AgentImageList
from search_models import BaseEmbeddingModel


def _resize_if_needed(img, max_size):
    """Resize PIL image so its longest edge == max_size, preserving aspect ratio.
    Returns img unchanged if max_size is None or image already fits."""
    if max_size is None:
        return img
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    import PIL.Image
    ratio = max_size / max(w, h)
    return img.resize((int(w * ratio), int(h * ratio)), PIL.Image.LANCZOS)

class SearchDatabaseTool(Tool):
    """
    A tool that searches a Qdrant vector database for relevant documents.
    
    This tool embeds the query text and searches for the most similar documents
    filtering by document name.
    """
    
    name = "search_database"
    description = """Searches a vector database for the top document pages most semantically relevant to your query.

**First Search**: You MUST use the user's question EXACTLY as provided in square brackets [ ]. Do NOT modify, rephrase, or add keywords.

**Subsequent Searches** (2nd, 3rd, etc.): Refine based on what was found vs. what's missing - adjust content type, terminology, or scope as needed.

**Subsequent Searches** (only after first search): Refine based on what was found vs. what's missing. 
**IMPORTANT**: Always include the `excluded_pages` parameter with page numbers you've already seen to avoid repetitive results and save tokens.

Returns: Search results with page numbers and metadata.
"""
    
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "If this tool is chosen as first search: Extract and use the EXACT user question from [ ] brackets UNCHANGED - copy it verbatim. "
                "SUBSEQUENT CALLS (2nd, 3rd, etc.): Then you may refine based on search results - adjust content type, terminology, or scope as needed. "
            )
        },
        "doc_name": {
            "type": "string", 
            "description": (
                "The target document name to search within. "
                "IMPORTANT: Extract the EXACT document name from between < and > brackets in the user query. "
                "For example, if query mentions 'Document <annual_report_2023>', use 'annual_report_2023'. "
                "Do NOT include 'Document' prefix, angle brackets, or any other text - only the name itself."
            )
        },
        "excluded_pages": {
            "type": "array",
            "items": {"type": "integer"},
            "description": (
                "Optional list of page numbers to EXCLUDE from search results. "
                "Use this to avoid retrieving pages you've already seen in previous searches. "
                "For example: [1, 3, 5] will exclude pages 1, 3, and 5 from the results. "
                "If not provided or empty, no pages will be filtered."
            ),
            "nullable": True
        }
    }
    output_type = "any"
    
    def __init__(
        self,
        client: QdrantClient,
        embed_model: BaseEmbeddingModel,
        collection_name: str,
        embed_model_name: str,
        max_image_size: Optional[int] = None,
    ):
        """
        Initialize the search database tool.

        Args:
            client: Qdrant client instance
            embed_model: The embedding model instance
            collection_name: Name of the collection to search
            embed_model_name: Name of the embedding model for search
            max_image_size: If set, retrieved page images are downscaled so their
                longest edge is at most this many pixels (preserves aspect ratio).
                Useful for limiting VLM token budget on high-resolution scans.
        """
        super().__init__()
        self.client = client
        self.embed_model = embed_model
        self.collection_name = collection_name
        self.embed_model_name = embed_model_name
        self.max_image_size = max_image_size
    
    def forward(self, query: str, doc_name: str, excluded_pages: Optional[List[int]] = []) -> List:
        """
        Execute the database search.
        
        Args:
            query: The search query text
            doc_name: document name to filter results
            excluded_pages: Optional list of page numbers to exclude from results
            
        Returns:
            List of PIL Images showing the most relevant document pages
        """
        # Embed the query
        query_vector = self.embed_model.embed_text(query)
        
        # Build filter if doc_name is provided
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="document_name",
                    match=MatchValue(value=doc_name)
                )
            ],
            must_not=[
                FieldCondition(
                    key="page_num",
                    match=MatchAny(any=excluded_pages)
                )
            ],
        )
        
        # Search the database with a higher limit to account for filtering
        # We retrieve more results than needed and filter locally
        excluded_pages = excluded_pages or []
        
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector,
            using=self.embed_model_name,
            limit=3, # TODO: make this configurable
            with_payload=True,
            query_filter=query_filter
        )

        import PIL
        img_files = []
        result_pages = []
        for result_point in search_results.points:
            page_num = result_point.payload["page_num"]
                
            with PIL.Image.open(result_point.payload["full_img_path"]) as img_ref:
                img_ref.load()
                img_files.append(_resize_if_needed(img_ref, self.max_image_size))

            result_pages.append(page_num)

        return SearchResults(img_files, query, doc_name, result_pages)

class GetSpecificPagesTool(Tool):
    """
    A tool that retrieves specific page numbers from a document in the Qdrant database.
    
    This tool allows direct access to specific pages when you know which page numbers
    you want to examine, bypassing semantic search.
    """
    
    name = "get_specific_pages"
    description = """Retrieves specific page numbers from a document in the database.

Use this tool when:
- Specific page numbers are mentioned in user query
- The reflection tool or search results mention specific page numbers that need to be checked
- You want to verify information on particular pages
- You need to examine adjacent pages to ones already seen

Returns: The requested document page images.
"""
    
    inputs = {
        "doc_name": {
            "type": "string", 
            "description": (
                "The target document name. "
                "Extract the EXACT document name from between < and > brackets in the user query. "
                "For example, if query mentions 'Document <annual_report_2023>', use 'annual_report_2023'."
            )
        },
        "page_numbers": {
            "type": "array",
            "items": {"type": "integer"},
            "description": (
                "List of specific page numbers to retrieve from the document. "
                "For example: [1, 5, 10] will retrieve pages 1, 5, and 10. "
                "Maximum 5 pages per request to avoid token overload."
            )
        }
    }
    output_type = "any"
    
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        max_image_size: Optional[int] = None,
    ):
        """
        Initialize the get specific pages tool.

        Args:
            client: Qdrant client instance
            collection_name: Name of the collection to search
            max_image_size: If set, retrieved page images are downscaled so their
                longest edge is at most this many pixels.
        """
        super().__init__()
        self.client = client
        self.collection_name = collection_name
        self.max_image_size = max_image_size
    
    def forward(self, doc_name: str, page_numbers: List[int]) -> List:
        """
        Retrieve specific pages from the database.
        
        Args:
            doc_name: Document name to filter results
            page_numbers: List of page numbers to retrieve (max 5)
            
        Returns:
            List of PIL Images showing the requested document pages
        """
        # Limit to 5 pages to avoid token overload
        if len(page_numbers) > 5:
            page_numbers = page_numbers[:5]
        
        # Build filter for specific pages
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="document_name",
                    match=MatchValue(value=doc_name)
                ),
                FieldCondition(
                    key="page_num",
                    match=MatchAny(any=page_numbers)
                )
            ]
        )
        
        # Scroll through results to get all matching pages
        scroll_results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=len(page_numbers),
            with_payload=True
        )
        
        if len(scroll_results[0]) == 0:
            return f"Pages {page_numbers} not found in document '{doc_name}'. Try search_database tool instead."
        else:
            import PIL
            img_files = []
            result_pages = []
            
            for point in scroll_results[0]:
                page_num = point.payload["page_num"]
                
                with PIL.Image.open(point.payload["full_img_path"]) as img_ref:
                    img_ref.load()
                    img_files.append(_resize_if_needed(img_ref, self.max_image_size))

                result_pages.append(page_num)

            return SpecificPagesResults(img_files, doc_name, result_pages)

class SearchMemoryTool(Tool):
    """
    A tool that maintains search memory by tracking all search queries,
    pages visited, and information found across multiple searches.
    """
    
    name = "update_search_memory"
    description = """Records and maintains search memory across multiple searches.

**MANDATORY**: Call this tool IMMEDIATELY after EVERY "search_database" OR "get_specific_pages" call to update the search memory.

This tool:
1. Records the query, pages visited, and relevant information found
2. Maintains cumulative history of all searches
3. Tracks all pages that have been visited
4. Provides a simple confirmation of what was recorded

Returns: Confirmation message with search count and pages added.
"""
    
    inputs = {
        "query": {
            "type": "string",
            "description": "The exact query string that was just used in search_database"
        },
        "pages_visited": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "List of page numbers returned from the most recent search or page retrieval"
        },
        "relevant_information": {
            "type": "string",
            "description": (
                "Concise summary of information found on the returned pages that is RELEVANT to answering the original user question. "
                "Include specific facts, numbers, claims, or evidence. "
                "If pages contain no relevant information, state 'No relevant information found.'"
            )
        }
    }
    output_type = "string"

    def __init__(self):
        """Initialize the search memory tool with empty state.

        Note: `original_question` is NOT a tool input — the caller sets it
        directly on the instance via ``tool.original_question = question``
        before running the agent. Keeping it out of the schema prevents the
        teacher model from learning a spurious "memory tools always pass
        original_question" pattern that cross-contaminates other memory
        tools like `reflect_on_search` (see LESSONS_LEARNED §5 and §7).
        """
        super().__init__()
        self.search_history = []
        self.all_pages_visited = set()
        self.search_count = 0
        self.original_question = None

    def forward(
        self,
        query: str,
        pages_visited: List[int],
        relevant_information: str,
    ) -> str:
        """
        Record search results in memory.

        Args:
            query: The search query that was executed
            pages_visited: Page numbers returned from search
            relevant_information: Summary of relevant findings from these pages

        Returns:
            Search memory
        """
        # Record this search
        self.search_count += 1
        search_entry = {
            "search_number": self.search_count,
            "query": query,
            "pages": pages_visited,
            "findings": relevant_information
        }
        self.search_history.append(search_entry)
        self.all_pages_visited.update(pages_visited)
        
        pages_str = ", ".join(map(str, pages_visited))
        return f"✓ Search #{self.search_count} recorded: Query '{query}' → Pages [{pages_str}] → {len(pages_visited)} pages added to memory."
    
    def reset(self):
        """Reset the search memory (useful between different questions)."""
        self.search_history = []
        self.all_pages_visited = set()
        self.search_count = 0
        self.original_question = None

class ReflectionTool(Tool):
    """
    A tool that reflects on the accumulated search memory and provides
    reasoning about whether sufficient information has been gathered,
    along with recommendations for next actions.
    
    This tool reads from the search memory but does not modify it.
    """
    
    name = "reflect_on_search"
    description = """Analyzes the accumulated search memory and provides reasoning about next actions.

**USAGE**: Call this tool after updating search memory to get recommendations on whether to:
1. Provide a final answer (if sufficient information gathered)
2. Continue searching (if more information needed)
3. Provide "Not answerable" (if max 5 searches reached without sufficient information)

This tool examines all searches performed so far and recommends the optimal next step.

Returns: Detailed analysis of search progress and recommended next action.
"""
    
    inputs = {
        "is_sufficient": {
            "type": "boolean",
            "description": (
                "Your assessment: Is the cumulative information gathered so far SUFFICIENT to fully answer the original question? "
                "True = ready to provide final answer, False = need more searches"
            )
        },
        "missing_information": {
            "type": "string",
            "description": (
                "If is_sufficient=False, describe what specific information is still needed to answer the question. "
                "Be specific: mention if you need specific page numbers, concepts, or topics. "
                "If is_sufficient=True, set to 'None'"
            ),
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self, search_memory: SearchMemoryTool):
        """Initialize the reflection tool with reference to search memory."""
        super().__init__()
        self.search_memory = search_memory
    
    def forward(
        self,
        is_sufficient: bool,
        missing_information: Optional[str] = None
    ) -> str:
        """
        Reflect on search memory and provide next action recommendation.
        
        Args:
            is_sufficient: Whether enough information has been gathered
            missing_information: What information is still needed (if any)
            
        Returns:
            Formatted analysis of search progress and next action recommendation
        """
        # Build the reflection output
        output_lines = []
        output_lines.append("=" * 80)
        output_lines.append(f"REFLECTION ON SEARCH PROGRESS (Search {self.search_memory.search_count}/5)")
        output_lines.append("=" * 80)
        output_lines.append(f"\nOriginal Question: {self.search_memory.original_question}\n")
        
        # Summary of search memory
        output_lines.append("SEARCH MEMORY SUMMARY:")
        for entry in self.search_memory.search_history:
            pages_str = ", ".join(map(str, entry["pages"]))
            output_lines.append(f"\n{entry['search_number']}. Query: \"{entry['query']}\"")
            output_lines.append(f"   Pages: {pages_str}")
            output_lines.append(f"   Findings: {entry['findings']}")
        
        # Summary of all pages visited
        pages_list = sorted(list(self.search_memory.all_pages_visited))
        output_lines.append(f"\nALL PAGES VISITED: {pages_list}")
        output_lines.append(f"TOTAL UNIQUE PAGES: {len(pages_list)}")
        
        # Next action recommendation
        output_lines.append("\n" + "=" * 80)
        output_lines.append("RECOMMENDED NEXT ACTION:")
        output_lines.append("=" * 80)
        
        if is_sufficient:
            output_lines.append("\n✓ ASSESSMENT: SUFFICIENT INFORMATION GATHERED")
            output_lines.append("→ RECOMMENDATION: Provide final_answer based on accumulated findings above")
            output_lines.append("→ Synthesize all relevant information from the search memory into a complete answer")
        elif self.search_memory.search_count >= 5:
            output_lines.append("\n⚠ MAXIMUM SEARCHES REACHED (5/5)")
            output_lines.append("→ RECOMMENDATION: Provide final_answer with 'Not answerable.' if insufficient information to fully answer the question")
        else:
            output_lines.append(f"\n⊕ ASSESSMENT: CONTINUE SEARCHING ({self.search_memory.search_count}/5 searches completed)")
            output_lines.append(f"→ MISSING INFORMATION: {missing_information}")
            output_lines.append("→ RECOMMENDATION: call search_database or get_specific_pages based on what is needed")
            output_lines.append("→ After retrieval, call update_search_memory and reflect_on_search again")
        
        output_lines.append("=" * 80)
        
        return "\n".join(output_lines)
    
class SearchResults(AgentImageList):
    """Page-image retrieval result. MUST extend AgentImageList so the
    customized ``process_tool_calls`` in src/smolagents/agents.py routes
    ``self._raw_list`` into ``memory_step.observations_images`` — otherwise
    the VLM only sees the text summary and hallucinates findings (14%
    accuracy regression on longdocurl teacher eval; see LESSONS_LEARNED)."""

    def __init__(self, value, query, doc_name, pages):
        AgentImageList.__init__(self, value)

        self._query = query
        self._doc_name = doc_name
        self._pages = pages

    def to_string(self):
        pages_str = ", ".join(map(str, self._pages))
        return f"Search results for '{self._query}' in '{self._doc_name}': Found relevant content on pages {pages_str}."

class SpecificPagesResults(AgentImageList):
    """Direct-page retrieval result. Same AgentImageList requirement as SearchResults."""

    def __init__(self, value, doc_name, pages):
        AgentImageList.__init__(self, value)
        self._doc_name = doc_name
        self._pages = pages

    def to_string(self):
        pages_str = ", ".join(map(str, self._pages))
        return f"Retrieved specific pages from '{self._doc_name}': {pages_str}."