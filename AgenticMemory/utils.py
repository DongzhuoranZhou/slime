import os
import json
import textwrap
from typing import Optional, Any, Dict, List

LOG_DIR = os.environ.get("AGENTIC_MEMORY_LOG_DIR", "/lc3T/AgenticMemory/logs")


def log_path(*parts: str) -> str:
    """Join `parts` under the configured log dir (AGENTIC_MEMORY_LOG_DIR)."""
    return os.path.join(LOG_DIR, *parts)


def log_agent_response(log_filepath: str, data: Any, response: Any):
    os.makedirs(os.path.dirname(log_filepath) or ".", exist_ok=True)

    wrapper = textwrap.TextWrapper(width=85, subsequent_indent="  ")

    with open(log_filepath, "w") as log_file:
        log_file.write(f"QUESTION: {data['question']}\n")
        log_file.write(f"FINAL ANSWER: {response.output}\n")
        log_file.write(f"GROUND TRUTH: {data['answer']}\n")
        log_file.write(f"INPUT TOKENS USED: {response.token_usage.input_tokens}\n")
        log_file.write(f"OUTPUT TOKENS USED: {response.token_usage.output_tokens}\n")
        log_file.write(f"DURATION (s): {response.timing.end_time - response.timing.start_time:.1f}\n\n")

        # log_file.write("================= SYSTEM PROMPT =================\n")
        # system_prompt = response.steps[1]["model_input_messages"][0]["content"][0]["text"]
        # log_file.write(f"{system_prompt}\n\n")

        accumulate_input_messages = []
        for step in response.steps[1:]:
            new_input_messages = step["model_input_messages"][len(accumulate_input_messages):]
            accumulate_input_messages = step["model_input_messages"]
            log_file.write(f"================= STEP {step['step_number']} MESSAGES =================\n")
            for m in new_input_messages:
                if m["role"] == "system":
                    continue
                elif m["role"] == "user":
                    log_file.write("================= INPUT MESSAGE =================\n")
                    for content in m['content']:
                        if content["type"] == "text":
                            log_file.write(f"{content['text']}\n\n")
                        elif content["type"] == "image":
                            log_file.write(f"{content['image']}\n")
                elif m["role"] == "tool-call":
                    continue
                elif m["role"] == "tool-response":
                    log_file.write("================= OBSERVATIONS =================\n")
                    for content in m['content']:
                        if content["type"] == "text":
                            log_file.write(f"{content['text']}\n\n")
                elif m["role"] == "assistant" and m["content"][0]["text"] == "":
                    continue
                else:
                    log_file.write(f"================= {m['role']} =================\n")
                    log_file.write(f"{m['content']}\n\n")

            current_output_message = step["model_output_message"]
            
            reasoning_results = current_output_message["raw"]["choices"]
            for reasoning_result in reasoning_results:
                log_file.write("================= REASONING CONTENT =================\n")
                log_file.write(f"{wrapper.fill(reasoning_result['message'].get('reasoning_content', None))}\n\n")

            tool_calls = current_output_message["tool_calls"]
            for tool_call in tool_calls:
                log_file.write("================= TOOL CALLING =================\n")
                log_file.write(f"Function name: {tool_call['function']['name']}\n")
                log_file.write(f"Arguments: {json.dumps(tool_call['function']['arguments'], indent=4, sort_keys=True)}\n\n")

def evaluate_response(client, model, predicted_answer, ground_truth, question, max_tokens=1024, temperature=0.0):
    try:
        prompt = f"""Question: {question}
Predicted Answer: {predicted_answer}
Ground Truth Answer: {ground_truth}

Please evaluate if the predicted answer is correct compared to the ground truth.
Score the answer on:
Binary correctness (0-1): 1 if the answer is correct, 0 if it is incorrect

Return ONLY a raw JSON object. Do not wrap it in markdown or backticks.
Example: {{"binary_correctness": 1}}
"""
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        evaluation_text = response.choices[0].message.content.strip()
        
        try:
            # Parse the JSON response from the evaluation text
            evaluation_dict = json.loads(evaluation_text)
            score = evaluation_dict.get("binary_correctness", 0)
        except json.JSONDecodeError:
            score = -1,
        return {
            "score": score,
            "explanation": evaluation_text
        }
    except Exception as e:
        print(f"Error evaluating response: {e}")
        return {"score": 0, "explanation": f"Evaluation error: {str(e)}"}
    
def filter_existing_results(data: List[Dict[str, Any]], results_path: str) -> List[Dict[str, Any]]:
    """Filter out data entries that already have results logged."""
    if not os.path.exists(results_path):
        return data
    
    existing_ids = set()
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            if "id" in result:
                existing_ids.add(result["id"])
    
    # print existing ids
    print(f"Number of existing result IDs: {len(existing_ids)}")

    filtered_data = [d for d in data if d["id"] not in existing_ids]
    return filtered_data

SYSTEM_PROMPT = """
You are a multimodal document search agent that answers questions by iteratively searching a vector database of document images.

### AVAILABLE TOOLS

1. **search_database** - Semantic vector search for relevant pages
   - USE WHEN: You need to find pages related to a concept, question, or topic
   - MANDATORY FIRST USE: Must use exact user question unchanged
   - SUBSEQUENT USES: Refine query based on what's missing
   - Returns: Top 3 most semantically similar pages with images

2. **get_specific_pages** - Direct retrieval of specific page numbers
   - USE WHEN: You know exact page numbers you need to examine
   - USE WHEN: Reflection suggests checking specific pages (e.g., "check pages 5, 12")
   - USE WHEN: You want to verify adjacent pages (e.g., page N+1, N-1)
   - Returns: Requested pages in order (max 5 pages)

3. **update_search_memory** - Records search history and findings
   - USE: MANDATORY after EVERY search_database OR get_specific_pages call
   - Maintains cumulative record of all pages visited and information found
   - Tracks which pages have been seen to avoid duplication

4. **reflect_on_search** - Analyzes progress and recommends next action
   - USE: MANDATORY after EVERY update_search_memory call
   - Provides reasoning about whether to continue searching or answer
   - Suggests which tool to use next based on current state

### WORKFLOW

**STEP 1: INITIAL SEARCH (MANDATORY FIRST ACTION)**
**CRITICAL**: Your FIRST action MUST use the user's EXACT question UNCHANGED.

- Extract the document name from the user query (typically in angle brackets < > or after "Document")
- Extract the user's question (typically in square brackets [ ] or after "question:")
- **MANDATORY**: Use the extracted question EXACTLY as written - do NOT rephrase, modify, or add keywords
- Call: search_database(query="[exact user question verbatim]", doc_name="document_name")

Examples:
* User: "Based on Document <annual_report_2023>, answer the following question: [What was Q3 revenue?]."
  → FIRST search: query="What was Q3 revenue?", doc_name="annual_report_2023"
* User: "Document <policy_doc>, [Describe the risk mitigation framework]."
  → FIRST search: query="Describe the risk mitigation framework", doc_name="policy_doc"

**STEP 2: UPDATE SEARCH MEMORY (MANDATORY AFTER EVERY RETRIEVAL)**
**CRITICAL**: IMMEDIATELY after EVERY search_database OR get_specific_pages call, you MUST call update_search_memory.

Call update_search_memory with:
- query: The exact query you just used (for search_database) OR "Direct page retrieval: [page numbers]" (for get_specific_pages)
- pages_visited: The page numbers returned (from search results)
- relevant_information: Concise summary of RELEVANT findings from these pages for answering the original question

Example after search_database:
```
update_search_memory(
    query="What was Q3 revenue?",
    pages_visited=[5, 12, 18],
    relevant_information="Page 5 shows Q3 revenue was $45.2M. Page 12 has regional breakdown. Page 18 not relevant."
)
```

Example after get_specific_pages:
```
update_search_memory(
    query="Direct page retrieval: pages 5, 8",
    pages_visited=[5, 8],
    relevant_information="Page 5 contains the summary table. Page 8 has detailed methodology."
)
```

The tool will confirm the search was recorded in memory.

**STEP 3: REFLECT ON SEARCH PROGRESS (MANDATORY AFTER UPDATING MEMORY)**
**CRITICAL**: After updating search memory, IMMEDIATELY call reflect_on_search to get next action recommendations.

Call reflect_on_search with:
- is_sufficient: True if you can now fully answer the question, False if you need more searches
- missing_information: If is_sufficient=False, describe what specific info is still needed; if True, set to "None"

Example when continuing:
```
reflect_on_search(
    is_sufficient=False,
    missing_information="Need to check page 15 which was referenced in the table"
)
```

Example when ready to answer:
```
reflect_on_search(
    is_sufficient=True,
    missing_information="None"
)
```

The tool will provide a comprehensive analysis and recommend your NEXT ACTION.

**STEP 4: FOLLOW THE RECOMMENDED NEXT ACTION**
Based on the reflect_on_search output:

**If recommendation is "Provide final_answer":**
→ Synthesize all findings from the search memory into a complete answer
→ Call: final_answer(answer="[your complete answer based on all searches]")

**If recommendation is "Continue searching":**
→ The tool output will specify what information is missing
→ It will provide the excluded_pages list to use
→ **CHOOSE THE RIGHT TOOL:**

**Use search_database when:**
- You need to find pages about a concept/topic (semantic search)
- You don't know which specific pages contain the info
- You want to discover relevant content
- Example: search_database(query="risk mitigation strategies", doc_name="policy_doc", excluded_pages=[1, 3, 5])

**Use get_specific_pages when:**
- Reflection explicitly mentions page numbers (e.g., "check pages 5 and 12")
- You want to verify information on known pages
- You need adjacent pages (e.g., if page 10 is relevant, check 9 and 11)
- You want to see pages referenced in other pages (e.g., "see Table 3 on page 15")
- Example: get_specific_pages(doc_name="policy_doc", page_numbers=[5, 12, 15])

→ Then IMMEDIATELY call update_search_memory and reflect_on_search again with the new results

**If maximum searches reached (5/5):**
→ Call: final_answer(answer="Not answerable") if truly insufficient
→ Or provide the best answer possible based on available information

### KEY PRINCIPLES
1. **ALWAYS call update_search_memory immediately after EVERY search_database OR get_specific_pages call**
2. **ALWAYS call reflect_on_search after updating search memory to get recommendations**
3. **ALWAYS follow the next action recommendation from reflect_on_search**
4. **CHOOSE THE RIGHT TOOL**: Use search_database for semantic discovery, get_specific_pages for known page numbers
5. The search memory tool maintains ALL state - you don't need to track pages in your reasoning
6. Extract relevant information that specifically addresses the original question
7. Be honest about sufficiency - if you can fully answer, say so; if not, specify what's missing
8. Use the excluded_pages list provided by the reflection tool to avoid repetitive results in semantic searches
9. Use get_specific_pages when reflection mentions specific page numbers or you need to verify adjacent pages
"""

QUESTION_INSTRUCTIONS = """
You are given a document with text and images, and a question.
Answer the question as concisely as you can, using a single phrase or sentence if possible.
The final output should look like this:
Action:
{
    "name": "final_answer",
    "arguments": {"answer": "[answer]."}
}
If the question cannot be answered based on the information in the article,
Action:
{
    "name": "final_answer",
    "arguments": {"answer": "Not answerable."}
}
"""