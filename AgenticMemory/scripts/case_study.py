#%%
import json
import pprint
agent_result = "/workspace/smolagent_dev/results/top50_run_V1.jsonl"
llm_judge_path = "/workspace/smolagent_dev/results/combined_evaluation_results.json"

with open(agent_result, "r") as f:
    data = [json.loads(line) for line in f.readlines()]

with open(llm_judge_path, "r") as f:
    llm_judge_data = json.load(f)

data_index = 48

print(f"QUESTION: {data[data_index]['data']['question']}")
print(f"EXPECTED ANSWER: {data[data_index]['data']['answer']}")
print(f"GROUND TRUTH PAGES: {data[data_index]['data']['ans_page_list']}\n")

print(f"RAG OUTPUT: {llm_judge_data[data_index]['rag']['predicted_answer']}")
print(f"RAG SCORE: {llm_judge_data[data_index]['rag']['llm_score']}\n")

for step in range(len(data[data_index]["steps"])):
    try:
        print(f"Step {step}: {data[data_index]['steps'][step]['tool_calls'][0]['function']['name']}")
    except:
        pass

print(f"AGENT OUTPUT: {data[data_index]['output']}")
print(f"AGENT SCORE: {llm_judge_data[data_index]['agent']['llm_score']}")
print("\n")

for step in range(len(data[data_index]["steps"])):
    if step != 0:
        print(f"================= STEP {step} =================")
        current_step = data[data_index]["steps"][step]

        # print("---------- INPUT MESSAGE ----------")
        # pprint.pprint(current_step["model_input_messages"][1:])
        # print("\n")

        try:
            reasoning_content = current_step["model_output_message"]["raw"]["choices"][0]["message"]["reasoning_content"]
            print("---------- REASONING CONTENT ----------")
            pprint.pprint(current_step["model_output_message"]["raw"]["choices"][0]["message"]["reasoning_content"])
            print("\n")
        except:
            print("---------- REASONING CONTENT ----------")
            print("No reasoning content available.")
            print("\n")

        print("---------- TOOL CALLS ----------")
        pprint.pprint(current_step["tool_calls"])
        print("\n")

        print("---------- OBSERVATIONS ----------")
        pprint.pprint(current_step["observations"])
        pprint.pprint(current_step["observations_images"])
        print("\n\n")

#%%
import json
import pprint
agent_result = "/workspace/smolagent_dev/results/top50_run_V1.jsonl"
llm_judge_path = "/workspace/smolagent_dev/results/combined_evaluation_results.json"

with open(agent_result, "r") as f:
    data = [json.loads(line) for line in f.readlines()]

with open(llm_judge_path, "r") as f:
    llm_judge_data = json.load(f)

data_index = 2

print(f"QUESTION: {data[data_index]['data']['question']}")
print(f"EXPECTED ANSWER: {data[data_index]['data']['answer']}")
print(f"GROUND TRUTH PAGES: {data[data_index]['data']['ans_page_list']}\n")

print(f"RAG OUTPUT: {llm_judge_data[data_index]['rag']['predicted_answer']}")
print(f"RAG SCORE: {llm_judge_data[data_index]['rag']['llm_score']}\n")

accumulate_input_messages = []
for step in data[data_index]["steps"][1:]:
    new_input_messages = step["model_input_messages"][len(accumulate_input_messages):]
    accumulate_input_messages = step["model_input_messages"]
    print(f"================= STEP {step['step_number']} MESSAGES =================")
    for m in new_input_messages:
        if m["role"] == "system":
            pprint.pprint("================= SYSTEM PROMPT =================")
            for content in m['content']:
                pprint.pprint(content['text'])
        elif m["role"] == "user":
            pprint.pprint("================= INPUT MESSAGE =================")
            for content in m['content']:
                if content["type"] == "text":
                    pprint.pprint(content['text'])
                elif content["type"] == "image":
                    pprint.pprint(content['image'])
        elif m["role"] == "tool-call":
            # pprint.pprint(f"================= TOOL CALLING =================")
            # pprint.pprint("TOOL CALL MESSAGE:")
            # for content in m['content']:
            #     if content["type"] == "text":
            #         pprint.pprint(content['text'])
            continue
        elif m["role"] == "tool-response":
            pprint.pprint("================= OBSERVATIONS =================")
            for content in m['content']:
                if content["type"] == "text":
                    pprint.pprint(content['text'])
        elif m["role"] == "assistant" and m["content"][0]["text"] == "":
            continue
        else:
            pprint.pprint(f"================= {m['role']} =================")
            pprint.pprint(f"{m['content']}")
    
    current_output_message = step["model_output_message"]
    
    reasoning_results = current_output_message["raw"]["choices"]
    for reasoning_result in reasoning_results:
        pprint.pprint("================= REASONING CONTENT =================")
        pprint.pprint(reasoning_result["message"].get("reasoning_content", None))
    
    planed_tool_calls = current_output_message["tool_calls"]
    for planed_tool_call in planed_tool_calls:
        pprint.pprint("================= TOOL CALLING =================")
        pprint.pprint(f"Function name: {planed_tool_call['function']['name']}")
        pprint.pprint(f"Arguments: {planed_tool_call['function']['arguments']}")

    print("\n\n")

# %%
