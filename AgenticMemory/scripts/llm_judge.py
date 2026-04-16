#%%
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

def get_result_path(config):
    base_dir = Path("/workspace/VLM-Memory/results")
    result_dir = base_dir / config["retrieve"] / config["embed_model"] / config["vlm_model"]

    for json_path in result_dir.glob("*.json"):
        if f"K128_N{config['K']}_" in json_path.name and f"{config['dataset']}_" in json_path.name:
            return str(json_path)
    
    raise FileNotFoundError("No matching result file found.")

# evaluation script from SimpleDoc
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

if __name__ == "__main__":
    sample_num = 50

    config = {
        "retrieve": "RAG",
        "K": 10,
        "embed_model": "jina_multi",
        "vlm_model": "gemini-3-pro-preview",
        "dataset": "mmlongdoc"  # mmlongdoc, longdocurl
    }

    model_id="gpt-4o-2024-11-20"

    load_dotenv()
    API_KEY = os.environ.get("API_KEY")
    client = OpenAI(
        api_key=API_KEY, 
        base_url="https://api-gateway.glm.ai/v1"
    )

    rag_paths = [
        "/workspace/VLM-Memory/results/RAG/ground_truth/gemini-3-pro-preview/longdocurl_longdocurl_in131072_sizeNone_sampFalsemax16000min0t1.0p1.0_chatTrue_42.json",
        "/workspace/VLM-Memory/results/RAG/ground_truth/gemini-3-pro-preview/mmlongdoc_mmlongdoc_in131072_sizeNone_sampFalsemax16000min0t1.0p1.0_chatTrue_42.json"
    ]
    for dataset in ["longdocurl", "mmlongdoc"]:
        if dataset == "longdocurl":
            rag_path = rag_paths[0]
        else:
            rag_path = rag_paths[1]
            
        with open(rag_path, "r") as f:
            rag_results = json.load(f)

        llm_scores = []
        for result in tqdm(rag_results["data"]):
            question = result["question"]
            ground_truth = result["answer"]
            predicted_answer = result["output"]
            eval_result = evaluate_response(client, 
                            model_id, 
                            predicted_answer, 
                            ground_truth, 
                            question, 
                            max_tokens=1024, 
                            temperature=0.0)
            eval_result["id"] = result["id"]
            llm_scores.append(eval_result)
        
        output_path = f"results/gemini3pro_gt_eval_results_{dataset}.json"
        with open(output_path, "w") as f:
            json.dump(llm_scores, f, indent=4)

    # agent_result_path = "/workspace/smolagent_dev/results/top50_run.jsonl"
    # with open(agent_result_path, "r") as f:
    #     agent_results = [json.loads(line) for line in f.readlines()]

    # combined_results = []
    # for rag_result, agent_result in tqdm(
    #     zip(rag_results["data"][:sample_num], agent_results[:sample_num]), 
    #     total=sample_num
    #     ):
    #     question = rag_result["question"]
    #     ground_truth = rag_result["answer"]
    #     predicted_answer = rag_result["parsed_output"]
    #     rule_based_score = rag_result["doc_qa"]
    #     eval_rag_result = evaluate_response(client, 
    #                     model_id, 
    #                     predicted_answer, 
    #                     ground_truth, 
    #                     question, 
    #                     max_tokens=1024, 
    #                     temperature=0.0)
        
    #     question = agent_result["data"]["question"]
    #     ground_truth = agent_result["data"]["answer"]
    #     predicted_answer = agent_result["output"]
    #     rule_based_score = agent_result["acc"]
    #     eval_agent_result = evaluate_response(client, 
    #                     model_id, 
    #                     predicted_answer, 
    #                     ground_truth, 
    #                     question, 
    #                     max_tokens=1024, 
    #                     temperature=0.0)
        
    #     # save all results in dict
    #     combined_result = {
    #         "question": question,
    #         "ground_truth": ground_truth,
    #         "rag": {
    #             "predicted_answer": rag_result["parsed_output"],
    #             "rule_based_score": rag_result["doc_qa"],
    #             "llm_score": eval_rag_result
    #         },
    #         "agent": {
    #             "predicted_answer": agent_result["output"],
    #             "rule_based_score": agent_result["acc"],
    #             "llm_score": eval_agent_result
    #         }
    #     }

    #     combined_results.append(combined_result)

    # # calculate the average rule based score and llm score
    # total_rag_rule = sum([res["rag"]["rule_based_score"] for res in combined_results])
    # total_rag_llm = sum([res["rag"]["llm_score"]["score"] for res in combined_results])
    # total_agent_rule = sum([res["agent"]["rule_based_score"] for res in combined_results])
    # total_agent_llm = sum([res["agent"]["llm_score"]["score"] for res in combined_results])
    # print("---- Final Evaluation Results ----")
    # print(f"RAG - Rule Based Accuracy over {len(combined_results)} samples: {total_rag_rule/len(combined_results):.4f}")
    # print(f"RAG - LLM Based Accuracy over {len(combined_results)} samples: {total_rag_llm/len(combined_results):.4f}")
    # print(f"Agent - Rule Based Accuracy over {len(combined_results)} samples: {total_agent_rule/len(combined_results):.4f}")
    # print(f"Agent - LLM Based Accuracy over {len(combined_results)} samples: {total_agent_llm/len(combined_results):.4f}")

    # # save results
    # output_path = "results/combined_evaluation_results.json"
    # with open(output_path, "w") as f:
    #     json.dump(combined_results, f, indent=4)

# %%
# load json file
import json
import pprint
longdocurl_result_path = "/workspace/smolagent_dev/results/gemini3pro_gt_eval_results_longdocurl.json"
mmlongdoc_result_path = "/workspace/smolagent_dev/results/gemini3pro_gt_eval_results_mmlongdoc.json"

with open(longdocurl_result_path, "r") as f:
    longdocurl_data = json.load(f)
with open(mmlongdoc_result_path, "r") as f:
    mmlongdoc_data = json.load(f)

longdocurl_scores = [item["score"] for item in longdocurl_data]
mmlongdoc_scores = [item["score"] for item in mmlongdoc_data]
print(f"LongDocURL LLM Evaluation Accuracy over {len(longdocurl_scores)} samples: {sum(longdocurl_scores)/len(longdocurl_scores):.4f}")
print(f"MMLongDoc LLM Evaluation Accuracy over {len(mmlongdoc_scores)} samples: {sum(mmlongdoc_scores)/len(mmlongdoc_scores):.4f}")
# %%
