import re
import math
import json
import string
from collections import Counter

def lower(text):
    return text.lower()

def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

def white_space_fix(text):
    return ' '.join(text.split())

def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)

def normalize_answer(s):
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def parse_output(output, prefix="Answer:"):
    def lstrip_string(s, sub):
        return re.sub(f'^{re.escape(sub)}', '', s, flags=re.IGNORECASE)
    patterns = [re.compile(f"(?:{prefix})(.*)(?:\n|$)", flags=re.IGNORECASE),  # prefix + answer + sentence end
                re.compile(r"(?:^)(.*)(?:\n|$)")] # the beginning + answer + sentence end
    for pat in patterns:
        matches = pat.search(output)
        if matches is not None:
            return lstrip_string(matches[1].strip(), prefix).strip() # 0 index includes the non-capturing group # lstrip again because for chat models sometimes it will repeat the prefix
    # if still not found, return None, but should actually never get this case...
    return None

def get_docqa_clean_string(s):
    s = str(s).lower().strip()
    s = s.replace(",", "")

    suffix_list = ["kg", "meters", "acres", "minutes", "miles", "mile",
                   "feet",
                   "million", "thousand", "billion", "mm", "m"]

    for suffix in suffix_list:
        s = re.sub(re.escape(suffix) + r'$', '', s).strip()

    # remove parenthesis
    # s = re.sub(r'\s*\([^)]*\)', "", s).strip()
    # remove quotes
    s = re.sub(r"^['\"]|['\"]$", "", s).strip()
    s = s.strip().strip("$").strip()
    s = s.strip().strip("£").strip()
    s = s.strip().strip("%").strip()
    return s


def is_float_equal(reference, prediction, include_percentage=False) -> bool:
    if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    for item in gt_result:
        if math.isclose(item, prediction, rel_tol=0.01):
            return True
    return False


def need_exact_match_check(s):
    patterns = [
        r'https://',
        r'.*\.(py|ipynb)$',
        r'^page',
        r'^\d+(-\d+|\s\d+)?$',
        r'(a\.m\.|p\.m\.)',
        r'^\d{4}[-/\s]\d{1,2}[-/\s]\d{1,2}$',  # YYYY-MM-DD, YYYY/MM/DD
        r'^\d{1,2}[-/\s]\d{1,2}[-/\s]\d{2,4}$',  # DD-MM-YYYY, MM/DD/YYYY
        r'^\d{4}[-/\s]\d{1,2}$',  # YYYY-MM, YYYY/MM
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    ]

    return any(re.search(pattern, s) for pattern in patterns)

def extract_number_list(pred):
    pred_clean = pred.replace(',', '')
    num_pattern = r'(-?\d+(\.\d*)?|-?\.\d+)' # r'-?(\d+(\.\d*)?|\.\d+)'
    matches = re.findall(num_pattern, pred_clean)
    numbers = []
    for match_tuple in matches:
        match = match_tuple[0]
        try: # TODO filter inf number. Note this is added at the end of the experiment. previous is numers.append(float(match))
            num = float(match)
            if math.isfinite(num):
                numbers.append(num)
        except ValueError:
            continue
    return numbers

def get_str_type(num_str):
    try:
        num = float(get_docqa_clean_string(num_str))
        if num == int(num) and "%" not in num_str:
            return "Integer"
        else:
            return "Float"
    except:
        return "String"

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def calculate_metrics(prediction, answers, metrics):
    metric_list = [m.strip() for m in metrics.split(",")]
    metric_res = {}
    if "doc_qa" in metric_list:
        answer, answer_type = answers
        metric_res["doc_qa"] = eval_docqa_score(answer, prediction, answer_type)

    return metric_res

def eval_docqa_score(gt, pred, answer_type):
    if answer_type == "Integer":
        gt = float(get_docqa_clean_string(str(gt)))
        assert int(gt) == gt
        gt = int(gt)
        pred = get_docqa_clean_string(str(pred))
        pred_num_list = [int(num) for num in extract_number_list(pred) if int(num) == num]
        score = any(gt == pred_num for pred_num in pred_num_list)
    elif answer_type == "Float":
        gt = float(get_docqa_clean_string(str(gt)))
        pred = get_docqa_clean_string(str(pred))
        pred_num_list = extract_number_list(pred)
        score = any(is_float_equal(gt, pred_num, include_percentage=True)
                    for pred_num in pred_num_list)
    elif answer_type in ["String", "None"]:
        if need_exact_match_check(gt):
            score = gt in pred
        else:
            score = f1_score(pred, gt)[0]
    elif answer_type == "List":
        gt_list = json.loads(gt)
        # merge f1 score text to prevent low precision
        merge_flag = [True if isinstance(item, str) and get_str_type(item) == "String" and not
                              need_exact_match_check(item)
                      else False for item in gt_list] # we merge all answers that are string and don't need EM for better recall
        merged_str = " ".join([item for item, m_flag in zip(gt_list, merge_flag) if m_flag]).strip()
        if merged_str:
            new_gt_list = [merged_str] + [item for item, m_flag in zip(gt_list, merge_flag) if not m_flag]
            gt_list = new_gt_list

        gt_score_list = [] # This is the greedy score similar to that used in LongDocURL
        for gt in gt_list:
            assert not isinstance(gt, list)
            if isinstance(gt, int):
                gt_type = "Integer"
            elif isinstance(gt, float):
                gt_type = "Float"
            else: # String answers can also represent int and float
                gt_type = get_str_type(gt)
            gt_score = eval_docqa_score(gt, pred, gt_type)
            gt_score_list.append(gt_score)
        score = sum(gt_score_list) / len(gt_list)
    else:
        raise KeyError("Wrong answer type:", answer_type)
    return float(score)

def evaluation(prediction, example):
    """
    Returns: metrics (dict) and additional info to update the original sample with (dict)
    """
    answer = example["answer"]
    answer_type = example["answer_format"]

    parsed_pred = parse_output(prediction)
    if parsed_pred is None:
        parsed_pred = prediction

    mets = eval_docqa_score(answer, parsed_pred, answer_type)
    return mets