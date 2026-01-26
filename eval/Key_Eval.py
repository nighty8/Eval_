import json
import re
import importlib.util
import os
import argparse
import random
import time
from datetime import datetime
from tqdm import tqdm
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.parser import *
from utils.data_loader import load_data
from utils.math_normalization import *
from utils.grader import *
import pickle
from math import comb
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_jsonl(
    file_path,
    input="input",
    output="output",
    answer="answer",
    big_model= None,
    all_token=None,
    is_gzip=False,
    code = 'code',
    gt = 'gt',
    pred = 'pred',
    chosen_entropy = 'chosen_entropy',
    matches = 'matches'
):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    n = 0 
    with open(file_path, "r",encoding="utf-8") as f:
        for line in f:
            n += 1
            item = json.loads(line)
            new_item = dict(
                input=item[input] if input in item else None,
                output=item[output] if output in item else None,
                answer= item[answer]   if answer in item else None,
                big_model= item[big_model]   if big_model in item else None,
                all_token= item[all_token]   if all_token in item else None,
                code= item[code][0]   if code in item else None,
                gt = item[gt]  if gt in item else None,
                chosen_entropy = item[chosen_entropy]  if chosen_entropy in item else None,
                matches = item[matches]  if matches in item else None,
                pred = item[pred][0]  if pred in item else None,
            )
            item = new_item
            list_data_dict.append(item)
    return list_data_dict



import re
INVALID_ANS = "[invalid]"
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def key_eval(data_path,deepseek_tokenize=None,output_dir=None):
    answers = []
    wrong_list,correct_list = [],[]
    total_len = 0
    a,b = 0,0
    correct_cnt,total_cnt = 0,0

    total_big_model = 0

    total_forward_time = 0
    # list_data_dict = load_jsonl(data_path, input="input", output="output",answer='answer',big_model= 'big_model',all_token='all_token')
    list_data_dict = load_jsonl(data_path, input="input", output="output",answer='answer')

    for sample in tqdm(list_data_dict):
        # gt_ans = sample['gt']
        # gt_ans = sample['answer']
        gt_ans = sample['output']

        minus = '<|im_start|>user' + sample['input'] + 'Please reason step-by-step and put your choice letter without any other text with \\boxed{} in the end.'

        # gt_cot, gt_ans = parse_ground_truth(true_answer, "math")
        # print(gt_ans)

        # single_sub_len = len(sample['big_model'])
        # single_total_len = sample['all_token']

        # gt_ans = extract_answer_from_output(sample['output'])
        # print(gt_ans)
        
        # model_answer = extract_answer(sample["code"],"olympia")
        # print(model_answer)
        # model_answer = sample["pred"]
        # print(model_answer)
        # print(gt_ans)
        model_answer = extract_answer(sample['answer'],"olympia")

        matches = sample['matches']
        # total_forward_time += len(matches)

        if deepseek_tokenize:
            single_len = len(deepseek_tokenize.encode(sample['answer'].replace(minus, '', 1)))
            # single_len = len(deepseek_tokenize.encode(sample['code']))
            total_len += single_len

        # total_big_model += len(sample['big_model'])

        # print(model_answer)
        is_correct_list = [check_is_correct(model_answer, gt_ans)]
        # print(gt_ans)
        # print(model_answer)
        # print(is_correct_list)
        # print('-'*56)
        is_correct = any(is_correct_list)
        # print(is_correct)
        if is_correct:
            correct_cnt += 1
            answers.append(True)
            correct_list.append(total_cnt)
        else:
            answers.append(False)
            wrong_list.append(total_cnt)
        total_cnt += 1
    print(total_len)
    print(total_len / total_cnt)
    print(total_forward_time)
        # a += single_sub_len
        # b += single_total_len
    return correct_cnt,total_cnt,correct_list
    # print(wrong_list)
    # print(total_len / total_cnt)
    # print(a/b)

    # os.makedirs(output_dir, exist_ok=True)
    # with open(os.path.join(output_dir, "results.txt"), "w") as f:
    #     for answer in answers:
    #         print(answer, file=f)

def majority_voting(data_path,deepseek_tokenize=None,output_dir=None):
    answers = []
    wrong_list, correct_list = [], []
    total_len = 0
    correct_cnt, total_cnt = 0, 0

    # 16
    list_data_dicts = []
    for i in range(1, 17):
        path_i = f"{data_path}{i}{'.json'}"
        list_data_dicts.append(
            load_jsonl(path_i, input="input", output="output", answer='answer')
        )

    for idx in tqdm(range(len(list_data_dicts[0]))):
        gt_ans = list_data_dicts[0][idx]['output']

        gen_answers = []
        for d in list_data_dicts:
            ans = extract_answer(d[idx]['answer'], "gpqa")
            gen_answers.append(ans)

            if deepseek_tokenize:
                single_len = len(deepseek_tokenize.encode(d[idx]['answer'].replace("-", '', 1)))
                total_len += single_len

        majority_ans, _ = Counter(gen_answers).most_common(1)[0]

        is_correct = check_is_correct(majority_ans, gt_ans)
        if is_correct:
            correct_cnt += 1
            answers.append(True)
            correct_list.append(total_cnt)
        else:
            answers.append(False)
            wrong_list.append(total_cnt)

        total_cnt += 1

    print(total_len)
    return correct_cnt, total_cnt, correct_list

def JsonLoad(data_path):
    with open(data_path,'r',encoding='utf-8') as json_file:
        data = [json.loads(line) for line in json_file]
    return data

from collections import defaultdict, Counter
def AnotherEval(data_path):
    data = JsonLoad(data_path)
    question_answers = defaultdict(list)
    golden_truth = {}

    for item in data:
        qid = item['question_mark']
        golden_answer = item['output']
        answer = extract_answer(item['answer'],'minerva')
        question_answers[qid].append(answer)
        golden_truth[qid] = golden_answer

    most_common_answers = {}
    for qid, answers in question_answers.items():
        most_common_answer_x = Counter(answers).most_common(1)[0][0]
        most_common_answers[qid] = most_common_answer_x

    
    same_count = sum(1 for k in most_common_answers if check_is_correct(most_common_answers[k],golden_truth.get(k)))
    total = len(most_common_answers)  
    similarity = same_count / total
    print(similarity)
    print(most_common_answers)
    print(golden_truth)

def Draw(list1,list2,save_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 6))
    sns.kdeplot(list1, label='1.5', fill=True, alpha=0.5)
    sns.kdeplot(list2, label='2', fill=True, alpha=0.5)
    plt.title('Distribution Comparison: Density Plot')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)

def main():
    correct, total = 0, 0
    correct_list = []
    number_correct_list1 = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="json file path")
    args = parser.parse_args()

    path = args.path

    correct1, total1, number_correct_list = key_eval(path)
    correct += correct1
    total += total1
    correct_list.append(correct1)
    number_correct_list1.append(number_correct_list)

    print(correct_list)
    print(correct / total)
    print(number_correct_list1)

if __name__ == "__main__":
    main()
