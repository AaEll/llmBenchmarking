import argparse
import json
import os
import time
import math
import numpy as np
import pandas as pd
import re

import pandas as pd
import tensor_parallel as tp
import torch
from torch.distributions import Categorical
from tqdm import tqdm, tqdm_notebook
from scipy.stats import entropy
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from  transformers.generation.logits_process import LogitsProcessorList
from  transformers.generation.stopping_criteria import StoppingCriteriaList, MaxLengthCriteria
import transformers
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from inference_helpers import MMLU_self_reflection_confidence_promptv1,MMLU_self_reflection_confidence_promptv2,MMLU_self_reflection_confidence_promptv3,MMLU_self_reflection_confidence_promptv4, logit_confidence,softmax_confidence,entropy_confidence, ensemble_entropy_confidence, min_confidence_agg,max_confidence_agg,avg_confidence_agg,attention_weighted_agg, load,confidence_infer,compute_metric

TASKS = [
        'abstract_algebra',
        'anatomy',
        'astronomy',
        'business_ethics',
        'clinical_knowledge',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_medicine',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'econometrics',
        'electrical_engineering',
        'elementary_mathematics',
        'formal_logic',
        'global_facts',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_european_history',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_mathematics',
        'high_school_microeconomics',
        'high_school_physics',
        'high_school_psychology',
        'high_school_statistics',
        'high_school_us_history',
        'high_school_world_history',
        'human_aging',
        'human_sexuality',
        'international_law',
        'jurisprudence',
        'logical_fallacies',
        'machine_learning',
        'management',
        'marketing',
        'medical_genetics',
        'miscellaneous',
        'moral_disputes',
        'moral_scenarios',
        'nutrition',
        'philosophy',
        'prehistory',
        'professional_accounting',
        'professional_law',
        'professional_medicine',
        'professional_psychology',
        'public_relations',
        'security_studies', 
        'sociology',
        'us_foreign_policy',
        'virology',
        'world_religions']

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True, choices = ["A", "B", "C", "D"]):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def main(ckpt_dir: str, param_size: str, model_type: str):
    
    run_results = {}
    benchmark = 'MMLU'

    output_filename = 'results/run_results_%s_%s_%sb.json' % (benchmark, model_type, param_size)

    
    
    confidence_aggregation_funcs = [min_confidence_agg,max_confidence_agg,avg_confidence_agg,attention_weighted_agg]
    token_confidence_funcs = [logit_confidence,softmax_confidence,entropy_confidence, ensemble_entropy_confidence]
    sequence_confidence_funcs = [MMLU_self_reflection_confidence_promptv1,MMLU_self_reflection_confidence_promptv2,MMLU_self_reflection_confidence_promptv3,MMLU_self_reflection_confidence_promptv4 ]

    
    model, tokenizer = load(ckpt_dir, model_type)
    start_time = time.time()
    for task in TASKS:
        print('Testing %s ...' % task)
        records = []
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", task + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", task + "_test.csv"), header=None)
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = args.ntrain
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, task, k)
            prompt = train_prompt + prompt_end
            while len(tokenizer.tokenize(prompt)) + 1> 2048: # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = '\n\n'.join(prompt_split)
            label = test_df.iloc[i, test_df.shape[1]-1]
            records.append({'prompt':prompt, 'answer':label})

        pred_answers, confidences = confidence_infer(model, tokenizer, [record['prompt'] for record in records],token_confidence_funcs, confidence_aggregation_funcs, sequence_confidence_funcs)
        gold_answers = [record['answer'] for record in records]
        run_results[task] = {'pred_answers':pred_answers, 'gold_answers':gold_answers, 'confidences' :confidences}
    with open(output_filename, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)
    
    compute_metric(output_filename)
    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--param_size', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--ntrain', type=int, default=5)
    args = parser.parse_args()
    
    main(args.ckpt_dir, args.param_size, args.model_type)
