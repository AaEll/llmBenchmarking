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

choices = ["A", "B", "C", "D"]

def compute_metric(output_filename):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]['pred_answers']
        gold_answers = run_results[task]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold: acc += 1
        print("ACC-%s: %.4f" % (task, acc/len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    print("ACC-all: %.4f" % (total_acc/total_num))


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
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


# def custom_stopping_criteria(input_ids, score, **kwargs):
#     stop_ids = [29871, 13, 13] # \n\n 
#     return input_ids[-len(stop_ids)]

def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    input_tokens = {k:input_tokens[k] for k in input_tokens if k in ["input_ids", "attention_mask"]}
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens

def load(ckpt_dir, model_type):
    n_gpus = torch.cuda.device_count()

    if model_type == 'llama':
        # we use tensor parallel for loading llama
        tokenizer = LlamaTokenizer.from_pretrained(ckpt_dir, use_fast=False, padding_side="left")
        
        model = LlamaForCausalLM.from_pretrained(ckpt_dir, attention_dropout=.1, low_cpu_mem_usage = True, torch_dtype=torch.float16)
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])

        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
    elif model_type == 'flan':
        # we use tensor parallel for loading llama

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=False, padding_side="left")
        
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir, low_cpu_mem_usage = True, torch_dtype=torch.float16)
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])

        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
    elif model_type == 'falcon':
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map = 'balanced_low_0', torch_dtype=torch.bfloat16, trust_remote_code=True)

        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
    elif model_type == 'moss':
        
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, padding_side="left")
        config = AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
        model.tie_weights()
        model = load_checkpoint_and_dispatch(model, model_path, device_map="auto", no_split_module_classes=["MossBlock"], dtype=torch.float16)
        
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
    elif model_type == 'guanaco':



        model_name = "llama-65b"
        adapters_name = 'guanaco-65b'

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory= {i: '24000MB' for i in range(torch.cuda.device_count())},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            ),
        )
        model = PeftModel.from_pretrained(model, adapters_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)


    elif model_type == 'vicuna':
        
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map = 'balanced_low_0', revision="main", trust_remote_code=False)


        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
        
    elif model_type == 'starcoder':
        
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir,device_map = 'balanced_low_0', trust_remote_code=True)
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
        

    else:
        # mpt-30b's tokenizer only has the fast version
        use_fast = "mosaicml/mpt-30b" in ckpt_dir
        # however, tensor parallel for running falcon will occur bugs
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast = use_fast, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map = 'balanced_low_0', torch_dtype=torch.bfloat16, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = 0


    model.eval()

    return model, tokenizer

def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts


def batch_infer(model, tokenizer, prompts):
    batch_size = 8
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model.generate(**encode_inputs, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id)
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    answers = [answer[-1] for answer in answers]
    return answers

def confidence_infer(model, tokenizer, prompts, token_confidence_funcs, confidence_aggregation_funcs, sequence_confidence_funcs):
    answers = []
    confidences = []
    for prompt in tqdm(prompts):
        answer, confidence_dict = generate_with_confidence(model, tokenizer, prompt, 1, token_confidence_funcs, confidence_aggregation_funcs, sequence_confidence_funcs )
        
        answers.extend(answer)
        confidences.extend(confidence_dict)
        
    return answers,confidences



def min_confidence_agg(values, all_ids, model):
    return 'min', min(values)
def max_confidence_agg(values, all_ids, model):
    return 'max', max(values)
def avg_confidence_agg(values, all_ids, model):
    return 'avg', torch.mean(torch.cat(tuple([v.unsqueeze(0) for v in values])),dim = -1)
def attention_weighted_agg(values, all_ids, model):
    #based on Guan et. al. 2023 Shifting Attention to Relevance
    with torch.no_grad():
        model_inputs = model.prepare_inputs_for_generation(all_ids, )
        outputs = model(
            **model_inputs,
                return_dict=True,
                output_attentions=True,
                output_hidden_states=False,
        )
    attn_weights = outputs['attentions'][0][-1,-1,-1,-1*len(values):]
    return 'attention_weighted', torch.dot(torch.cat(tuple([v.unsqueeze(0) for v in values])).half() ,attn_weights)/torch.sum(attn_weights)

def logit_confidence(next_tokens_scores, next_token, all_ids, model):
    return 'logit', torch.max(next_tokens_scores[-1,-1,:])
    
def softmax_confidence(next_tokens_scores, next_token, all_ids, model):
    return 'softmax' , torch.max(torch.nn.functional.softmax(next_tokens_scores[-1,-1,:], dim = -1))
    
def entropy_confidence(next_tokens_scores, next_token, all_ids, model):
    return 'entropy' , Categorical(probs=torch.nn.functional.softmax(next_tokens_scores[-1,-1,:], dim = -1)).entropy()

def ensemble_entropy_confidence(next_tokens_scores, next_token, all_ids, model, n_ensemble=5):
    model.train()
    estimate_vector = torch.zeros(next_tokens_scores.shape[-1], dtype = torch.float32, device = model.device)
    logits_processor = LogitsProcessorList()

    for i in range(n_ensemble):
        
        with torch.no_grad():
            model_inputs = model.prepare_inputs_for_generation(all_ids, )
            outputs = model(
                **model_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
            )
        next_tokens_scores = logits_processor(outputs['logits'][:,-1,:], outputs['logits'])
        next_token = torch.argmax(next_tokens_scores[:,-1,:])
        estimate_vector[next_token.item()] += 1
        
    estimate_vector = estimate_vector/n_ensemble # probabilities should sum to 1
    model.eval()
    return 'ensemble_entropy' , Categorical(probs=estimate_vector).entropy() 

def MMLU_self_reflection_confidence_promptv1(n_prompt_tokens,all_ids, model, tokenizer, prompt):
    proposed_answer = tokenizer.decode(all_ids[-1,-1*(all_ids.shape[-1] - n_prompt_tokens):])
    prompt = prompt[:-7] # remove the "Answer:" at the end of the prompt
    prompt = prompt + f'''Proposed Answer: {proposed_answer}
Is the proposed answer:
(A) True
(B) False
The proposed answer is: '''
    encode_inputs = prepare_prompt(tokenizer, prompt)
    outputs = model.generate(**encode_inputs, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id)

    return 'self_reflection_promptv1',1 if outputs[-1,-1].item() == tokenizer.encode('A')[-1] else 0 #return 1 if model outputs 'A' else 0

def prepare_prompt(tokenizer, prompt):
    input_tokens = tokenizer.encode_plus(prompt, return_tensors="pt", padding=True)
    input_tokens = {
            k:input_tokens[k] for k in input_tokens if k in ["input_ids", "attention_mask"]
    }
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')
    return input_tokens




def generate_with_confidence(model, tokenizer, prompt, max_new_tokens, token_confidence_funcs, confidence_aggregation_funcs, sequence_confidence_funcs ):
    all_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device) 
    n_prompt_tokens = all_ids.shape[-1]
    logits_processor = LogitsProcessorList()
    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=n_prompt_tokens+max_new_tokens)])
    pad_token_id = model.generation_config.pad_token_id
    eos_token_id = model.generation_config.eos_token_id

    all_token_confidences = {}
    sequence_confidences = {}

    while True:

        with torch.no_grad():
            model_inputs = model.prepare_inputs_for_generation(all_ids, )
        

            outputs = model(
                **model_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
            )
        
        next_tokens_scores = logits_processor(outputs['logits'][:,-1,:], outputs['logits'])
        next_token = torch.argmax(next_tokens_scores[:,-1,:])

        token_confidences = { id: val for id, val in [func(next_tokens_scores, next_token, all_ids, model) for func in token_confidence_funcs]}

        for id, val in token_confidences.items():
            all_token_confidences[id] = all_token_confidences.get(id,[])
            all_token_confidences[id].append(val)
        
        all_ids = torch.cat([all_ids,next_token.unsqueeze(0).unsqueeze(0)],axis = -1)

        if stopping_criteria(all_ids, next_tokens_scores):
            break


    for func in confidence_aggregation_funcs:
        for token_confidence_id , values in all_token_confidences.items():
            agg_id , val = func(values, all_ids, model)
            sequence_confidences[agg_id+'|'+token_confidence_id] = val

    for func in sequence_confidence_funcs:
        seq_confidence_id , val = func(n_prompt_tokens,all_ids, model, tokenizer, prompt)
        sequence_confidences[seq_confidence_id] = val       

    
    answer = tokenizer.decode(all_ids[-1,-1*max_new_tokens:])
    return [answer], sequence_confidences


def main(ckpt_dir: str, param_size: str, model_type: str):
    
    run_results = {}
    benchmark = 'MMLU'

    output_filename = 'run_results_%s_%s_%sb.json' % (benchmark, model_type, param_size)

    confidence_aggregation_funcs = [min_confidence_agg,max_confidence_agg,avg_confidence_agg,attention_weighted_agg]
    token_confidence_funcs = [logit_confidence,softmax_confidence,entropy_confidence, ensemble_entropy_confidence]
    sequence_confidence_funcs = [MMLU_self_reflection_confidence_promptv1, ]

    
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
