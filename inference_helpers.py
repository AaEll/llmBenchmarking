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
        model = load_checkpoint_and_dispatch(model, ckpt_dir, device_map="auto", no_split_module_classes=["MossBlock"], dtype=torch.float16)
        
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
        confidences.append(confidence_dict)
        
    return answers,confidences



def min_confidence_agg(values, all_ids, model):
    return 'min', min(values).item()
def max_confidence_agg(values, all_ids, model):
    return 'max', max(values).item()
def avg_confidence_agg(values, all_ids, model):
    return 'avg', torch.mean(torch.cat(tuple([v.unsqueeze(0) for v in values])),dim = -1).item()
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
    return 'attention_weighted', (torch.dot(torch.cat(tuple([v.unsqueeze(0) for v in values])).half() ,attn_weights)/torch.sum(attn_weights)).item()

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

def MMLU_self_reflection_confidence_promptv2(n_prompt_tokens,all_ids, model, tokenizer, prompt):
    proposed_answer = tokenizer.decode(all_ids[-1,-1*(all_ids.shape[-1] - n_prompt_tokens):])
    prompt = prompt[:-7] # remove the "Answer:" at the end of the prompt
    prompt = ''.join(prompt.split('A.')[:-1]) # remove multiple choice
    prompt = prompt + f'''Proposed Answer: {proposed_answer}
Is the proposed answer:
(A) True
(B) False
The proposed answer is: '''
    encode_inputs = prepare_prompt(tokenizer, prompt)
    outputs = model.generate(**encode_inputs, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id)

    return 'self_reflection_promptv2',1 if outputs[-1,-1].item() == tokenizer.encode('A')[-1] else 0 #return 1 if model outputs 'A' else 0

def MMLU_self_reflection_confidence_promptv3(n_prompt_tokens,all_ids, model, tokenizer, prompt):
    proposed_answer = tokenizer.decode(all_ids[-1,-1*(all_ids.shape[-1] - n_prompt_tokens):])
    prompt = prompt[:-7] # remove the "Answer:" at the end of the prompt
    prompt = ''.join(prompt.split('\n\n')[-2:]) # remove multiple shot
    prompt = prompt + f'''Proposed Answer: {proposed_answer}
Is the proposed answer:
(A) True
(B) False
The proposed answer is: '''
    encode_inputs = prepare_prompt(tokenizer, prompt)
    outputs = model.generate(**encode_inputs, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id)

    return 'self_reflection_promptv3',1 if outputs[-1,-1].item() == tokenizer.encode('A')[-1] else 0 #return 1 if model outputs 'A' else 0

def MMLU_self_reflection_confidence_promptv4(n_prompt_tokens,all_ids, model, tokenizer, prompt):
    #closest to Anthropic paper
    proposed_answer = tokenizer.decode(all_ids[-1,-1*(all_ids.shape[-1] - n_prompt_tokens):])
    prompt = prompt[:-7] # remove the "Answer:" at the end of the prompt
    prompt = ''.join(prompt.split('A.')[:-1]) # remove multiple choice
    prompt = ''.join(prompt.split('\n\n')[-2:]) # remove multiple shot
    prompt = prompt + f'''Proposed Answer: {proposed_answer}
Is the proposed answer:
(A) True
(B) False
The proposed answer is: '''
    encode_inputs = prepare_prompt(tokenizer, prompt)
    outputs = model.generate(**encode_inputs, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id)

    return 'self_reflection_promptv4',1 if outputs[-1,-1].item() == tokenizer.encode('A')[-1] else 0 #return 1 if model outputs 'A' else 0

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


