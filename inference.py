def score_text(text):
    system_prompt =  chatGPT_system_prompt
    prompt =  'Write an explanatory essay based on ideas and information that can be found in the passage set. Manage your time carefully so that you can read the passages; plan your response; write your response; and revise and edit your response. Be sure to use evidence from multiple sources; and avoid overly relying on one source. Your response should be in the form of a multiparagraph essay. Write your essay in the space provided.\n[/INST]'

    all_ids = tokenizer.encode(system_prompt + prompt + text, return_tensors="pt").to('cuda') 
    only_prompt_ids = tokenizer.encode(system_prompt + prompt , return_tensors="pt").to('cpu') 
    n_tokens = all_ids.shape[-1]
    
    logits_processor = LogitsProcessorList()
    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=n_tokens)])
    pad_token_id = model.generation_config.pad_token_id
    eos_token_id = model.generation_config.eos_token_id
    
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
        
    eos_token_id_tensor = torch.tensor(eos_token_id).to(all_ids.device) if eos_token_id is not None else None
    output_scores =  True
    #scores = ()

    with torch.no_grad():
        
        model_inputs = model.prepare_inputs_for_generation(all_ids, )
    
        outputs = model(
            **model_inputs,
            return_dict=False,
        )

    next_token_logits = outputs[0][:, :, :]

    # pre-process distribution
    next_tokens_scores = logits_processor(model_inputs, next_token_logits)
    #top_scores, top_idxs = torch.topk(next_tokens_scores, 2000, dim =-1)

    target_token_scores = next_tokens_scores.gather(2,all_ids.view([1,-1,1]))[-1,:,-1]
    target_token_pos = target_token_pos = torch.sum(torch.stack([
            torch.le(target_token_scores,x_i).long() for x_i in torch.unbind(next_tokens_scores, dim=2)
        ], dim=2),axis = 2).tolist()
    max_token_scores = torch.max(next_tokens_scores,axis = 2).values[-1,:].tolist()
    softmax_sum_token_scores = torch.sum(torch.exp(next_tokens_scores), axis = -1)[-1,:].tolist()
    target_token_idx = all_ids.tolist()
    
    return {
        "softmax_sum_token_scores" : softmax_sum_token_scores,
        "max_token_scores" : max_token_scores,
        "target_token_pos" : target_token_pos,
        "target_token_scores" : target_token_scores.tolist(),
        "n_tokens" : n_tokens,
        "n_prompt_tokens" : only_prompt_ids.shape[-1],
        "target_token_idx" : target_token_idx,
    }
