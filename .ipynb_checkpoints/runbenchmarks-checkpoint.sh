#!/bin/sh
source env.txt
pip install -r requirements.py

apt-get update
apt-get install git-lfs

model_name_list=( "Llama-2-7b-hf" "Llama-2-13b-hf" "Llama-2-70b-hf" )
n_param_list=( "7" "13" "70" )
model_type_list=( "llama" "llama" "llama" )

for i in "${!model_name_list[@]}"; 
do  
    (
    export MODEL_NAME=${model_name_list[i]}
    export MODEL_PARAM=${n_param_list[i]}
    export MODEL_TYPE=${model_type_list[i]}
	#prep
    git clone https://$HF_USERNAME:$HF_TOKEN@huggingface.co/meta-llama/$MODEL_NAME models/$MODEL_NAME/

    #run benchmarks and store result
    python runMMLU.py --ckpt_dir models/$MODEL_NAME --param_size $MODEL_PARAM --model_type $MODEL_TYPE --data_dir benchmarks/MMLU/ --ntrain 5
    
    #cleanup
    rm -rf models/$MODEL_NAME
    )
done

