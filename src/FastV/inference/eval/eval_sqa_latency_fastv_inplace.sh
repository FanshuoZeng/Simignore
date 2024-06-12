export CUDA_VISIBLE_DEVICES=0

model_path=../LLaVA/llava-v1.5-7b
# output_path=sqa_eval_fastv
# output_path=sqa_eval_fastv_2
output_path=sqa_eval_fastv_3
mkdir -p $output_path

# rank_list=(0 72 144 288 432) # rank equals to (1-R)*N_Image_Tokens, R=(87.5% 75% 50% 25%)

# Ks=(2 6 12 16 20 24 28 32) 
Ks=(10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)
rank_list=(432)

rank_list=(144)
Ks=(5)

for rank in ${rank_list[@]}; do
    for k in ${Ks[@]}; do
    # auto download the ocrvqa dataset
    python ./src/FastV/inference/eval/inference_sqa_test2.py \
        --model-path $model_path \
        --use-fast-v \
        --fast-v-inplace \
        --fast-v-sys-length 36 \
        --fast-v-image-token-length 576 \
        --fast-v-attention-rank $rank \
        --fast-v-agg-layer $k \
        --output-path $output_path/sqa_test_image_7b_FASTV_4bit_inplace_${rank}_${k}.jsonl 
    done
done
#         --fast-v-inplace \



# Baseline
# python ./src/FastV/inference/eval/inference_sqa_test2.py \
#     --model-path $model_path \
#     --output-path $output_path/sqa_test_image_7b_baseline_split_unignore.jsonl 


# python ./src/FastV/inference/eval/inference_sqa_test2.py \
#     --model-path $model_path \
#     --output-path $output_path/sqa_test_image_7b_baseline.jsonl 
