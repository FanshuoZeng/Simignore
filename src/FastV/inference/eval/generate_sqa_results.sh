# rank_list=(72 144 288 432) # rank equals to (1-R)*N_Image_Tokens, R=(87.5% 75% 50% 25%)
# # rank_list=(72) # rank equals to (1-R)*N_Image_Tokens, R=(87.5% 75% 50% 25%)

# Ks=(2 6 11 16 20 24 28 32) 
# rank_list=(0 72 144 288 432) # rank equals to (1-R)*N_Image_Tokens, R=(87.5% 75% 50% 25%)
# Ks=(2 6 12 16 20 24 28 32) 
# Ks=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)
Ks=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)
rank_list=(72 144 288 432)

rank_list=(144)
Ks=(5)
# result_file=sqa_eval_fastv

# result_file=sqa_eval_fastv_2
# output_path=vqa_2/results/ScienceQA


result_file=sqa_eval_fastv_3
output_path=vqa_3/results/ScienceQA
mkdir -p $output_path

for rank in ${rank_list[@]}; do
    for k in ${Ks[@]}; do
    # auto download the ocrvqa dataset
    python src/FastV/inference/eval/eval_scienceqa.py \
    --base-dir /hy-tmp/LLaVA/ScienceQA/data/scienceqa \
    --result-file $result_file/sqa_test_image_7b_FASTV_4bit_inplace_${rank}_${k}.jsonl  \
    --output-file $output_path/sqa_test_image_7b_FASTV_4bit_inplace_${rank}_${k}_output.json \
    --output-result $output_path/sqa_test_image_7b_FASTV_4bit_inplace_${rank}_${k}_result.json
    done
done

# Baseline
python src/FastV/inference/eval/eval_scienceqa.py \
    --base-dir /hy-tmp/LLaVA/ScienceQA/data/scienceqa \
    --result-file $result_file/sqa_test_image_7b_baseline_split_unignore.jsonl \
    --output-file $output_path/sqa_test_image_7b_baseline_split_unignore_output.json \
    --output-result $output_path/sqa_test_image_7b_baseline_split_unignore_result.json


# python src/FastV/inference/eval/eval_scienceqa.py \
#     --base-dir /hy-tmp/LLaVA/ScienceQA/data/scienceqa \
#     --result-file $result_file/sqa_test_image_7b_baseline.jsonl \
#     --output-file $output_path/sqa_test_image_7b_baseline_output.json \
#     --output-result $output_path/sqa_test_image_7b_baseline_result.json