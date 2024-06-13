export CUDA_VISIBLE_DEVICES=0

model_path=../LLaVA/llava-v1.5-7b
output_path=sqa_eval_simignore
mkdir -p $output_path


# 使用simignore
python ./src/Simignore/inference/eval/inference_sqa_llava_7b.py \
    --model-path $model_path \
    --use-simignore \
    --output-path $output_path/sqa_test_image_7b_simignore.jsonl 

# baseline
python ./src/Simignore/inference/eval/inference_sqa_llava_7b.py \
    --model-path $model_path \
    --output-path $output_path/sqa_test_image_7b_baseline.jsonl 