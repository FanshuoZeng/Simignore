
result_file=sqa_eval_simignore
output_path=vqa/results/ScienceQA
mkdir -p $output_path


# 使用simignore
python src/Simignore/inference/eval/eval_scienceqa.py \
    --base-dir /hy-tmp/LLaVA/ScienceQA/data/scienceqa \
    --result-file $result_file/sqa_test_image_7b_simignore.jsonl \
    --output-file $output_path/sqa_test_image_7b_simignore_output.json \
    --output-result $output_path/sqa_test_image_7b_simignore_result.json



# baseline
python src/Simignore/inference/eval/eval_scienceqa.py \
    --base-dir /hy-tmp/LLaVA/ScienceQA/data/scienceqa \
    --result-file $result_file/sqa_test_image_7b_baseline.jsonl \
    --output-file $output_path/sqa_test_image_7b_baseline_output.json \
    --output-result $output_path/sqa_test_image_7b_baseline_result.json