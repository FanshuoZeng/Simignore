import argparse
import json
import os
import re
import random

'''
python eval_science_qa.py \
    --base-dir /path/to/ScienceQA/data/scienceqa \
    --result-file vqa/results/ScienceQA/test_llava-13b.jsonl \
    --output-file vqa/results/ScienceQA/test_llava-13b_output.json \
    --output-result vqa/results/ScienceQA/test_llava-13b_result.json \
'''

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--output-result', type=str)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    return parser.parse_args()


def convert_caps(results):
    fakecaps = []
    for result in results:
        image_id = result['question_id']
        caption = result['text']
        fakecaps.append({"image_id": int(image_id), "caption": caption})
    return fakecaps


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return -1
        return random.choice(range(len(choices)))


if __name__ == "__main__":
    args = get_args()

    base_dir = args.base_dir
    split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))[args.split]
    problems = json.load(open(os.path.join(base_dir, "problems.json")))
    predictions = [json.loads(line) for line in open(args.result_file)]
    predictions = {pred['question_id']: pred for pred in predictions}
    split_problems = {idx: problems[idx] for idx in split_indices}

    results = {'correct': [], 'incorrect': []}
    sqa_results = {}
    sqa_results['acc'] = None
    sqa_results['correct'] = None
    sqa_results['count'] = None
    sqa_results['IMG-Accuracy'] = None
    sqa_results['results'] = {}
    sqa_results['outputs'] = {}

    for prob_id, prob in split_problems.items():
        if prob_id not in predictions:
            pred = {'text': 'FAILED', 'prompt': 'Unknown'}
            pred_text = 'FAILED'
        else:
            pred = predictions[prob_id]
            pred_text = pred['text']


        # if pred_text in args.options:
        #     answer = pred_text
        # elif len(pred_text) >= 3 and pred_text[0] in args.options and pred_text[1:3] == ". ":
        #     answer = pred_text[0]
        # else:
        #     pattern = re.compile(r'The answer is ([A-Z]).')
        #     res = pattern.findall(pred_text)
        #     if len(res) == 1:
        #         answer = res[0]  # 'A', 'B', ...
        #     else:
        #         answer = "FAILED"

# =============================start===================================================

        if pred_text in args.options:
            answer = pred_text
        elif len(pred_text) >= 3 and pred_text[0] in args.options and pred_text[1:3] == ". ":
            answer = pred_text[0]
        elif len(pred_text) >= 3 and pred_text[1] in args.options and pred_text[0] == "(" and pred_text[2] == ")":
            answer = pred_text[1]
        elif len(pred_text) >= 2 and pred_text[0] in args.options and  pred_text[1] == ")":
            answer = pred_text[0]
        elif len(pred_text) >=24 and pred_text[23] in args.options and pred_text[0:23] == "The correct answer is (":
            answer = pred_text[23]
        else:
            pattern = re.compile(r'The answer is ([A-Z]).')
            res = pattern.findall(pred_text)
            if len(res) == 1:
                answer = res[0]  # 'A', 'B', ...
            else:
                answer = "FAILED"  

        if answer == "FAILED":          
            def find_options(pred_text, options):

                pattern = "|".join([f"\(({option})\)" for option in options])
                regex = re.compile(pattern)


                matches = regex.findall(pred_text)


                matched_options = [match for match in matches if match]
                return len(matched_options), matched_options

            num_matches, matched_options = find_options(pred_text, args.options)
            if num_matches == 1:
                matched_options = matched_options[0]
                matched_options = [latter for latter in matched_options]
                for lr in matched_options:
                    if lr in args.options:
                        answer = lr
            else:
                def find_correct_answer(pred_text):   

                    pattern = r"The correct answer is \(([A-E])\)"
                    regex = re.compile(pattern)


                    match = regex.search(pred_text)


                    if match:
                        correct_answer = match.group(1)
                        return correct_answer
                    else:
                        return None

                correct_answer = find_correct_answer(pred_text)
                if correct_answer:
                    answer = correct_answer
                else:
                    answer = "FAILED"
# =============================end===================================================

        pred_idx = get_pred_idx(answer, prob['choices'], args.options)

        analysis = {
            'question_id': prob_id,
            'parsed_ans': answer,
            'ground_truth': args.options[prob['answer']],
            'question': pred['prompt'],
            'pred': pred_text,
            'is_multimodal': '<image>' in pred['prompt'],
        }

        sqa_results['results'][prob_id] = get_pred_idx(answer, prob['choices'], args.options)
        sqa_results['outputs'][prob_id] = pred_text

        if pred_idx == prob['answer']:
            results['correct'].append(analysis)
        else:
            results['incorrect'].append(analysis)

    correct = len(results['correct'])
    total = len(results['correct']) + len(results['incorrect'])

    ###### IMG ######
    multimodal_correct = len([x for x in results['correct'] if x['is_multimodal']])
    multimodal_incorrect = len([x for x in results['incorrect'] if x['is_multimodal']])
    multimodal_total = multimodal_correct + multimodal_incorrect
    ###### IMG ######

    print(f'Total: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%, IMG-Accuracy: {multimodal_correct / multimodal_total * 100:.2f}%')

    sqa_results['acc'] = correct / total * 100
    sqa_results['correct'] = correct
    sqa_results['count'] = total
    sqa_results['IMG-Accuracy'] = multimodal_correct / multimodal_total * 100
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    with open(args.output_result, 'w') as f:
        json.dump(sqa_results, f, indent=2)
