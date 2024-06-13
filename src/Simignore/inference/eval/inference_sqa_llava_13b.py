# %%
import os
# %%
import argparse
import torch
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, OPTION_TOKEN_INDEX, OPTION_AND_QUESTION_TOKEN_INDEX, QUESTION_TOKEN_INDEX, CONTEXT_TOKEN_INDEX, CONTEXT_AND_QUESTION_AND_OPTION_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, tokenizer_image_token_split_option, tokenizer_image_token_split_option_and_question, tokenizer_image_token_split_option_without_letter, tokenizer_image_token_split_option_and_question_and_context, tokenizer_image_token_split_question, tokenizer_image_token_split_C_and_Q_and_O

from PIL import Image
import json
from tqdm import tqdm


import requests
from io import BytesIO
from transformers import TextStreamer

import os
from datasets import load_from_disk
import torch
import json
from tqdm import tqdm
import math
import re	
contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                        "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                        "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                        "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                        "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                        "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                        "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                        "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                        "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                        "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                        "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                        "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                        "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                        "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                        "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                        "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                        "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                        "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                        "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                        "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                        "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                        "youll": "you'll", "youre": "you're", "youve": "you've"}
manualMap    = { 'none': '0',
                        'zero': '0',
                        'one': '1',
                        'two': '2',
                        'three': '3',
                        'four': '4',
                        'five': '5',
                        'six': '6',
                        'seven': '7',
                        'eight': '8',
                        'nine': '9',
                        'ten': '10'
                    }
articles     = ['a',
                        'an',
                        'the'
                    ]


periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip   = re.compile("(\d)(\,)(\d)")
punct        = [';', r"/", '[', ']', '"', '{', '}',
                        '(', ')', '=', '+', '\\', '_', '-',
                        '>', '<', '@', '`', ',', '?', '!']
def processPunctuation( inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub("",
                                    outText,
                                    re.UNICODE)
    return outText
def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText
def clean_text(pred):
    pred = pred.replace('\n', ' ')
    pred = pred.replace('\t', ' ')
    pred = pred.strip()
    pred = processPunctuation(pred)
    pred = processDigitArticle(pred)

    return pred



def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image





def format_choices(choices):
    # example: ['Phoenix', 'Baton Rouge', 'Honolulu', 'Cheyenne'] -> "(A) Phoenix. (B) Baton Rouge. (C) Honolulu. (D) Cheyenne."
    return " ".join([f"({chr(ord('A') + i)}) {choice}" for i, choice in enumerate(choices)])

def format_anwser(choices,anwser_index):
    # example: choices: ['Phoenix', 'Baton Rouge', 'Honolulu', 'Cheyenne'] , anwser_index:0 -> "(A) Phoenix"
    return f"{chr(ord('A') + anwser_index)}"
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]
def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

question_file = "Simignore/data/scienceqa/llava_test_CQM-A_image.json"
num_chunks = 1
chunk_idx = 0
questions = json.load(open(os.path.expanduser(question_file), "r"))
questions = get_chunk(questions, num_chunks, chunk_idx)



idx_list=[]
outputs_list=[]
cur_prompt_list=[]
ans_id_list=[]


image_folder = "Simignore/data/scienceqa/images/test"
valid_images=[]

valid_prompt = []

for i, line in enumerate(tqdm(questions)):
    idx = line["id"]
    idx_list.append(idx)
    question = line['conversations'][0]
    qs = question['value'].replace('<image>', '').strip()
    cur_prompt = qs
    valid_prompt.append(qs)
    if 'image' in line:
        image_file = line["image"]

        image = Image.open(os.path.join(image_folder, image_file))
    else:
        image = None
    valid_images.append(image)

# valid_prompt = valid_prompt[1549 : 1549 + 1]
# valid_images = valid_images[1549 : 1549 + 1]



if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str, required=True, default="/hy-tmp/LLaVA/llava-v1.5-13b")
    parser.add_argument('--use-simignore', default=False, action='store_true', help='whether to use Simignore')
    # output path
    parser.add_argument('--output-path', type=str, required=True, help='the path to save the output json file')

    pargs = parser.parse_args()

    print(pargs)

        # %%
    class InferenceArgs:
        model_path = pargs.model_path
        model_base = None
        image_file = None
        device = "cuda"
        conv_mode = None
        temperature = 0.2
        max_new_tokens = 512
        load_8bit = False
        load_4bit = True
        debug = False
        image_aspect_ratio = 'pad'
        pca_prompt_data = ''
        pca_img_dir = ''
        output_path = ''


    # %%
    args = InferenceArgs()

    # %%
    disable_torch_init()
    # model_name = llava-v1.5-7b
    model_name = get_model_name_from_path(args.model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    # conv_mode = "llava_v1"
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    conv_mode = "vicuna_v1"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    
    if pargs.use_simignore == True:
        model.config.use_simignore = True
        model.config.use_fast_v = True
        model.config.fast_v_inplace = True
        model.config.fast_v_sys_length = 36
        model.config.fast_v_image_token_length = 576
        model.config.fast_v_attention_rank = 144
        model.config.fast_v_agg_layer = 4
        model.config.model_name = 'llava_1.5_13b'

        model.config.simignore_step = 2
        model.config.simignore_k2 = 451
        model.config.simignore_i_index = 0
    else:
        model.config.use_simignore = False
        model.config.use_fast_v = False

    model.model.reset_simignore()

    


    # %%
    def inference(prompts,images):

        outputs = []
        for prompt,image in tqdm(zip(prompts,images),total=len(prompts)):
            cur_prompt = prompt
            if image is not None:
                image = image.convert('RGB')
                image_tensor = process_images([image], image_processor, args)
                
                
                if type(image_tensor) is list:
                    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                else:
                    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            else:
                image_tensor = None
            inp = prompt
            conv = conv_templates[args.conv_mode].copy()
            if image is not None:

                if model.config.mm_use_im_start_end:
                    '''
                    DEFAULT_IM_START_TOKEN = "<im_start>"
                    DEFAULT_IMAGE_TOKEN = "<image>"
                    DEFAULT_IM_END_TOKEN = "<im_end>"
                    '''
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp # False
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                cur_prompt = '<image>' + '\n' + cur_prompt

                inp = inp + '\n' + "Answer with the option's letter from the given choices directly."
                # inp = inp + '\n' + "Just output the letter of the correct answer."
                
                "Just output the letter of the correct answer."

                conv.append_message(conv.roles[0], inp)
                image = None
            else:
                # pargs.fast_v_image_token_length = 0
                # later messages
                inp = inp + '\n' + "Answer with the option's letter from the given choices directly."
                # inp = inp + '\n' + "Just output the letter of the correct answer."


                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."
            # cur_prompt = cur_prompt + '\n' + "Just output the letter of the correct answer."

            cur_prompt_list.append(cur_prompt)

            prompt = conv.get_prompt() #+ "In the image, there is a kitchen with a refrigerator, a sink, and a cup on the counter. The cup is placed on the counter, and there is a backpack nearby. The room appears to be empty, and there are no other objects or people in the scene.\n\nTo bring a clean cup to the person, the robot should first look for a cup. Since the cup is already on the counter, the robot can proceed to pick up the cup. After picking up the cup, the robot should then put the cup into the sink to clean it. Finally, the robot can return the clean cup to the person.\n\nBased on the image, the correct sequence of actions for the robot is (A) Look for a cup, (B) Pick up a cup, and (C) Put the cup into the sink."

            '''
            A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>
            Context: N/A
            Question: What is the name of the colony shown?
            Options: (A) Maryland (B) New Hampshire (C) Rhode Island (D) Vermont
            Answer with the option's letter from the given choices directly. ASSISTANT:
            '''

            if pargs.use_simignore == False:
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            if pargs.use_simignore == True:
                input_ids = tokenizer_image_token_split_option(prompt, tokenizer, IMAGE_TOKEN_INDEX, OPTION_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                
            # input_ids = tokenizer_image_token_split_option_without_letter(prompt, tokenizer, IMAGE_TOKEN_INDEX, OPTION_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            # input_ids = tokenizer_image_token_split_option_and_question(prompt, tokenizer, IMAGE_TOKEN_INDEX, OPTION_AND_QUESTION_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            # input_ids = tokenizer_image_token_split_option_and_question_and_context(prompt, tokenizer, IMAGE_TOKEN_INDEX, OPTION_TOKEN_INDEX, QUESTION_TOKEN_INDEX, CONTEXT_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            # input_ids = tokenizer_image_token_split_question(prompt, tokenizer, IMAGE_TOKEN_INDEX, QUESTION_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            
            # input_ids = tokenizer_image_token_split_C_and_Q_and_O(prompt, tokenizer, IMAGE_TOKEN_INDEX, CONTEXT_AND_QUESTION_AND_OPTION_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()


            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)


            with torch.inference_mode():

                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    attention_mask=None,
                    do_sample=False,
                    max_new_tokens=1,
                    use_cache=False,
                    stopping_criteria=[stopping_criteria],
                    output_attentions=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    )

            

            output = tokenizer.decode(output_ids['sequences'][0, input_ids.shape[1]:],skip_spectial_tokens=True).strip().replace("</s>","")
            outputs.append(output)
            # print(output)
            outputs_list=outputs
            ans_id = shortuuid.uuid()  
            ans_id_list.append(ans_id)

        return idx_list, cur_prompt_list, outputs_list, ans_id_list

    # %%
    # inference and compute cider scores
    idx_list_tutput, cur_prompt_list_output, outputs_list_output, ans_id_list_output = inference(valid_prompt,valid_images)



    output_path = pargs.output_path


    with open(output_path, 'w') as file:
        for idx, cur_prompt, outputs, ans_id in zip(idx_list_tutput, cur_prompt_list_output, outputs_list_output, ans_id_list_output):
            data = {"question_id": idx, "prompt": cur_prompt, "text": outputs, "answer_id": ans_id, "model_id": model_name, "metadata": {}}
            file.write(json.dumps(data) + '\n')

