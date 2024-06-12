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
# 没用到
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
# 没用到
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
# 没用到
def clean_text(pred):
    pred = pred.replace('\n', ' ')
    pred = pred.replace('\t', ' ')
    pred = pred.strip()
    pred = processPunctuation(pred)
    pred = processDigitArticle(pred)

    return pred



# %%
# 没用到
def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


# 一个完整的提示 ex:
'''
"\nAnalyse the image and choose the best answer for the following question:\nWhat is in the motorcyclist's mouth?\nOptions: (A) toothpick (B) food (C) popsicle stick (D) cigarette\nJust output the letter of the correct answer."
'''
# 列表
# valid_prompt = [TEMPLATE.format(question=question, options=format_choices(choice)) for question, choice in zip(valid_questions, valid_choices)]


### 参照LLaVA的model_vqa_science.py读取Sqa数据集
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

# 读取Sqa数据，将图片变成一个list列表。没有图片的置为None
# 这里先将路径写死吧
question_file = "/hy-tmp/LLaVA/ScienceQA/data/scienceqa/llava_test_CQM-A_image.json"
num_chunks = 1
chunk_idx = 0
questions = json.load(open(os.path.expanduser(question_file), "r"))
questions = get_chunk(questions, num_chunks, chunk_idx)
'''
questions是一个列表，questions[0]长这样：
{'id': '4', 'conversations': [{'from': 'human', 'value': 'Context: N/A\nQuestion: Which figure of speech is used in this text?\nSing, O goddess, the anger of Achilles son of Peleus, that brought countless ills upon the Achaeans.\n—Homer, The Iliad\nOptions: (A) chiasmus (B) apostrophe'}, {'from': 'gpt', 'value': 'The answer is B.'}]}
'''

# 初始化需要的列表
idx_list=[]
outputs_list=[]
cur_prompt_list=[]
ans_id_list=[]

# 图片列表 
# 图片位置先写死
image_folder = "/hy-tmp/LLaVA/ScienceQA/data/scienceqa/images/test"
valid_images=[]
# valid_prompt列表——初始提示列表
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
        # 图片
        image = Image.open(os.path.join(image_folder, image_file))
    else:
        image = None
    valid_images.append(image)

# 拿出k个数据测试
valid_prompt = valid_prompt[1549 : 1549 + 1]
valid_images = valid_images[1549 : 1549 + 1]


# 两个生物链的数据，第 84 个我们的方法回答错误而baseline回答正确；第 590 个我们的方法回答正确(???错误)，而baseline回答错误。

# 两个地理题的数据，第 144 个我们的方法回答错误而baseline回答正确；第 12 个我们的方法回答正确，而baseline回答错误。

# index:164 394 110 774 1201 1591 1630 423 173 1317

# 86 164 173 394 468 484 500 774 865 947 86 438 811 1202


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str, required=True, default="/hy-tmp/LLaVA/llava-v1.5-13b")
    parser.add_argument('--use-fast-v', default=False, action='store_true', help='whether to use fast-v')
    parser.add_argument('--fast-v-inplace', default=False, action='store_true', help='whether to use fast-v inplace version to check real latency, no supported kv cache yet')
    parser.add_argument('--fast-v-sys-length', type=int, required=False, help='the length of system prompt')
    parser.add_argument('--fast-v-image-token-length', type=int, required=False, help='the length of image token')
    parser.add_argument('--fast-v-attention-rank', type=int, required=False, help='the rank of attention matrix')
    parser.add_argument('--fast-v-agg-layer', type=int, required=False, help='the layer of attention matrix')
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
    # 作用是禁用 PyTorch 的默认初始化操作，以加速模型创建过程。
    disable_torch_init()
    # model_name = llava-v1.5-7b
    model_name = get_model_name_from_path(args.model_path)
    # 这里的model应该是经过FastV修改之后的，model.config应该包含fastv相关参数。。我追溯load_pretrained_model()函数，试图验证目前尚未追溯到
    # 追溯到了，就是继承的modeling——llama的LlamaModel
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
    #如果根据模型路径推断的conv_mode 与 args.conv_mode 定义的模型不一致，则使用 args.conv_mode 定义的模型。（前提是定义的 args.conv_mode 模型 not None）
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    
    # set model fastv config
    if pargs.use_fast_v == True:
        model.config.use_fast_v = True
        model.config.fast_v_inplace = pargs.fast_v_inplace
        model.config.fast_v_sys_length = pargs.fast_v_sys_length
        model.config.fast_v_image_token_length = pargs.fast_v_image_token_length
        model.config.fast_v_attention_rank = pargs.fast_v_attention_rank
        model.config.fast_v_agg_layer = pargs.fast_v_agg_layer
    else:
        model.config.use_fast_v = False
    # 设置fastv的参数为配置中的参数，即 fastv.parameter = fastv.config.parameter
    model.model.reset_fastv()

    


    # %%
    def inference(prompts,images):
        # print("prompts")
        # print(prompts)
        # print("images1:")
        # print(images)
        outputs = []
        for prompt,image in tqdm(zip(prompts,images),total=len(prompts)):
            # prompt = ""
            cur_prompt = prompt
            # print("prompt1是:")
            # print(prompt)
            if image is not None:
                image = image.convert('RGB')
                # print("images2:")
                # print(images)
                image_tensor = process_images([image], image_processor, args)
                # print(image_tensor[0][1][122])
                # 1 * 3 * 336 * 336
                # image_tensor_list = image_tensor.tolist()
                # print("image_tensor_list:")
                # print(len(image_tensor_list))
                # print(len(image_tensor_list[0]))
                # print(len(image_tensor_list[0][0]))
                # print(len(image_tensor_list[0][0][0]))
                '''
                # 将列表写入文本文件
                with open('outputtxt.txt', 'w') as f:
                    for row in image_tensor_list:
                        # 将每个数值转换为字符串，并使用空格分隔
                        # 也可以根据需要使用其他分隔符或格式
                        row_str = ' '.join(map(str, row))
                        f.write(row_str + '\n')  # 每个tensor row写入文件的新行
                '''
                # conv 是什么？  
                # 这里的conv_templates.[args.conv_mode] 是 conv_llava_v1，conv 应该也是 conv_llava_v1
                '''
                conv_llava_v1 = Conversation(
                    system="A chat between a curious human and an artificial intelligence assistant. "
                        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
                    roles=("USER", "ASSISTANT"),
                    version="v1",
                    messages=(),
                    offset=0,
                    sep_style=SeparatorStyle.TWO,
                    sep=" ",
                    sep2="</s>",
                )
                '''
                # def copy(seif) return Conversation(...) Conversation是啥？
                
                if type(image_tensor) is list:
                    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                else:
                    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            else:
                image_tensor = None
            inp = prompt
            conv = conv_templates[args.conv_mode].copy()
            if image is not None:
                # first message
                # mm_use_im_start_end这个是干什么的？
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
                # conv.roles[0] 是什么？
                # conv.roles[0]是'USER'
                # conv_llava_v1.roles = ("USER", "ASSISTANT")
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
            # get_prompt()是什么的？
            '''
            '''
            prompt = conv.get_prompt() #+ "In the image, there is a kitchen with a refrigerator, a sink, and a cup on the counter. The cup is placed on the counter, and there is a backpack nearby. The room appears to be empty, and there are no other objects or people in the scene.\n\nTo bring a clean cup to the person, the robot should first look for a cup. Since the cup is already on the counter, the robot can proceed to pick up the cup. After picking up the cup, the robot should then put the cup into the sink to clean it. Finally, the robot can return the clean cup to the person.\n\nBased on the image, the correct sequence of actions for the robot is (A) Look for a cup, (B) Pick up a cup, and (C) Put the cup into the sink."
            # IMAGE_TOKEN_INDEX = -200
            # 是将文本 prompt 转换为模型的输入张量， unsqueeze(0) 是在维度 0 上添加一个新的维度，将其转换为一个包含单个示例的批次
            # print("prompt:")
            # print(prompt) 在原来prompt的开头和结尾加了一些提示
            # print("prompt2是:")
            # print(prompt)
            # prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>"
            # prompt = ""

            # print("最终的prompt是：")
            # print(prompt)
            '''
            A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>
            Context: N/A
            Question: What is the name of the colony shown?
            Options: (A) Maryland (B) New Hampshire (C) Rhode Island (D) Vermont
            Answer with the option's letter from the given choices directly. ASSISTANT:
            '''

            # prompt = "## Options: (A) Maryland (B) New Hampshire (C) Rhode Island (D) Vermont ##"
            # prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nContext: N/A\nQuestion: What is the name of the colony shown?\n<Options>Options: (A) Maryland (B) New Hampshire (C) Rhode Island (D) Vermont<Options>\nAnswer with the option's letter from the given choices directly. ASSISTANT:"
            # 在'Options:'前添加'<Options>'
            
            # print("prompt是：：：：：")
            # print(prompt)
            '''
            只分割image
            '''
            # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            '''
            分割option(包括分割image)
            '''
            # input_ids = tokenizer_image_token_split_option(prompt, tokenizer, IMAGE_TOKEN_INDEX, OPTION_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            '''
            分割option(包括分割image), 并拼接去除无关信息的option
            '''
            input_ids = tokenizer_image_token_split_option_without_letter(prompt, tokenizer, IMAGE_TOKEN_INDEX, OPTION_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            '''
            分割option的question(包括分割image), 但是option和question连接在一起
            '''
            # input_ids = tokenizer_image_token_split_option_and_question(prompt, tokenizer, IMAGE_TOKEN_INDEX, OPTION_AND_QUESTION_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            '''
            把image、context、question、option单独分割
            '''
            # input_ids = tokenizer_image_token_split_option_and_question_and_context(prompt, tokenizer, IMAGE_TOKEN_INDEX, OPTION_TOKEN_INDEX, QUESTION_TOKEN_INDEX, CONTEXT_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            '''
            分割question(包括分割image)
            '''
            # input_ids = tokenizer_image_token_split_question(prompt, tokenizer, IMAGE_TOKEN_INDEX, QUESTION_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            
            '''
            把context、question、option一起分割
            '''
            # input_ids = tokenizer_image_token_split_C_and_Q_and_O(prompt, tokenizer, IMAGE_TOKEN_INDEX, CONTEXT_AND_QUESTION_AND_OPTION_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()


            # input_ids_list = input_ids.tolist()
            # print("input_ids:")
            # print(len(input_ids_list[0]))

            # 在生成文本时设定停止条件
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            # 目前input_ids未涉及图像张量
            with torch.inference_mode():
                # generate() 在transformers/generation/utils.py定义 好像不是
                # /hy-tmp/FastV/src/LLaVA/llava/model/language_model/llava_llama.py
                # print("开始")
                # print("inference的input_ids是：")
                # print(input_ids.shape)
                # print(input_ids)
                # print("model.generate()函数：")
                # print(model.generate)
                output_ids = model.generate(
                    # 这时候文本和图像信息维度还不一致
                    input_ids,
                    # 加入图像信息
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
                # print("结束")
            

            output = tokenizer.decode(output_ids['sequences'][0, input_ids.shape[1]:],skip_spectial_tokens=True).strip().replace("</s>","")
            outputs.append(output)
            # print(output)
            outputs_list=outputs
            #用来生成唯一标识符
            ans_id = shortuuid.uuid()  
            ans_id_list.append(ans_id)

        return idx_list, cur_prompt_list, outputs_list, ans_id_list

    # %%
    # inference and compute cider scores
    idx_list_tutput, cur_prompt_list_output, outputs_list_output, ans_id_list_output = inference(valid_prompt,valid_images)

    # print("outputs_list_output是：")
    # print(outputs_list_output)

    output_path = pargs.output_path


    with open(output_path, 'w') as file:
        for idx, cur_prompt, outputs, ans_id in zip(idx_list_tutput, cur_prompt_list_output, outputs_list_output, ans_id_list_output):
            # 创建字典
            data = {"question_id": idx, "prompt": cur_prompt, "text": outputs, "answer_id": ans_id, "model_id": model_name, "metadata": {}}
            # 将字典转换为JSON字符串并写入文件
            file.write(json.dumps(data) + '\n')

