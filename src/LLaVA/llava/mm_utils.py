from PIL import Image
from io import BytesIO
import base64

import torch
from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, OPTION_TOKEN_INDEX, OPTION_AND_QUESTION_TOKEN_INDEX, QUESTION_TOKEN_INDEX, CONTEXT_TOKEN_INDEX, CONTEXT_AND_QUESTION_AND_OPTION_TOKEN_INDEX
import re
import string

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    # print("原始的prompt：")
    # print(prompt)
    # print("分割后的prompt")
    # print(prompt.split('<image>'))
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
    # print(prompt_chunks)
    # print(len(prompt_chunks[0]))
    # print(len(prompt_chunks[1]))

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    # print("tokenizer.bos_token_id = ", tokenizer.bos_token_id) # 1
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    # print(input_ids)
    # print(image_token_index) # -200

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        # print("当前input_ids是：")
        # print(input_ids)
        # print("x[offset:]是：")
        # print(x[offset:])
        input_ids.extend(x[offset:])
        # print("修改后的input_ids是：")
        # print(input_ids)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def tokenizer_image_token_split_option(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, optin_token_index=OPTION_TOKEN_INDEX, return_tensors=None):
    # 在options的前后加上标签
    prompt = prompt.replace("Options:", "<Options>Options:")
    prompt = prompt.replace("\nAnswer with the option's letter from the given choices directly", "<Options>\nAnswer with the option's letter from the given choices directly")
    # prompt = prompt.replace("\nJust output the letter of the correct answer", "<Options>\nJust output the letter of the correct answer")
    
    "Just output the letter of the correct answer."

    # print("添加标签后的prompt：")
    # print(prompt)

    # print("分割后的prompt")
    # print(prompt.split('<image>'))

    # option的分隔符是<Options>
    prompt_split_img_list = prompt.split('<image>')
    # print("prompt_split_img_list是：")
    # print(prompt_split_img_list)
    prompt_split_opt_list = prompt_split_img_list[1].split('<Options>')
    # print("prompt_split_opt_list是：")
    # print(prompt_split_opt_list)
    # print(prompt_split_opt_list[1])  #这个是option的内容
    prompt_list=[]
    prompt_list.append(prompt_split_img_list[0])
    prompt_list.append(prompt_split_opt_list[0])
    prompt_list.append(prompt_split_opt_list[1])
    prompt_list.append(prompt_split_opt_list[2])
    # print("prompt_list是：")
    # print(prompt_list)

    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt_list]
    # print("未拼接的为：")
    # print(prompt_chunks)

    prompt_chunks_without_first = prompt_chunks[1:]
    # print("prompt_chunks_without_first是：")
    # print(prompt_chunks_without_first)

    input_ids_without_first = []
    offset = 0
    if len(prompt_chunks_without_first) > 0 and len(prompt_chunks_without_first[0]) > 0 and prompt_chunks_without_first[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids_without_first.append(prompt_chunks_without_first[0][0])

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
    for x in insert_separator(prompt_chunks_without_first, [optin_token_index] * (offset + 1)):
        input_ids_without_first.extend(x[offset:])

    # print("input_ids_without_first是：")
    # print(input_ids_without_first)

    prompt_chunks_with_first = []
    prompt_chunks_with_first.append(prompt_chunks[0])
    prompt_chunks_with_first.append(input_ids_without_first)

    input_ids = []
    offset = 0
    # print("tokenizer.bos_token_id = ", tokenizer.bos_token_id) # 1
    if len(prompt_chunks_with_first) > 0 and len(prompt_chunks_with_first[0]) > 0 and prompt_chunks_with_first[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks_with_first[0][0])

    for x in insert_separator(prompt_chunks_with_first, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    
    # print("拼接之后的input_ids是：")
    # print(input_ids)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def tokenizer_image_token_split_option_and_question(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, optin_and_question_token_index=OPTION_AND_QUESTION_TOKEN_INDEX, return_tensors=None):
    # 在options的前后加上标签
    prompt = prompt.replace("Question:", "<Options_And_Question>Question:")
    prompt = prompt.replace("\nAnswer with the option's letter from the given choices directly", "<Options_And_Question>\nAnswer with the option's letter from the given choices directly")
    
    # print("添加标签后的prompt：")
    # print(prompt)
    
    # print("分割后的prompt")
    # print(prompt.split('<image>'))

    # option的分隔符是<Options>
    prompt_split_img_list = prompt.split('<image>')
    # print("prompt_split_img_list是：")
    # print(prompt_split_img_list)

    prompt_split_opt_and_question_list = prompt_split_img_list[1].split('<Options_And_Question>')
    # print("prompt_split_opt_and_question_list是：")
    # print(prompt_split_opt_and_question_list)
    # print(prompt_split_opt_list[1])  #这个是option的内容
    prompt_list=[]
    prompt_list.append(prompt_split_img_list[0])
    prompt_list.append(prompt_split_opt_and_question_list[0])
    prompt_list.append(prompt_split_opt_and_question_list[1])
    prompt_list.append(prompt_split_opt_and_question_list[2])

    # print("prompt_list是：")
    # print(prompt_list)

    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt_list]
    # print("未拼接的prompt_chunks为：")
    # print(prompt_chunks)

    prompt_chunks_without_first = prompt_chunks[1:]
    # print("prompt_chunks_without_first是：")
    # print(prompt_chunks_without_first)

    input_ids_without_first = []
    offset = 0
    if len(prompt_chunks_without_first) > 0 and len(prompt_chunks_without_first[0]) > 0 and prompt_chunks_without_first[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids_without_first.append(prompt_chunks_without_first[0][0])

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
    for x in insert_separator(prompt_chunks_without_first, [optin_and_question_token_index] * (offset + 1)):
        input_ids_without_first.extend(x[offset:])

    # print("input_ids_without_first是：")
    # print(input_ids_without_first)

    prompt_chunks_with_first = []
    prompt_chunks_with_first.append(prompt_chunks[0])
    prompt_chunks_with_first.append(input_ids_without_first)

    input_ids = []
    offset = 0
    # print("tokenizer.bos_token_id = ", tokenizer.bos_token_id) # 1
    if len(prompt_chunks_with_first) > 0 and len(prompt_chunks_with_first[0]) > 0 and prompt_chunks_with_first[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks_with_first[0][0])

    for x in insert_separator(prompt_chunks_with_first, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    
    # print("拼接之后的input_ids是：")
    # print(input_ids)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def tokenizer_image_token_split_option_without_letter(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, optin_token_index=OPTION_TOKEN_INDEX, return_tensors=None):
    # 在options的前后加上标签
    prompt = prompt.replace("Options:", "<Options>Options:")
    prompt = prompt.replace("\nAnswer with the option's letter from the given choices directly", "<Options>\nAnswer with the option's letter from the given choices directly")
    # prompt = prompt.replace("Options:", "<Options>Options:")
    # prompt = prompt.replace("\nJust output the letter of the correct answer", "<Options>\nJust output the letter of the correct answer")

    
    # prompt = prompt.replace("\nGive reasons for your answer.", "<Options>\nGive reasons for your answer.")


    "Just output the letter of the correct answer."
    
    # 奖option去掉开头的字母 (A)
    # 将文本分割成两部分
    options_part = prompt.split("Options: ")
    # 检查是否成功分割并获取第二部分
    if len(options_part) > 1:
        # 进一步分割以去除后面的部分
        prompt_option = options_part[1].split("<Options>\nAnswer with the option's letter from the given choices directly")[0]
        # prompt_option = options_part[1].split("<Options>\nJust output the letter of the correct answer")[0]

        # prompt_option = options_part[1].split("<Options>\nGive reasons for your answer.")[0]

    # Splitting the options using regular expression
    split_options = re.split(r' (?=\([A-Z]\))', prompt_option)

    options_str = ""

    for opt in split_options:
        options_str += opt[3:]
    options_str = options_str[1:]

    option_without_letter = tokenizer(options_str).input_ids
    # print(option_without_letter)

    # print(options_str)


    # print("添加标签后的prompt：")
    # print(prompt)

    # print("分割后的prompt")
    # print(prompt.split('<image>'))

    # option的分隔符是<Options>
    prompt_split_img_list = prompt.split('<image>')
    # print("prompt_split_img_list是：")
    # print(prompt_split_img_list)
    prompt_split_opt_list = prompt_split_img_list[1].split('<Options>')
    # print("prompt_split_opt_list是：")
    # print(prompt_split_opt_list)
    # print(prompt_split_opt_list[1])  #这个是option的内容
    prompt_list=[]
    prompt_list.append(prompt_split_img_list[0])
    prompt_list.append(prompt_split_opt_list[0])
    prompt_list.append(prompt_split_opt_list[1])
    prompt_list.append(prompt_split_opt_list[2])

    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt_list]
    # print("未拼接的为：")
    # print(prompt_chunks)
    # print(len(prompt_chunks[0]))

    prompt_chunks_without_first = prompt_chunks[1:]
    # print("prompt_chunks_without_first是：")
    # print(prompt_chunks_without_first)

    input_ids_without_first = []
    offset = 0
    if len(prompt_chunks_without_first) > 0 and len(prompt_chunks_without_first[0]) > 0 and prompt_chunks_without_first[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids_without_first.append(prompt_chunks_without_first[0][0])

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
    for x in insert_separator(prompt_chunks_without_first, [optin_token_index] * (offset + 1)):
        input_ids_without_first.extend(x[offset:])

    # print("input_ids_without_first是：")
    # print(input_ids_without_first)

    prompt_chunks_with_first = []
    prompt_chunks_with_first.append(prompt_chunks[0])
    prompt_chunks_with_first.append(input_ids_without_first)

    input_ids = []
    offset = 0
    # print("tokenizer.bos_token_id = ", tokenizer.bos_token_id) # 1
    if len(prompt_chunks_with_first) > 0 and len(prompt_chunks_with_first[0]) > 0 and prompt_chunks_with_first[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks_with_first[0][0])

    for x in insert_separator(prompt_chunks_with_first, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    
    # print("拼接之后的input_ids是：")
    # print(input_ids)

    # print("未拼接option_without_letter的input_ids是：")
    # print(input_ids)
    input_ids.extend(option_without_letter[0:])
    # print("拼接option_without_letter之后的input_ids是：")
    # print(input_ids)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def tokenizer_image_token_split_option_and_question_and_context(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, optin_token_index=OPTION_TOKEN_INDEX, question_token_index=QUESTION_TOKEN_INDEX, context_token_index=CONTEXT_TOKEN_INDEX, return_tensors=None):
    # 在options的前后加上标签
    # prompt = prompt.replace("Options:", "<Options>Options:")
    prompt = prompt.replace("\nAnswer with the option's letter from the given choices directly", "<Options>\nAnswer with the option's letter from the given choices directly")
    # prompt = prompt.replace("Question:", "<Question>Question:")
    prompt = prompt.replace("\nOptions:", "<Question>\nOptions:")
    # prompt = prompt.replace("Context:", "<Context>Context:")
    prompt = prompt.replace("\nQuestion:", "<Context>\nQuestion:")

    # print("添加标签后的prompt：")
    # print(prompt)

    '''
    A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>
    Context: N/A<Context>
    Question: What is the name of the colony shown?<Question>
    Options: (A) Maryland (B) New Hampshire (C) Rhode Island (D) Vermont<Options>
    Answer with the option's letter from the given choices directly. ASSISTANT:
    '''
    # 奖option去掉开头的字母 (A)
    # 将文本分割成两部分
    options_part = prompt.split("Options: ")
    # 检查是否成功分割并获取第二部分
    if len(options_part) > 1:
        # 进一步分割以去除后面的部分
        prompt_option = options_part[1].split("<Options>\nAnswer with the option's letter from the given choices directly")[0]

    # Splitting the options using regular expression
    split_options = re.split(r' (?=\([A-Z]\))', prompt_option)

    # print(split_options)
    options_str = ""

    for opt in split_options:
        options_str += opt[3:]
    options_str = options_str[1:]

    option_without_letter = tokenizer(options_str).input_ids

    # print(option_without_letter)


    
    # print("分割后的prompt")
    # print(prompt.split('<image>'))

    prompt_split_img_list = prompt.split('<image>')
    # print("prompt_split_img_list是：")
    # print(prompt_split_img_list)

    prompt_split_context_list = prompt_split_img_list[1].split('<Context>')
    # print("prompt_split_context_list是：")
    # print(prompt_split_context_list)

    prompt_split_context_question_list = prompt_split_context_list[1].split('<Question>')
    # print("prompt_split_context_question_list是：")
    # print(prompt_split_context_question_list)

    prompt_split_context_question_option_list = prompt_split_context_question_list[1].split('<Options>')
    # print("prompt_split_context_question_option_list是：")
    # print(prompt_split_context_question_option_list)


    prompt_list=[]
    prompt_list.append(prompt_split_img_list[0])
    prompt_list.append(prompt_split_context_list[0])
    prompt_list.append(prompt_split_context_question_list[0])
    prompt_list.append(prompt_split_context_question_option_list[0])
    prompt_list.append(prompt_split_context_question_option_list[1])

    # print("prompt_list是：")
    # print(prompt_list)

    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt_list]
    # print("未拼接的prompt_chunks为：")
    # print(prompt_chunks)

    prompt_chunks_without_top_3 = prompt_chunks[3:]
    # print("prompt_chunks_without_top_3是：")
    # print(prompt_chunks_without_top_3)

    input_ids_without_top_3 = []
    offset = 0
    if len(prompt_chunks_without_top_3) > 0 and len(prompt_chunks_without_top_3[0]) > 0 and prompt_chunks_without_top_3[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids_without_top_3.append(prompt_chunks_without_top_3[0][0])

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
    for x in insert_separator(prompt_chunks_without_top_3, [optin_token_index] * (offset + 1)):
        input_ids_without_top_3.extend(x[offset:])

    # print("input_ids_without_top_3是：")
    # print(input_ids_without_top_3)


    prompt_chunks_without_top_2 = []
    prompt_chunks_without_top_2.append(prompt_chunks[2])
    prompt_chunks_without_top_2.append(input_ids_without_top_3)
    # print("prompt_chunks_without_top_2是：")
    # print(prompt_chunks_without_top_2)
    input_ids_without_top_2=[]
    offset = 0
    if len(prompt_chunks_without_top_2) > 0 and len(prompt_chunks_without_top_2[0]) > 0 and prompt_chunks_without_top_2[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids_without_top_2.append(prompt_chunks_without_top_2[0][0])

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
    for x in insert_separator(prompt_chunks_without_top_2, [question_token_index] * (offset + 1)):
        input_ids_without_top_2.extend(x[offset:])

    # print("input_ids_without_top_2是：")
    # print(input_ids_without_top_2)

    prompt_chunks_without_top_1 = []
    prompt_chunks_without_top_1.append(prompt_chunks[1])
    prompt_chunks_without_top_1.append(input_ids_without_top_2)
    # print("prompt_chunks_without_top_1是：")
    # print(prompt_chunks_without_top_1)

    input_ids_without_top_1=[]
    offset = 0
    if len(prompt_chunks_without_top_1) > 0 and len(prompt_chunks_without_top_1[0]) > 0 and prompt_chunks_without_top_1[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids_without_top_1.append(prompt_chunks_without_top_1[0][0])

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
    for x in insert_separator(prompt_chunks_without_top_1, [context_token_index] * (offset + 1)):
        input_ids_without_top_1.extend(x[offset:])

    # print("input_ids_without_top_1是：")
    # print(input_ids_without_top_1)


    prompt_chunks_with_all = []
    prompt_chunks_with_all.append(prompt_chunks[0])
    prompt_chunks_with_all.append(input_ids_without_top_1)

    input_ids = []
    offset = 0
    # print("tokenizer.bos_token_id = ", tokenizer.bos_token_id) # 1
    if len(prompt_chunks_with_all) > 0 and len(prompt_chunks_with_all[0]) > 0 and prompt_chunks_with_all[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks_with_all[0][0])

    for x in insert_separator(prompt_chunks_with_all, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    # print("拼接之后的input_ids是：")
    # print(input_ids)

    # option 在<Question>和<Option>之间, -600 和 -500 之间
    input_ids = [x for x in input_ids if x != -700]
    input_ids = [x if x != -600 else -500 for x in input_ids]

    # print("修改之后的input_ids是：")
    # print(input_ids)

    input_ids.extend(option_without_letter[0:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def tokenizer_image_token_split_question(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, question_token_index=QUESTION_TOKEN_INDEX, return_tensors=None):
    # 在options的前后加上标签
    prompt = prompt.replace("Question:", "<Question>Question:")
    prompt = prompt.replace("\nOptions:", "<Question>\nOptions:")
    # print(prompt)
    # 奖option去掉开头的字母 (A)
    # 将文本分割成两部分
    question_part = prompt.split("Question: ")
    # 检查是否成功分割并获取第二部分
    if len(question_part) > 1:
        # 进一步分割以去除后面的部分
        prompt_question = question_part[1].split("<Question>\nOptions:")[0]

    # Splitting the options using regular expression
    split_questions = re.split(r' (?=\([A-Z]\))', prompt_question)

    questions_str = ""

    questions_str = split_questions[0]
    # 使用str.maketrans创建一个翻译表，将标点符号映射到空格
    translation_table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    # 使用translate方法将字符串中的标点符号替换为一个空格
    questions_str = questions_str.translate(translation_table)

    # print("questions_str是：")
    # print(questions_str)

    question_token = tokenizer(questions_str).input_ids
    # print(option_without_letter)
    # print(question_token)

    # print(options_str)


    # print("添加标签后的prompt：")
    # print(prompt)

    # print("分割后的prompt")
    # print(prompt.split('<image>'))

    # option的分隔符是<Options>
    prompt_split_img_list = prompt.split('<image>')
    # print("prompt_split_img_list是：")
    # print(prompt_split_img_list)
    prompt_split_question_list = prompt_split_img_list[1].split('<Question>')

    prompt_list=[]
    prompt_list.append(prompt_split_img_list[0])
    prompt_list.append(prompt_split_question_list[0])
    prompt_list.append(prompt_split_question_list[1])
    prompt_list.append(prompt_split_question_list[2])
    # print("prompt_list是：")
    # print(prompt_list)

    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt_list]
    # print("未拼接的为：")
    # print(prompt_chunks)
    # print(len(prompt_chunks[0]))

    prompt_chunks_without_first = prompt_chunks[1:]
    # print("prompt_chunks_without_first是：")
    # print(prompt_chunks_without_first)

    input_ids_without_first = []
    offset = 0
    if len(prompt_chunks_without_first) > 0 and len(prompt_chunks_without_first[0]) > 0 and prompt_chunks_without_first[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids_without_first.append(prompt_chunks_without_first[0][0])

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
    for x in insert_separator(prompt_chunks_without_first, [question_token_index] * (offset + 1)):
        input_ids_without_first.extend(x[offset:])

    # print("input_ids_without_first是：")
    # print(input_ids_without_first)

    prompt_chunks_with_first = []
    prompt_chunks_with_first.append(prompt_chunks[0])
    prompt_chunks_with_first.append(input_ids_without_first)

    input_ids = []
    offset = 0
    # print("tokenizer.bos_token_id = ", tokenizer.bos_token_id) # 1
    if len(prompt_chunks_with_first) > 0 and len(prompt_chunks_with_first[0]) > 0 and prompt_chunks_with_first[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks_with_first[0][0])

    for x in insert_separator(prompt_chunks_with_first, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    
    # print("拼接之后的input_ids是：")
    # print(input_ids)

    # print("未拼接option_without_letter的input_ids是：")
    # print(input_ids)
    input_ids.extend(question_token[0:])
    # print("拼接option_without_letter之后的input_ids是：")
    # print(input_ids)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def tokenizer_image_token_split_C_and_Q_and_O(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, c_and_q_and_o_token_index=CONTEXT_AND_QUESTION_AND_OPTION_TOKEN_INDEX, return_tensors=None):
    # 在options的前后加上标签
    # prompt = prompt.replace("Context:", "<Context_and_Question_and_Option>Context:")
    prompt = prompt.replace("\nAnswer with the option's letter from the given choices directly", "<Context_and_Question_and_Option>\nAnswer with the option's letter from the given choices directly")
    

    # print("添加标签后的prompt：")
    # print(prompt)
    
    # print("分割后的prompt")
    # print(prompt.split('<image>'))

    # option的分隔符是<Options>
    prompt_split_img_list = prompt.split('<image>')
    # print("prompt_split_img_list是：")
    # print(prompt_split_img_list)

    prompt_split_opt_and_question_list = prompt_split_img_list[1].split('<Context_and_Question_and_Option>')
    # print("prompt_split_opt_and_question_list是：")
    # print(prompt_split_opt_and_question_list)
    # print(prompt_split_opt_list[1])  #这个是option的内容
    prompt_list=[]
    prompt_list.append(prompt_split_img_list[0])
    prompt_list.append(prompt_split_opt_and_question_list[0])
    prompt_list.append(prompt_split_opt_and_question_list[1])
    # prompt_list.append(prompt_split_opt_and_question_list[2])

    # print("prompt_list是：")
    # print(prompt_list)

    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt_list]
    # print("未拼接的prompt_chunks为：")
    # print(prompt_chunks)

    prompt_chunks_without_first = prompt_chunks[1:]
    # print("prompt_chunks_without_first是：")
    # print(prompt_chunks_without_first)

    input_ids_without_first = []
    offset = 0
    if len(prompt_chunks_without_first) > 0 and len(prompt_chunks_without_first[0]) > 0 and prompt_chunks_without_first[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids_without_first.append(prompt_chunks_without_first[0][0])

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
    for x in insert_separator(prompt_chunks_without_first, [c_and_q_and_o_token_index] * (offset + 1)):
        input_ids_without_first.extend(x[offset:])

    # print("input_ids_without_first是：")
    # print(input_ids_without_first)

    prompt_chunks_with_first = []
    prompt_chunks_with_first.append(prompt_chunks[0])
    prompt_chunks_with_first.append(input_ids_without_first)

    input_ids = []
    offset = 0
    # print("tokenizer.bos_token_id = ", tokenizer.bos_token_id) # 1
    if len(prompt_chunks_with_first) > 0 and len(prompt_chunks_with_first[0]) > 0 and prompt_chunks_with_first[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks_with_first[0][0])

    for x in insert_separator(prompt_chunks_with_first, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    
    # print("拼接之后的input_ids是：")
    # print(input_ids)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids



def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
