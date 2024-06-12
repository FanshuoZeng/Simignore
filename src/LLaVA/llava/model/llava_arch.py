#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from sklearn.decomposition import PCA
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, OPTION_TOKEN_INDEX, OPTION_AND_QUESTION_TOKEN_INDEX, QUESTION_TOKEN_INDEX, CONTEXT_TOKEN_INDEX, CONTEXT_AND_QUESTION_AND_OPTION_TOKEN_INDEX
import numpy
import random
import inspect
USE_TEST_7 = True
USE_ONLY_OPTION = True
USE_ONLY_QUESTION = False
USE_QUESTION_OPTION = False
USE_OPTION_WITHOUT_LETTER = True
USE_OPTION_QUESTION_CONTEXT = False
USE_C_AND_Q_AND_O = False

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        # print(images.shape) # torch.Size([1, 3, 336, 336])
        image_features = self.get_model().get_vision_tower()(images)
        # print(image_features.shape) # torch.Size([1, 576, 1024])
        image_features = self.get_model().mm_projector(image_features)
        # print(image_features.shape) # torch.Size([1, 576, 4096]) # torch.Size([1, 576, 5120])
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        # print("调用prepare_inputs_labels_for_multimodal()")
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            print("会进入这个分支吗1？") #不会
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            print("会进入这个分支吗2？") #不会
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            image_features = self.encode_images(images).to(self.device)
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            
            '''
            是否去除选项字母
            '''
            # USE_OPTION_WITHOUT_LETTER
            if USE_OPTION_WITHOUT_LETTER:
                indices_of_num_1 = (cur_input_ids == 1).nonzero(as_tuple=True)[0]

                if len(indices_of_num_1) > 1:
                        split_point_of_num_1 = indices_of_num_1[1]
                        tensor_1 = cur_input_ids[:split_point_of_num_1]
                        tensor_2 = cur_input_ids[split_point_of_num_1:]
                        # 去除1
                        # tensor_2 = tensor_2[1:]
                        cur_input_ids = tensor_1
                        cur_options_without_letter_ids = tensor_2
                else:
                    print("error:Less than two 1 token")

            '''
            测试7：开始
            '''
            # OPTION_TOKEN_INDEX
            # OPTION_AND_QUESTION_TOKEN_INDEX
            USE_WITCH = OPTION_TOKEN_INDEX
            if USE_QUESTION_OPTION:
                USE_WITCH=OPTION_AND_QUESTION_TOKEN_INDEX
            elif USE_ONLY_QUESTION:
                USE_WITCH=QUESTION_TOKEN_INDEX
            if USE_TEST_7:
                if USE_ONLY_OPTION or USE_OPTION_QUESTION_CONTEXT or USE_ONLY_QUESTION:
                    # 获取OPTION_TOKEN_INDEX的索引
                    indices_of_options = (cur_input_ids == USE_WITCH).nonzero(as_tuple=True)[0]
                    # 如果有至少两个OPTION_TOKEN_INDEX
                    if len(indices_of_options) > 1:
                        # 获取第一个和第二个OPTION_TOKEN_INDEX之间的元素
                        cur_options_ids = cur_input_ids[indices_of_options[0]+1:indices_of_options[1]]
                        # # 前面补1
                        # cur_options_ids_1 = torch.tensor([1])
                        # cur_options_ids_1 = cur_options_ids_1.to('cuda:0')
                        # cur_options_ids = cur_options_ids.to('cuda:0')
                        # cur_options_ids = torch.cat((cur_options_ids_1, cur_options_ids), dim=0)
                    else:
                        print("error:Less than two -500 token")
                        cur_options_ids = torch.tensor([])  # 没有足够的OPTION_TOKEN_INDEX
                    cur_input_ids = cur_input_ids[cur_input_ids != USE_WITCH]
            # print(cur_input_ids)
            # C_and_Q_and_O
            if USE_C_AND_Q_AND_O:
                indices_of_cqo = (cur_input_ids == CONTEXT_AND_QUESTION_AND_OPTION_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                indices_of_iamge = (cur_input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                if len(indices_of_cqo) > 0:
                        cur_cqo_ids = cur_input_ids[indices_of_iamge[0]+1:indices_of_cqo[0]]
                else:
                    print("error:Less than one -900 token")
                    cur_options_ids = torch.tensor([])  # 没有足够的OPTION_TOKEN_INDEX
                cur_input_ids = cur_input_ids[cur_input_ids != CONTEXT_AND_QUESTION_AND_OPTION_TOKEN_INDEX]
                # print(cur_input_ids)
                # print(cur_cqo_ids)


            '''
            测试7：结束
            '''


            if num_images == 0:
                print("num_images == 0成立")
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            '''
            测试7：开始
            '''
            if USE_TEST_7:
                cur_options_embeds = self.get_model().embed_tokens(cur_options_ids)
            if USE_OPTION_WITHOUT_LETTER:
                cur_options_embeds = self.get_model().embed_tokens(cur_options_without_letter_ids)
            if USE_C_AND_Q_AND_O:
                cur_cqo_embeds = self.get_model().embed_tokens(cur_cqo_ids)
            '''
            测试7：结束
            '''
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                # print("进入循环:::::", i) # 两次循环
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    # 第一次循环时进入,即i=0时
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))



            '''在一个余弦度量空间测量'''
            if False:

                image_embedding_np = cur_new_input_embeds[1].cpu().numpy()
                prompt_embedding_np = cur_new_input_embeds[2].cpu().numpy()

                pca = PCA(n_components=2)
                image_embedding_2d = pca.fit_transform(image_embedding_np)
                prompt_embedding_2d = pca.fit_transform(prompt_embedding_np)
                
                def cosine_similarity(A, B):
                    dot_product = numpy.dot(A, B.T)
                    norms = numpy.linalg.norm(A, axis=1, keepdims=True) * numpy.linalg.norm(B, axis=1)
                    similarity = dot_product / norms
                    return similarity
                # 假设这些是您的数据
                prompt_embedding_2d = numpy.random.rand(221, 2)  # 生成随机的 prompt embeddings
                image_embedding_2d = numpy.random.rand(576, 2)  # 生成随机的 image embeddings

                # 计算余弦相似度
                similarities = cosine_similarity(image_embedding_2d, prompt_embedding_2d)

                # 求每个image_embedding与所有prompt_embedding的最大相似度
                max_similarities = numpy.max(similarities, axis=1)

                KK = 452
                # 对这些最大相似度进行排序，并取前452个
                top_k_indices = numpy.argsort(-max_similarities)[:KK]
                # print("Indices of the top K=452 image embeddings based on cosine similarity:")
                # print(top_k_indices)
                # print(len(top_k_indices))
                new_unique_x_values = [i + 35 for i in top_k_indices]
                '''把 unique_x_values 写入文件，用于聚类分析'''
                # with open('/hy-tmp/FastV/top_k_indices.txt', 'w') as file:
                #     # 将列表转换为字符串，然后写入文件
                #     # 这里使用 '\n' 作为分隔符，每个元素占一行
                #     for item in top_k_indices:
                #         file.write(str(item) + '\n')



            '''
            测试6：计算537 * 362个相似度
            '''
            if USE_TEST_7:
                test_1_image_embeds = cur_new_input_embeds[1]
                # test_1_text_embeds = cur_new_input_embeds[2]
                test_1_text_embeds = cur_options_embeds
                # 正则化
                test_1_image_embeds = F.normalize(test_1_image_embeds, p=2, dim=1)
                test_1_text_embeds = F.normalize(test_1_text_embeds, p=2, dim=1)  
                USE_PCA = False    
                if USE_PCA:
                    '''PCA降维'''
                    # 初始化PCA对象，设置主成分数为2
                    pca = PCA(n_components=2)
                    test_1_image_embeds_cpu = test_1_image_embeds.cpu()
                    test_1_text_embeds_cpu = test_1_text_embeds.cpu()
                    # 然后将其转换为NumPy数组
                    test_1_image_embeds_np = test_1_image_embeds_cpu.numpy()
                    test_1_text_embeds_np = test_1_text_embeds_cpu.numpy()
                    # 现在可以安全地使用PCA.fit_transform方法了
                    pca = PCA(n_components=2)
                    test_1_image_embeds_reduced = pca.fit_transform(test_1_image_embeds_np)
                    test_1_text_embeds_reduced = pca.fit_transform(test_1_text_embeds_np)
                    test_1_image_embeds_reduced_tensor = torch.from_numpy(test_1_image_embeds_reduced)
                    test_1_text_embeds_reduced_tensor = torch.from_numpy(test_1_text_embeds_reduced)
                    similarity_matrix = torch.mm(test_1_image_embeds_reduced_tensor, test_1_text_embeds_reduced_tensor.transpose(0, 1))

                # 计算余弦相似度矩阵
                # 正确的操作是将 image_tensor_norm 乘以 text_tensor_norm 的转置
                similarity_matrix = torch.mm(test_1_image_embeds, test_1_text_embeds.transpose(0, 1))
                # 将 similarity_matrix 展平成一维张量
                flat_similarity = similarity_matrix.view(-1)

                # 计算绝对值
                # flat_similarity = abs(flat_similarity)

                # 使用 torch.sort 获取排序后的值和索引，设置 descending=True 来获取降序排序
                sorted_values, sorted_indices = torch.sort(flat_similarity, descending=True)
                # print(sorted_values)
                # print("len(sorted_values) = ", len(sorted_values))
                # print(test_1_image_embeds.shape[0] * test_1_text_embeds.shape[0])
                # print("sorted_values[] = ",sorted_values[576*362 // 2])
                # 将 sorted_indices 转换为列表
                sorted_indices_list = sorted_indices.tolist()

                # 如果需要将一维索引转换为原始的二维索引位置，可以这样做：
                sorted_indices_2d_list = [None] * len(sorted_indices_list)
                rows, cols = similarity_matrix.shape
                for i, idx in enumerate(sorted_indices_list):
                    row = idx // cols
                    col = idx % cols
                    sorted_indices_2d_list[i] = (row, col)
                # sorted_values 现在包含了从大到小排序的相似度值
                # sorted_indices_list 现在是一个列表，包含了排序后的一维索引
                # sorted_indices_2d_list 现在是一个列表的列表，每个内部列表是一个元组 (row, col)，表示 sorted_similarity 在原始二维张量中的行列索引

                # print("Sorted values:", sorted_values.tolist()[0])
                # print("Sorted indices (1D):", len(sorted_indices_list))
                # print("Sorted indices (2D):", sorted_indices_2d_list)
                # 假设 sorted_indices_2d_list 是一个包含 (x, y) 元组的列表
                # 假设 k 是你想要提取的元组数量

                # 假设 sorted_indices_2d_list 是一个包含 (x, y) 元组的列表
                # 假设 k2 是你想要收集的不重复 x 值的数量

                # 初始化一个空列表来存储不重复的 x 值
                unique_x_values = []

                # 设置步长，即每个第i个元组，可以根据实际情况来设置
                step = 1  # 例如，如果step=2，则会选择第1个、第3个、第5个等
                k2 = 452
                # 循环直到 unique_x_values 的长度等于 k2 或者达到 sorted_indices_2d_list 的末尾
                i_index = 1
                while len(unique_x_values) < k2 and i_index < len(sorted_indices_2d_list) and sorted_values[i_index] >= -1:
                # while i_index < len(sorted_indices_2d_list) and sorted_values[i_index] >= 0:
                    # 取出当前索引 i 对应的元组
                    current_tuple = sorted_indices_2d_list[i_index]

                    # 提取 x 值
                    x_value = current_tuple[0]

                    # 只有当 x 值尚未在 unique_x_values 中时，才将其添加
                    if x_value not in unique_x_values:
                        unique_x_values.append(x_value)

                    # 增加 i 的值，按照设定的步长进行
                    i_index += step
                
                # unique_x_values = unique_x_values[-452:] # 取最后的
                # unique_x_values = unique_x_values[62:62+452] # 取中间的
                
                '''把 unique_x_values 写入文件，用于聚类分析'''
                with open('/hy-tmp/FastV/unique_x_values.txt', 'w') as file:
                    # 将列表转换为字符串，然后写入文件
                    # 这里使用 '\n' 作为分隔符，每个元素占一行
                    for item in unique_x_values:
                        file.write(str(item) + '\n')



                # 打印结果
                # print(len(unique_x_values))
                new_unique_x_values = [x + 35 for x in unique_x_values]

                # 一些分析
                # print("len(new_unique_x_values) = ", len(new_unique_x_values))
                unique_x_values_sorted = sorted(unique_x_values)
                # print(unique_x_values)
                # print(sorted_values[i_index - step])

                # 将列表转换为numpy数组
                unique_x_values_sorted_array = numpy.array(unique_x_values_sorted)
                # 计算标准差
                std_dev = numpy.std(unique_x_values_sorted_array)
                # print("标准差:", std_dev)

                # print(sorted_values)

                # 创建一个新的张量，初始时与image_tensor相同
                # preserved_image_tensor = torch.zeros_like(test_1_image_embeds)
                # # 只在top_indices指定的位置填入原image_tensor中的数据，其余位置保持为零
                # preserved_image_tensor[unique_x_values] = cur_new_input_embeds[1][unique_x_values]

                # cur_new_input_embeds[1] = preserved_image_tensor
            '''
            测试6：结束。比较成功。
            '''




            '''
            这是忽略text token 的实验
            '''

            if USE_TEST_7 and False:
                test_1_image_embeds = cur_new_input_embeds[1]
                test_1_text_embeds = cur_new_input_embeds[2]
                # test_1_text_embeds = cur_options_embeds
                # 正则化
                test_1_image_embeds = F.normalize(test_1_image_embeds, p=2, dim=1)
                test_1_text_embeds = F.normalize(test_1_text_embeds, p=2, dim=1)            
                # 计算余弦相似度矩阵
                # 正确的操作是将 image_tensor_norm 乘以 text_tensor_norm 的转置
                similarity_matrix = torch.mm(test_1_text_embeds, test_1_image_embeds.transpose(0, 1))
                # 将 similarity_matrix 展平成一维张量
                flat_similarity = similarity_matrix.view(-1)

                # 计算绝对值
                # flat_similarity = abs(flat_similarity)

                # 使用 torch.sort 获取排序后的值和索引，设置 descending=True 来获取降序排序
                sorted_values, sorted_indices = torch.sort(flat_similarity, descending=True)
                # print(sorted_values)
                # print("len(sorted_values) = ", len(sorted_values))
                # print(test_1_image_embeds.shape[0] * test_1_text_embeds.shape[0])
                # print("sorted_values[] = ",sorted_values[576*362 // 2])
                # 将 sorted_indices 转换为列表
                sorted_indices_list = sorted_indices.tolist()

                # 如果需要将一维索引转换为原始的二维索引位置，可以这样做：
                sorted_indices_2d_list = [None] * len(sorted_indices_list)
                rows, cols = similarity_matrix.shape
                for i, idx in enumerate(sorted_indices_list):
                    row = idx // cols
                    col = idx % cols
                    sorted_indices_2d_list[i] = (row, col)
                # sorted_values 现在包含了从大到小排序的相似度值
                # sorted_indices_list 现在是一个列表，包含了排序后的一维索引
                # sorted_indices_2d_list 现在是一个列表的列表，每个内部列表是一个元组 (row, col)，表示 sorted_similarity 在原始二维张量中的行列索引

                # print("Sorted values:", sorted_values.tolist()[0])
                # print("Sorted indices (1D):", len(sorted_indices_list))
                # print("Sorted indices (2D):", sorted_indices_2d_list)
                # 假设 sorted_indices_2d_list 是一个包含 (x, y) 元组的列表
                # 假设 k 是你想要提取的元组数量

                # 假设 sorted_indices_2d_list 是一个包含 (x, y) 元组的列表

                # 假设 k2 是你想要收集的不重复 x 值的数量

                # 初始化一个空列表来存储不重复的 x 值
                unique_x_values = []


                # 设置步长，即每个第i个元组，可以根据实际情况来设置
                step = 1  # 例如，如果step=2，则会选择第1个、第3个、第5个等

                k2 = round(test_1_text_embeds.shape[0] * 0.95)
                # k2 = round(test_1_image_embeds.shape[0] * 0.875)
                
                # print("k2 = ", k2)
                # print("test_1_text_embeds= ", test_1_text_embeds.shape[0])
                # print("k2 = ", k2)
                # 循环直到 unique_x_values 的长度等于 k2 或者达到 sorted_indices_2d_list 的末尾
                i_index = 0
                while len(unique_x_values) < k2 and i_index < len(sorted_indices_2d_list) and sorted_values[i_index] >= -1:
                # while i_index < len(sorted_indices_2d_list) and sorted_values[i_index] >= 0:
                    # 取出当前索引 i 对应的元组
                    current_tuple = sorted_indices_2d_list[i_index]
                    # print(current_tuple)
                    # 提取 y 值
                    x_value = current_tuple[0]

                    # 只有当 x 值尚未在 unique_x_values 中时，才将其添加

                    if x_value not in unique_x_values:
                        unique_x_values.append(x_value)

                    # 增加 i 的值，按照设定的步长进行
                    i_index += step
                
                # unique_x_values = unique_x_values[-452:] # 取最后的
                # unique_x_values = unique_x_values[62:62+452] # 取中间的
                
                # 打印结果
                # print(len(unique_x_values))
                new_unique_x_values = [x + 35 + 576 for x in unique_x_values]
                # new_unique_x_values = [x + 35  for x in unique_x_values]

            '''
            忽略 text token 实验结束
            '''

            '''K-means聚类image embedding'''
            embedding_tensor = cur_new_input_embeds[1]
            # 将 PyTorch 张量转换为 NumPy 数组
            embedding_np = embedding_tensor.cpu().numpy()
            # 指定要写入的文件路径
            file_path = '/hy-tmp/FastV/image_embedding.txt'

            # 将数据写入 txt 文件
            with open(file_path, 'w') as f:
                for row in embedding_np:
                    numpy.savetxt(f, row[None], fmt='%.6f')

            '''prompt'''
            prompt_embedding_tensor = cur_new_input_embeds[2]
            # 将 PyTorch 张量转换为 NumPy 数组
            # embedding_np = embedding_tensor.numpy()
            prompt_embedding_np = prompt_embedding_tensor.cpu().numpy()
            # 指定要写入的文件路径
            file_path = '/hy-tmp/FastV/prompt_embedding.txt'

            # 将数据写入 txt 文件
            with open(file_path, 'w') as f:
                for row in prompt_embedding_np:
                    numpy.savetxt(f, row[None], fmt='%.6f')





            '''
            忽略text token测试-2
            '''
            if USE_TEST_7 and USE_C_AND_Q_AND_O:
                test_1_image_embeds = cur_new_input_embeds[1]
                # test_1_text_embeds = cur_new_input_embeds[2]
                # test_1_text_embeds = cur_options_embeds
                test_1_text_embeds = cur_cqo_embeds
                # 正则化
                test_1_image_embeds = F.normalize(test_1_image_embeds, p=2, dim=1)
                test_1_text_embeds = F.normalize(test_1_text_embeds, p=2, dim=1)            
                # 计算余弦相似度矩阵
                # 正确的操作是将 image_tensor_norm 乘以 text_tensor_norm 的转置
                similarity_matrix = torch.mm(test_1_text_embeds, test_1_image_embeds.transpose(0, 1))
                # 将 similarity_matrix 展平成一维张量
                flat_similarity = similarity_matrix.view(-1)

                # 计算绝对值
                # flat_similarity = abs(flat_similarity)

                # 使用 torch.sort 获取排序后的值和索引，设置 descending=True 来获取降序排序
                sorted_values, sorted_indices = torch.sort(flat_similarity, descending=True)
                # print(sorted_values)
                # print("len(sorted_values) = ", len(sorted_values))
                # print(test_1_image_embeds.shape[0] * test_1_text_embeds.shape[0])
                # print("sorted_values[] = ",sorted_values[576*362 // 2])
                # 将 sorted_indices 转换为列表
                sorted_indices_list = sorted_indices.tolist()

                # 如果需要将一维索引转换为原始的二维索引位置，可以这样做：
                sorted_indices_2d_list = [None] * len(sorted_indices_list)
                rows, cols = similarity_matrix.shape
                for i, idx in enumerate(sorted_indices_list):
                    row = idx // cols
                    col = idx % cols
                    sorted_indices_2d_list[i] = (row, col)
                # sorted_values 现在包含了从大到小排序的相似度值
                # sorted_indices_list 现在是一个列表，包含了排序后的一维索引
                # sorted_indices_2d_list 现在是一个列表的列表，每个内部列表是一个元组 (row, col)，表示 sorted_similarity 在原始二维张量中的行列索引

                # print("Sorted values:", sorted_values.tolist()[0])
                # print("Sorted indices (1D):", len(sorted_indices_list))
                # print("Sorted indices (2D):", sorted_indices_2d_list)
                # 假设 sorted_indices_2d_list 是一个包含 (x, y) 元组的列表
                # 假设 k 是你想要提取的元组数量

                # 假设 sorted_indices_2d_list 是一个包含 (x, y) 元组的列表
                # 假设 k2 是你想要收集的不重复 x 值的数量

                # 初始化一个空列表来存储不重复的 x 值
                unique_x_values = []

                # 设置步长，即每个第i个元组，可以根据实际情况来设置
                step = 1  # 例如，如果step=2，则会选择第1个、第3个、第5个等
                k2 = round(test_1_text_embeds.shape[0] * 0.95)
                # k2 = 432
                # 循环直到 unique_x_values 的长度等于 k2 或者达到 sorted_indices_2d_list 的末尾
                i_index = 1
                while len(unique_x_values) < k2 and i_index < len(sorted_indices_2d_list) and sorted_values[i_index] >= -1:
                # while i_index < len(sorted_indices_2d_list) and sorted_values[i_index] >= 0:
                    # 取出当前索引 i 对应的元组
                    current_tuple = sorted_indices_2d_list[i_index]

                    # 提取 x 值
                    x_value = current_tuple[0]
                    # print(x_value)

                    # 只有当 x 值尚未在 unique_x_values 中时，才将其添加
                    if x_value not in unique_x_values:
                        unique_x_values.append(x_value)

                    # 增加 i 的值，按照设定的步长进行
                    i_index += step
                # print("i_index = ", i_index)
                # unique_x_values = unique_x_values[-452:] # 取最后的
                # unique_x_values = unique_x_values[62:62+452] # 取中间的
                

                # 打印结果
                # print(len(unique_x_values))
                new_unique_x_values = [x + 35 + 576 for x in unique_x_values]


            '''
            忽略text token测试-2 结束 
            '''


            '''
            使用其他相似度算法进行实验
            '''
            if USE_TEST_7 and False:
                test_1_image_embeds = cur_new_input_embeds[1]
                # test_1_text_embeds = cur_new_input_embeds[2]
                test_1_text_embeds = cur_options_embeds


                # 正则化
                test_1_image_embeds = F.normalize(test_1_image_embeds, p=2, dim=1)
                test_1_text_embeds = F.normalize(test_1_text_embeds, p=2, dim=1)            
                # 计算余弦相似度矩阵
                # 正确的操作是将 image_tensor_norm 乘以 text_tensor_norm 的转置
                similarity_matrix = torch.mm(test_1_image_embeds, test_1_text_embeds.transpose(0, 1))

                # # 计算距离矩阵
                # similarity_matrix = torch.cdist(test_1_image_embeds, test_1_text_embeds)



                # # 曼哈顿距离
                # def manhattan_distance(image_embeddings, text_embeddings):
                #     # 计算曼哈顿距离
                #     distance_matrix = torch.abs(image_embeddings.unsqueeze(1) - text_embeddings.unsqueeze(0))
                #     # 求和得到曼哈顿距离
                #     distance_matrix = torch.sum(distance_matrix, dim=-1)
                #     return distance_matrix
                # # 计算距离矩阵
                # similarity_matrix = manhattan_distance(test_1_image_embeds, test_1_text_embeds)



                # 将 similarity_matrix 展平成一维张量
                flat_similarity = similarity_matrix.view(-1)

                # 计算绝对值
                # flat_similarity = abs(flat_similarity)

                # 使用 torch.sort 获取排序后的值和索引，设置 descending=True 来获取降序排序
                sorted_values, sorted_indices = torch.sort(flat_similarity, descending=True)
                # print(sorted_values)
                # print("len(sorted_values) = ", len(sorted_values))
                # print(test_1_image_embeds.shape[0] * test_1_text_embeds.shape[0])
                # print("sorted_values[] = ",sorted_values[576*362 // 2])
                # 将 sorted_indices 转换为列表
                sorted_indices_list = sorted_indices.tolist()

                # 如果需要将一维索引转换为原始的二维索引位置，可以这样做：
                sorted_indices_2d_list = [None] * len(sorted_indices_list)
                rows, cols = similarity_matrix.shape
                for i, idx in enumerate(sorted_indices_list):
                    row = idx // cols
                    col = idx % cols
                    sorted_indices_2d_list[i] = (row, col)
                # sorted_values 现在包含了从大到小排序的相似度值
                # sorted_indices_list 现在是一个列表，包含了排序后的一维索引
                # sorted_indices_2d_list 现在是一个列表的列表，每个内部列表是一个元组 (row, col)，表示 sorted_similarity 在原始二维张量中的行列索引

                # print("Sorted values:", sorted_values.tolist()[0])
                # print("Sorted indices (1D):", len(sorted_indices_list))
                # print("Sorted indices (2D):", sorted_indices_2d_list)
                # 假设 sorted_indices_2d_list 是一个包含 (x, y) 元组的列表
                # 假设 k 是你想要提取的元组数量

                # 假设 sorted_indices_2d_list 是一个包含 (x, y) 元组的列表
                # 假设 k2 是你想要收集的不重复 x 值的数量

                # 初始化一个空列表来存储不重复的 x 值
                unique_x_values = []

                # 设置步长，即每个第i个元组，可以根据实际情况来设置
                step = 1  # 例如，如果step=2，则会选择第1个、第3个、第5个等
                k2 = 452
                # 循环直到 unique_x_values 的长度等于 k2 或者达到 sorted_indices_2d_list 的末尾
                i_index = 0
                while len(unique_x_values) < k2 and i_index < len(sorted_indices_2d_list) and sorted_values[i_index] >= -1:
                # while i_index < len(sorted_indices_2d_list) and sorted_values[i_index] >= 0:
                    # 取出当前索引 i 对应的元组
                    current_tuple = sorted_indices_2d_list[i_index]

                    # 提取 x 值
                    x_value = current_tuple[0]

                    # 只有当 x 值尚未在 unique_x_values 中时，才将其添加
                    if x_value not in unique_x_values:
                        unique_x_values.append(x_value)

                    # 增加 i 的值，按照设定的步长进行
                    i_index += step
                
                # unique_x_values = unique_x_values[-452:] # 取最后的
                # unique_x_values = unique_x_values[62:62+452] # 取中间的
                

                # 打印结果
                # print(len(unique_x_values))
                new_unique_x_values = [x + 35 for x in unique_x_values]

                # 一些分析
                # print("len(new_unique_x_values) = ", len(new_unique_x_values))
                unique_x_values_sorted = sorted(unique_x_values)
                # print(unique_x_values)
                # print(sorted_values[i_index - step])

                # 将列表转换为numpy数组
                unique_x_values_sorted_array = numpy.array(unique_x_values_sorted)
                # 计算标准差
                std_dev = numpy.std(unique_x_values_sorted_array)
                # print("标准差:", std_dev)

                # print(sorted_values)

                # 创建一个新的张量，初始时与image_tensor相同
                # preserved_image_tensor = torch.zeros_like(test_1_image_embeds)
                # # 只在top_indices指定的位置填入原image_tensor中的数据，其余位置保持为零
                # preserved_image_tensor[unique_x_values] = cur_new_input_embeds[1][unique_x_values]

                # cur_new_input_embeds[1] = preserved_image_tensor















            cur_new_input_embeds = torch.cat(cur_new_input_embeds) # 拼接起来?
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        # print("tokenizer_model_max_length是:")
        # print(tokenizer_model_max_length) # None
        if tokenizer_model_max_length is not None:
            print("进入tokenizer_model_max_length分支") # 不进入
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        # print("max_len = ", max_len) # 973
        batch_size = len(new_input_embeds)
        # print("batch_size = ", batch_size) # 1

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            # print("进入循环：i = ", i) # 只进入一次
            cur_len = cur_new_embed.shape[0] # 973
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                print("进入if分支~~~~~~~~~") #没有进入
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                # print("进入else分支~~~~~~~~~") #进入
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    # print("cur_len = ", cur_len)
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        '''
        测试6：开始
        '''
        

        if USE_TEST_7:
            ''' 这个是忽略image token 的实验'''
            for index in range(35, 35+576):
                if index not in new_unique_x_values:
                    attention_mask[0][index] = 0


        # print("attention_mask.shape = ", attention_mask.shape) # torch.Size([1, 675])

        # for index in range(35+288, 35+576):
        #     attention_mask[0][index] = 0

        '''
        Cluster 1 has 16 occurrences from ignore_x_values
        Cluster 2 has 2 occurrences from ignore_x_values
        Cluster 0 has 35 occurrences from ignore_x_values
        Cluster 3 has 67 occurrences from ignore_x_values
        Cluster 4 has 4 occurrences from ignore_x_values
        '''
        

        # 坏蛋 token list
        list_bad = [4, 9, 14, 56, 57, 72, 80, 89, 90, 93, 95, 96, 111, 113, 114, 115, 140, 145, 146, 177, 230, 238, 254, 271, 295, 327, 333, 341, 350, 351, 352, 361, 362, 363, 365, 366, 367, 368, 375, 377, 378, 379, 380, 381, 382, 456, 459, 472, 476, 482, 495, 496, 498, 509, 521]

        # c0
        list0 = [16, 85, 86, 100, 101, 102, 106, 107, 108, 111, 126, 127, 128, 131, 132, 133, 134, 142, 148, 149, 151, 158, 171, 172, 194, 230, 231, 233, 235, 236, 237, 238, 241, 248, 249, 251, 255, 256, 257, 259, 260, 262, 263, 265, 271, 273, 286, 295, 297, 298, 310, 314, 315, 317, 318, 319, 320, 321, 322, 336, 340, 341, 343, 346, 351, 358, 360, 361, 362, 363, 364, 366, 367, 370, 371, 375, 376, 378, 382, 385, 387, 389, 391, 392, 393, 394, 402, 403, 406, 418, 419, 430, 435, 441, 442, 446, 447, 453, 454, 464, 466, 467, 468, 469, 470, 471, 473, 474, 475, 478]

                
        # c1
        list1 = [3, 4, 9, 15, 19, 20, 21, 26, 27, 28, 30, 33, 35, 36, 37, 39, 40, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 55, 58, 59, 60, 61, 62, 63, 64, 65, 69, 70, 71, 73, 74, 75, 76, 77, 78, 81, 87, 88, 89, 91, 92, 94, 99, 105, 121, 129, 130, 144, 147, 170, 173, 190, 195, 219, 264, 267, 279, 292, 308, 327, 337, 339, 347, 352, 356, 380, 381, 404, 481, 482, 483, 484, 485, 491, 492, 493, 494, 495, 496, 499, 500, 506, 507, 508, 510, 516, 517, 519, 520, 523, 525, 530, 531, 532, 533, 535, 540, 543, 544, 547, 548, 549, 554, 555, 556, 557, 560, 562, 563, 564, 565, 567, 568, 569, 571, 572, 573]



        list2 = [67, 524]


        list3 = [6, 68, 104, 109, 110, 112, 113, 114, 115, 116, 117, 118, 122, 123, 124, 125, 135, 136, 137, 138, 139, 140, 141, 143, 145, 146, 152, 153, 155, 156, 157, 159, 160, 161, 162, 163, 164, 165, 168, 169, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 188, 189, 191, 192, 196, 197, 199, 200, 201, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 221, 222, 223, 224, 225, 226, 227, 229, 232, 234, 239, 240, 242, 243, 244, 245, 246, 247, 250, 252, 253, 254, 258, 261, 268, 269, 270, 272, 274, 275, 276, 277, 278, 280, 281, 282, 283, 284, 287, 288, 290, 291, 293, 294, 296, 300, 301, 303, 304, 305, 306, 307, 309, 311, 312, 316, 323, 324, 325, 326, 328, 329, 330, 331, 332, 333, 335, 338, 342, 344, 345, 348, 349, 350, 353, 354, 355, 357, 359, 365, 368, 369, 374, 383, 384, 386, 388, 390, 395, 396, 397, 398, 399, 400, 401, 405, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 432, 433, 434, 436, 437, 438, 439, 440, 443, 444, 445, 448, 449, 450, 451, 452, 455, 456, 457, 459, 460, 462, 463, 465, 472, 476, 477, 479, 501, 502, 558]


        list4 = [0, 1, 2, 5, 7, 8, 10, 11, 12, 13, 14, 17, 18, 22, 23, 24, 25, 29, 31, 32, 34, 38, 41, 42, 47, 54, 56, 57, 66, 72, 79, 80, 82, 83, 84, 90, 93, 95, 96, 97, 98, 103, 119, 120, 150, 154, 166, 167, 187, 193, 198, 202, 218, 220, 228, 266, 285, 289, 299, 302, 313, 334, 372, 373, 377, 379, 407, 431, 458, 461, 480, 486, 487, 488, 489, 490, 497, 498, 503, 504, 505, 509, 511, 512, 513, 514, 515, 518, 521, 522, 526, 527, 528, 529, 534, 536, 537, 538, 539, 541, 542, 545, 546, 550, 551, 552, 553, 559, 561, 566, 570, 574, 575]




        # list_need = [i + 35 for i in list0]
        # for index in list_need:
        #         attention_mask[0][index] = 0
        
        # list_need = [i + 35 for i in list1]
        # for index in list_need:
        #         attention_mask[0][index] = 0


        # list_need = [i + 35 for i in list2]
        # for index in list_need:
        #         attention_mask[0][index] = 0


        # list_need = [i + 35 for i in list3]
        # for index in list_need:
        #         attention_mask[0][index] = 0

        # list_need = [i + 35 for i in list4]
        # for index in list_need:
        #         attention_mask[0][index] = 0



        if USE_TEST_7 and False:
            '''忽略 text token 的实验'''
            for index in range(35+576, 35+576+test_1_text_embeds.shape[0]):
            # for index in range(35, 35+576):
                    if index not in new_unique_x_values:
                        # print(index)
                        attention_mask[0][index] = 0

        # 随机忽略5%的text token
        # random_values = random.sample(range(35 + 576, 35 + 576 + cur_cqo_embeds.shape[0]), round(cur_cqo_embeds.shape[0] * 0.05))
        # for item in random_values:
        #     attention_mask[0][item] = 0
        
        '''
        随机忽略image token的实验
        '''
        # # 使用random.sample()随机选择124个不同的值
        # random_values = random.sample(range(35, 35 + 576), 124)
        # for item in random_values:
        #     attention_mask[0][item] = 0
        '''
        测试6：结束
        '''

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False


