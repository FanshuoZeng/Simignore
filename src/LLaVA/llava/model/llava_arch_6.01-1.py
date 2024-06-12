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
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, OPTION_TOKEN_INDEX, OPTION_AND_QUESTION_TOKEN_INDEX, QUESTION_TOKEN_INDEX, CONTEXT_TOKEN_INDEX
import numpy
import random
import inspect
USE_TEST_7 = True
USE_ONLY_OPTION = True
USE_ONLY_QUESTION = False
USE_QUESTION_OPTION = False
USE_OPTION_WITHOUT_LETTER = True
USE_OPTION_QUESTION_CONTEXT = False

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
        
        image_features = self.get_model().get_vision_tower()(images)
        # print("image_features 1是:")
        # print(image_features.shape) # torch.Size([1, 576, 1024])
        # print(image_features)
        image_features = self.get_model().mm_projector(image_features)
        # print("image_features 2是:")
        # print(image_features.shape) # torch.Size([1, 576, 4096])
        # print(image_features)
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
            # print("会进入这个分支吗3？") #会
            # print("images是：")
            # print("type(images)是：", type(images))
            # images_list = images.tolist()
            # print(len(images_list))
            # print(len(images_list[0]))
            # print(len(images_list[0][0]))
            # print(len(images_list[0][0][0]))

            # images的维度是 1*3*336*336
            # image_features的维度 1*576*4096
            # print("用的哪个模型：",self.get_model()) # LlavaLlamaModel
            
            image_features = self.encode_images(images).to(self.device)
            # print("image_features是：")
            # print(image_features)
            # image_features_list = image_features.tolist()
            # print(len(image_features_list))
            # print(len(image_features_list[0]))
            # print(len(image_features_list[0][0]))
            # print(image_features_list[0][0])

            # print("input_ids是：")
            # print(input_ids.shape)
            # print(input_ids)

        # IMAGE_TOKEN_LENGTH = image_features.shape[-2]
        # ATTENTION_RANK = 144
        
        # keep_image_token_id = list(range(IMAGE_TOKEN_LENGTH))
        # random.shuffle(keep_image_token_id)
        # keep_image_token_id = sorted(keep_image_token_id[:ATTENTION_RANK])
        # image_features = image_features[:, keep_image_token_id, :]

        

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
        # print("input_ids是:::::::::::::::::::::::::")
        # print(input_ids[0].shape) #torch.Size([398])
        # print(input_ids)
        # print("labels是:")
        # print(labels[0].shape) # torch.Size([398])
        # print(labels) #全是-100
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # print("cur_input_ids[35] = ", cur_input_ids[35]) #  tensor(-200, device='cuda:0')
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # print("num_images=", num_images) # tensor(1, device='cuda:0')

            
            '''
            是否去除选项字母
            '''
            # USE_OPTION_WITHOUT_LETTER
            if USE_OPTION_WITHOUT_LETTER:
                # print("cur_input_ids 是：")
                # print(cur_input_ids)
                indices_of_num_1 = (cur_input_ids == 1).nonzero(as_tuple=True)[0]
                # print("indices_of_num_1是：")
                # print(indices_of_num_1)

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

            # print("cur_input_ids是：")
            # print(cur_input_ids)
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
            # print("image_token_indices是:")
            # print(image_token_indices) # [-1, 35, 398]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            # print("cur_labels是:")
            # print(cur_labels.shape)
            # print(cur_labels) # 全是-100的tensor数据, torch.Size([398])
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            # print("cur_input_ids_noim是:")
            # print(cur_input_ids_noim[0].shape) # cur_input_ids[0:35]
            # print(cur_input_ids_noim[1].shape) # cur_input_ids[36:398]
            # print("cur_labels_noim是:")
            # print(cur_labels_noim)
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            # print("split_sizes = ", split_sizes) # [35, 362]
            # print("self.get_model().embed_tokens是：")
            # print(self.get_model().embed_tokens)
            # file_path = inspect.getsourcefile(self.get_model().embed_tokens.__class__)
            # print(file_path)
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            '''
            测试7：开始
            '''
            if USE_TEST_7:
            # if False:

                cur_options_embeds = self.get_model().embed_tokens(cur_options_ids)
            if USE_OPTION_WITHOUT_LETTER:
                cur_options_embeds = self.get_model().embed_tokens(cur_options_without_letter_ids)
            '''
            测试7：结束
            '''
            # print("cur_options_ids是：")
            # print(cur_options_ids)
            # print("options的embedding是：")
            # print(cur_options_embeds.shape)
            # print(cur_options_embeds)
            # print("cur_input_embeds是:")
            # print(cur_input_embeds.shape) # torch.Size([397, 4096])
            # print(cur_input_embeds)
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            # print("cur_input_embeds_no_im是:")
            # print(cur_input_embeds_no_im[0].shape)
            # print(cur_input_embeds_no_im)
            # print(cur_input_embeds_no_im[1][361])
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                # print("进入循环:::::", i) # 两次循环
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    # 第一次循环时进入,即i=0时
                    cur_image_features = image_features[cur_image_idx]
                    # print("cur_image_features是:")
                    # print(cur_image_features.shape) # torch.Size([576, 4096])
                    # print(cur_image_features)
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            # print("cur_new_input_embeds是:") # cur_new_input_embeds是一个列表,长度为3,每一个都是tensor
            # print(cur_new_input_embeds[0].shape) # torch.Size([35, 4096])
            # print(cur_new_input_embeds[1].shape) # torch.Size([576, 4096])
            # print(cur_new_input_embeds[2].shape) # torch.Size([362, 4096])
            # print(cur_new_input_embeds[0]) #系统特征 有正有负
            # print(cur_new_input_embeds[1]) #图像特征 有正有负
            # print(cur_new_input_embeds[2]) #prompt特征 有正有负


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
                k2 = 0
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
                
                # i_index=3601
                # sorted_indices_2d_list_top = sorted_indices_2d_list[0:min(i_index, len(sorted_indices_2d_list))]
                # # 创建一个字典来存储row出现的次数
                # row_count = {}
                # # 遍历列表中的每个元组
                # for row, col in sorted_indices_2d_list_top:
                #     # 如果row已经在字典中，增加它的计数
                #     if row in row_count:
                #         row_count[row] += 1
                #     # 如果row不在字典中，初始化计数为1
                #     else:
                #         row_count[row] = 1
                # # # 打印每个row出现的次数
                # # for row, count in row_count.items():
                # #     print(f"Row {row} appears {count} times.")
                # # 使用sorted函数和lambda表达式按照出现次数从大到小排序
                # sorted_row_count = sorted(row_count.items(), key=lambda x: x[1], reverse=True)
                # # 打印排序后的row及其出现的次数
                # # for row, count in sorted_row_count:
                # #     print(f"Row {row} appears {count} times.")

                # unique_x_values=[]
                # for iii in range(min(len(sorted_row_count),k2)):
                #     unique_x_values.append(sorted_row_count[iii][0])


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

            cur_new_input_embeds = torch.cat(cur_new_input_embeds) # 拼接起来?
            # print("cur_new_input_embeds是2::")
            # print(cur_new_input_embeds.shape) # torch.Size([973, 4096])
            # print(cur_new_input_embeds)
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
        # print("new_labels_padded是:")
        # print(new_labels_padded.shape) # torch.Size([1, 973])
        # print(new_labels_padded) # 全是-100
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        # print("attention_mask是:")
        # print(attention_mask.shape) # torch.Size([1, 973])
        # print(attention_mask) # 全false
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # print("position_ids是:")
        # print(position_ids.shape) # torch.Size([1, 973])
        # print(position_ids) # 全0

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            # print("new_input_embeds是：")
            # print(cur_new_embed.shape) # torch.Size([973, 4096])
            # print(cur_new_embed)
            # print("cur_new_labels是：")
            # print(cur_new_labels.shape) # torch.Size([973])
            # print(cur_new_labels) # 全是-100

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
                # print("new_input_embeds_padded是：")
                # print(len(new_input_embeds_padded)) #1
                # print(new_input_embeds_padded[0].shape) # torch.Size([973, 4096])
                # print(new_input_embeds_padded)
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
        # print("new_input_embeds是：111")
        # print(new_input_embeds.shape) # torch.Size([1, 973, 4096])
        # print(new_input_embeds)
        # print("attention_mask是：")
        # print(attention_mask.shape)
        # print(attention_mask)

        '''
        测试6：开始
        '''
        if USE_TEST_7:
            for index in range(35, 35+576):
                if index not in new_unique_x_values:
                    attention_mask[0][index] = 0
        # print("attention_mask.shape = ", attention_mask.shape) # torch.Size([1, 675])

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
