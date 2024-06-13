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
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        USE_SIMIGNORE = False
        USE_ONLY_OPTION = False
        USE_ONLY_QUESTION = False
        USE_QUESTION_OPTION = False
        USE_OPTION_WITHOUT_LETTER = False
        USE_OPTION_QUESTION_CONTEXT = False
        USE_C_AND_Q_AND_O = False

        # print(self.get_model().config.model_name)

        USE_SIMIGNORE = self.get_model().config.use_simignore
        if USE_SIMIGNORE:
            STEP = self.get_model().config.simignore_step
            K2 = self.get_model().config.simignore_k2
            I_INDEX = self.get_model().config.simignore_i_index
            if self.get_model().config.model_name == 'llava_1.5_7b':
                # USE_SIMIGNORE = True
                USE_ONLY_OPTION = True
                USE_OPTION_WITHOUT_LETTER = True
                
            elif self.get_model().config.model_name == 'llava_1.5_13b':
                # USE_SIMIGNORE = True
                USE_ONLY_OPTION = True

        
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
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

            if USE_OPTION_WITHOUT_LETTER and USE_SIMIGNORE:
                indices_of_num_1 = (cur_input_ids == 1).nonzero(as_tuple=True)[0]

                if len(indices_of_num_1) > 1:
                        split_point_of_num_1 = indices_of_num_1[1]
                        tensor_1 = cur_input_ids[:split_point_of_num_1]
                        tensor_2 = cur_input_ids[split_point_of_num_1:]

                        # tensor_2 = tensor_2[1:]
                        cur_input_ids = tensor_1
                        cur_options_without_letter_ids = tensor_2
                else:
                    print("error:Less than two 1 token")


            USE_WITCH = OPTION_TOKEN_INDEX
            if USE_QUESTION_OPTION:
                USE_WITCH=OPTION_AND_QUESTION_TOKEN_INDEX
            elif USE_ONLY_QUESTION:
                USE_WITCH=QUESTION_TOKEN_INDEX
            if USE_SIMIGNORE:
                if USE_ONLY_OPTION or USE_OPTION_QUESTION_CONTEXT or USE_ONLY_QUESTION:
                    indices_of_options = (cur_input_ids == USE_WITCH).nonzero(as_tuple=True)[0]
                    if len(indices_of_options) > 1:
                        cur_options_ids = cur_input_ids[indices_of_options[0]+1:indices_of_options[1]]
                    else:
                        print("error:Less than two -500 token")
                        cur_options_ids = torch.tensor([])  
                    cur_input_ids = cur_input_ids[cur_input_ids != USE_WITCH]
            if USE_C_AND_Q_AND_O and USE_SIMIGNORE:
                indices_of_cqo = (cur_input_ids == CONTEXT_AND_QUESTION_AND_OPTION_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                indices_of_iamge = (cur_input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                if len(indices_of_cqo) > 0:
                        cur_cqo_ids = cur_input_ids[indices_of_iamge[0]+1:indices_of_cqo[0]]
                else:
                    print("error:Less than one -900 token")
                    cur_options_ids = torch.tensor([])  
                cur_input_ids = cur_input_ids[cur_input_ids != CONTEXT_AND_QUESTION_AND_OPTION_TOKEN_INDEX]



            if num_images == 0:
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

            if USE_SIMIGNORE:
                cur_options_embeds = self.get_model().embed_tokens(cur_options_ids)
            if USE_OPTION_WITHOUT_LETTER and USE_SIMIGNORE:
                cur_options_embeds = self.get_model().embed_tokens(cur_options_without_letter_ids)
            if USE_C_AND_Q_AND_O and USE_SIMIGNORE:
                cur_cqo_embeds = self.get_model().embed_tokens(cur_cqo_ids)


            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))


            if USE_SIMIGNORE:
                test_1_image_embeds = cur_new_input_embeds[1]
                # test_1_text_embeds = cur_new_input_embeds[2]
                test_1_text_embeds = cur_options_embeds
                # normalize
                test_1_image_embeds = F.normalize(test_1_image_embeds, p=2, dim=1)
                test_1_text_embeds = F.normalize(test_1_text_embeds, p=2, dim=1)  
                USE_PCA = False    
                if USE_PCA:
                    '''PCA'''
                    pca = PCA(n_components=2)
                    test_1_image_embeds_cpu = test_1_image_embeds.cpu()
                    test_1_text_embeds_cpu = test_1_text_embeds.cpu()
                    test_1_image_embeds_np = test_1_image_embeds_cpu.numpy()
                    test_1_text_embeds_np = test_1_text_embeds_cpu.numpy()
                    pca = PCA(n_components=2)
                    test_1_image_embeds_reduced = pca.fit_transform(test_1_image_embeds_np)
                    test_1_text_embeds_reduced = pca.fit_transform(test_1_text_embeds_np)
                    test_1_image_embeds_reduced_tensor = torch.from_numpy(test_1_image_embeds_reduced)
                    test_1_text_embeds_reduced_tensor = torch.from_numpy(test_1_text_embeds_reduced)
                    similarity_matrix = torch.mm(test_1_image_embeds_reduced_tensor, test_1_text_embeds_reduced_tensor.transpose(0, 1))

                # calculating similarity
                similarity_matrix = torch.mm(test_1_image_embeds, test_1_text_embeds.transpose(0, 1))

                flat_similarity = similarity_matrix.view(-1)

                sorted_values, sorted_indices = torch.sort(flat_similarity, descending=True)

                sorted_indices_list = sorted_indices.tolist()

                sorted_indices_2d_list = [None] * len(sorted_indices_list)
                rows, cols = similarity_matrix.shape
                for i, idx in enumerate(sorted_indices_list):
                    row = idx // cols
                    col = idx % cols
                    sorted_indices_2d_list[i] = (row, col)

                unique_x_values = []

                step = STEP  
                k2 = K2

                i_index = I_INDEX
                while len(unique_x_values) < k2 and i_index < len(sorted_indices_2d_list) and sorted_values[i_index] >= -1:
                    current_tuple = sorted_indices_2d_list[i_index]
                    x_value = current_tuple[0]
                    if x_value not in unique_x_values:
                        unique_x_values.append(x_value)
                    i_index += step
                    
                new_unique_x_values = [x + 35 for x in unique_x_values]





            cur_new_input_embeds = torch.cat(cur_new_input_embeds) 
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)

        if tokenizer_model_max_length is not None:
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
            cur_len = cur_new_embed.shape[0] # 973
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
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


        

        if USE_SIMIGNORE:
            for index in range(35, 35+576):
                if index not in new_unique_x_values:
                    attention_mask[0][index] = 0


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


