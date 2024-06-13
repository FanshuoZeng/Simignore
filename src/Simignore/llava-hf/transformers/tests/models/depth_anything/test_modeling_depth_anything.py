# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Testing suite for the PyTorch Depth Anything model. """


import unittest

from transformers import DepthAnythingConfig, Dinov2Config
from transformers.file_utils import is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import DepthAnythingForDepthEstimation
    from transformers.models.depth_anything.modeling_depth_anything import DEPTH_ANYTHING_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import DPTImageProcessor


class DepthAnythingModelTester:
    # Copied from tests.models.dpt.test_modeling_dpt_auto_backbone.DPTModelTester.__init__
    def __init__(
        self,
        parent,
        batch_size=2,
        num_channels=3,
        image_size=32,
        patch_size=16,
        use_labels=True,
        num_labels=3,
        is_training=True,
        hidden_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=8,
        out_features=["stage1", "stage2"],
        apply_layernorm=False,
        reshape_hidden_states=False,
        neck_hidden_sizes=[2, 2],
        fusion_hidden_size=6,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.out_features = out_features
        self.apply_layernorm = apply_layernorm
        self.reshape_hidden_states = reshape_hidden_states
        self.use_labels = use_labels
        self.num_labels = num_labels
        self.is_training = is_training
        self.neck_hidden_sizes = neck_hidden_sizes
        self.fusion_hidden_size = fusion_hidden_size
        # DPT's sequence length
        self.seq_length = (self.image_size // self.patch_size) ** 2 + 1

    # Copied from tests.models.dpt.test_modeling_dpt_auto_backbone.DPTModelTester.prepare_config_and_inputs
    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return DepthAnythingConfig(
            backbone_config=self.get_backbone_config(),
            reassemble_hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            neck_hidden_sizes=self.neck_hidden_sizes,
            fusion_hidden_size=self.fusion_hidden_size,
        )

    # Copied from tests.models.dpt.test_modeling_dpt_auto_backbone.DPTModelTester.get_backbone_config
    def get_backbone_config(self):
        return Dinov2Config(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            is_training=self.is_training,
            out_features=self.out_features,
            reshape_hidden_states=self.reshape_hidden_states,
        )

    # Copied from tests.models.dpt.test_modeling_dpt_auto_backbone.DPTModelTester.create_and_check_for_depth_estimation with DPT->DepthAnything
    def create_and_check_for_depth_estimation(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = DepthAnythingForDepthEstimation(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.predicted_depth.shape, (self.batch_size, self.image_size, self.image_size))

    # Copied from tests.models.dpt.test_modeling_dpt_auto_backbone.DPTModelTester.prepare_config_and_inputs_for_common
    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class DepthAnythingModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Depth Anything does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (DepthAnythingForDepthEstimation,) if is_torch_available() else ()
    pipeline_model_mapping = {"depth-estimation": DepthAnythingForDepthEstimation} if is_torch_available() else {}

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = DepthAnythingModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=DepthAnythingConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_from_and_save_pretrained_subfolder()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    @unittest.skip(reason="Depth Anything with AutoBackbone does not have a base model and hence no input_embeddings")
    def test_inputs_embeds(self):
        pass

    def test_for_depth_estimation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_depth_estimation(*config_and_inputs)

    @unittest.skip(reason="Depth Anything does not support training yet")
    def test_training(self):
        pass

    @unittest.skip(reason="Depth Anything does not support training yet")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Depth Anything with AutoBackbone does not have a base model and hence no input_embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="Depth Anything with AutoBackbone does not have a base model")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="Depth Anything with AutoBackbone does not have a base model")
    def test_save_load_fast_init_to_base(self):
        pass

    @unittest.skip(
        reason="This architecture seems to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seems to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in DEPTH_ANYTHING_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = DepthAnythingForDepthEstimation.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
@slow
class DepthAnythingModelIntegrationTest(unittest.TestCase):
    def test_inference(self):
        image_processor = DPTImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
        model = DepthAnythingForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf").to(torch_device)

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # verify the predicted depth
        expected_shape = torch.Size([1, 518, 686])
        self.assertEqual(predicted_depth.shape, expected_shape)

        expected_slice = torch.tensor(
            [[8.8204, 8.6468, 8.6195], [8.3313, 8.6027, 8.7526], [8.6526, 8.6866, 8.7453]],
        ).to(torch_device)

        self.assertTrue(torch.allclose(outputs.predicted_depth[0, :3, :3], expected_slice, atol=1e-6))
