# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import unittest

import numpy as np
from datasets import load_dataset

from transformers.testing_utils import require_torch, require_vision, slow
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

    from transformers.models.seggpt.modeling_seggpt import SegGptImageSegmentationOutput

if is_vision_available():
    from transformers import SegGptImageProcessor


class SegGptImageProcessingTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ):
        size = size if size is not None else {"height": 18, "width": 18}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def prepare_image_processor_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "size": self.size,
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["height"], self.size["width"]

    def expected_post_processed_shape(self):
        return self.size["height"] // 2, self.size["width"]

    def get_fake_image_segmentation_output(self):
        torch.manual_seed(42)
        return SegGptImageSegmentationOutput(
            pred_masks=torch.rand(self.batch_size, self.num_channels, self.size["height"], self.size["width"])
        )

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


def prepare_mask():
    ds = load_dataset("EduardoPacheco/seggpt-example-data")["train"]
    return ds[0]["mask"].convert("L")


def prepare_img():
    ds = load_dataset("EduardoPacheco/seggpt-example-data")["train"]
    images = [image.convert("RGB") for image in ds["image"]]
    masks = [image.convert("RGB") for image in ds["mask"]]
    return images, masks


@require_torch
@require_vision
class SegGptImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = SegGptImageProcessor if is_vision_available() else None

    def setUp(self):
        self.image_processor_tester = SegGptImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"height": 18, "width": 18})

        image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size=42)
        self.assertEqual(image_processor.size, {"height": 42, "width": 42})

    def test_image_processor_palette(self):
        num_labels = 3
        image_processing = self.image_processing_class(**self.image_processor_dict)
        palette = image_processing.get_palette(num_labels)
        self.assertEqual(len(palette), num_labels + 1)
        self.assertEqual(palette[0], (0, 0, 0))

    def test_mask_equivalence(self):
        image_processor = SegGptImageProcessor()

        mask_binary = prepare_mask()
        mask_rgb = mask_binary.convert("RGB")

        inputs_binary = image_processor(images=None, prompt_masks=mask_binary, return_tensors="pt")
        inputs_rgb = image_processor(images=None, prompt_masks=mask_rgb, return_tensors="pt")

        self.assertTrue((inputs_binary["prompt_masks"] == inputs_rgb["prompt_masks"]).all().item())

    def test_mask_to_rgb(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        mask = prepare_mask()
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)

        def check_two_colors(image, color1=(0, 0, 0), color2=(255, 255, 255)):
            pixels = image.transpose(1, 2, 0).reshape(-1, 3)
            unique_colors = np.unique(pixels, axis=0)
            if len(unique_colors) == 2 and (color1 in unique_colors) and (color2 in unique_colors):
                return True
            else:
                return False

        num_labels = 1
        palette = image_processing.get_palette(num_labels)

        # Should only duplicate repeat class indices map, hence only (0,0,0) and (1,1,1)
        mask_duplicated = image_processing.mask_to_rgb(mask)
        # Mask using palette, since only 1 class is present we have colors (0,0,0) and (255,255,255)
        mask_painted = image_processing.mask_to_rgb(mask, palette=palette)

        self.assertTrue(check_two_colors(mask_duplicated, color2=(1, 1, 1)))
        self.assertTrue(check_two_colors(mask_painted, color2=(255, 255, 255)))

    def test_post_processing_semantic_segmentation(self):
        image_processor = self.image_processing_class(**self.image_processor_dict)
        outputs = self.image_processor_tester.get_fake_image_segmentation_output()
        post_processed = image_processor.post_process_semantic_segmentation(outputs)

        self.assertEqual(len(post_processed), self.image_processor_tester.batch_size)

        expected_semantic_map_shape = self.image_processor_tester.expected_post_processed_shape()
        self.assertEqual(post_processed[0].shape, expected_semantic_map_shape)

    @slow
    def test_pixel_values(self):
        images, masks = prepare_img()
        input_image = images[1]
        prompt_image = images[0]
        prompt_mask = masks[0]

        image_processor = SegGptImageProcessor.from_pretrained("BAAI/seggpt-vit-large")

        inputs = image_processor(
            images=input_image, prompt_images=prompt_image, prompt_masks=prompt_mask, return_tensors="pt"
        )

        # Verify pixel values
        expected_prompt_pixel_values = torch.tensor(
            [
                [[-0.6965, -0.6965, -0.6965], [-0.6965, -0.6965, -0.6965], [-0.6965, -0.6965, -0.6965]],
                [[1.6583, 1.6583, 1.6583], [1.6583, 1.6583, 1.6583], [1.6583, 1.6583, 1.6583]],
                [[2.3088, 2.3088, 2.3088], [2.3088, 2.3088, 2.3088], [2.3088, 2.3088, 2.3088]],
            ]
        )

        expected_pixel_values = torch.tensor(
            [
                [[1.6324, 1.6153, 1.5810], [1.6153, 1.5982, 1.5810], [1.5810, 1.5639, 1.5639]],
                [[1.2731, 1.2556, 1.2206], [1.2556, 1.2381, 1.2031], [1.2206, 1.2031, 1.1681]],
                [[1.6465, 1.6465, 1.6465], [1.6465, 1.6465, 1.6465], [1.6291, 1.6291, 1.6291]],
            ]
        )

        expected_prompt_masks = torch.tensor(
            [
                [[-2.1179, -2.1179, -2.1179], [-2.1179, -2.1179, -2.1179], [-2.1179, -2.1179, -2.1179]],
                [[-2.0357, -2.0357, -2.0357], [-2.0357, -2.0357, -2.0357], [-2.0357, -2.0357, -2.0357]],
                [[-1.8044, -1.8044, -1.8044], [-1.8044, -1.8044, -1.8044], [-1.8044, -1.8044, -1.8044]],
            ]
        )

        self.assertTrue(torch.allclose(inputs.pixel_values[0, :, :3, :3], expected_pixel_values, atol=1e-4))
        self.assertTrue(
            torch.allclose(inputs.prompt_pixel_values[0, :, :3, :3], expected_prompt_pixel_values, atol=1e-4)
        )
        self.assertTrue(torch.allclose(inputs.prompt_masks[0, :, :3, :3], expected_prompt_masks, atol=1e-4))
