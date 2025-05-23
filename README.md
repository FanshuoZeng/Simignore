<h1 align="center">Enhancing Multimodal Large Language Models Complex Reasoning via Similarity Computation</h1>



<div align=center>
<img src="./fig/structure.jpg"/><br>
</div>

### News
- 2024.12.10 The paper has been accepted in AAAI 2025







## Setup
```bash
conda create -n simignore python=3.10
conda activate simignore
cd src
bash setup.sh
```

## Checkpoint

You can download **LLaVA1.5-7b** from [Hugging Face](https://huggingface.co/liuhaotian/llava-v1.5-7b) and save it under `checkpoint/llava`.
## ScienceQA dataset
You can download **ScienceQA** from [Google Drive](https://drive.google.com/drive/folders/1w8imCXWYn2LxajmGeGH_g5DaL2rabHev?usp=sharing) and unzip the images under `data/scienceqa/images`.

## Simignore Zero-shot Inference
We provide the Zero-shot inference procedure for the LLaVA1.5-7b model and the LLaVA1.5-13b model on the *ScienceQA(Image)* dataset. We conduct the following experiments on one 4090D GPU (24G).

Inference using LLaVA-v1.5-7b model.
```bash
bash ./src/Simignore/inference/eval/eval_sqa_latency_inplace.sh
```
Inference using LLaVA-v1.5-13b model.
```bash
bash ./src/Simignore/inference/eval/eval_sqa_latency_inplace_13b.sh 
```


## Simignore Evaluatio

Evaluation using LLaVA-v1.5-7b model.
```bash
bash ./src/Simignore/inference/eval/generate_sqa_results.sh
```
Evaluation using LLaVA-v1.5-13b model.
```bash



bash ./src/Simignore/inference/eval/generate_sqa_results_13b.sh 
```
## Main result
<div align=center>
<img src="./fig/main-result.png"/><br>
</div>

## Citation
```bibtex
@article{zhang2024enhancing,
  title={Enhancing Multimodal Large Language Models Complex Reason via Similarity Computation},
  author={Zhang, Xiaofeng and Zeng, Fanshuo and Quan, Yihao and Hui, Zheng and Yao, Jiawei},
  journal={The 39th Annual AAAI Conference on Artificial Intelligence},
  year={2024}
}

@article{zhang2024simignore,
  title={Simignore: Exploring and enhancing multimodal large model complex reasoning via similarity computation},
  author={Zhang, Xiaofeng and Zeng, Fanshuo and Gu, Chaochen},
  journal={Neural Networks},
  pages={107059},
  year={2024},
  publisher={Elsevier}
}

```
