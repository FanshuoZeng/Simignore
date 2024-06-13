<h1 align="center">Simignore</h1>




<div align=center>
<img src="./fig/structure.jpg"/><br>
</div>










## Setup
```bash
conda create -n simignore python=3.10
conda activate simignore
cd src
bash setup.sh
```

## ScienceQA dataset
You can download ScienceQA from [Google Drive]([https://www.google.com](https://drive.google.com/drive/folders/1w8imCXWYn2LxajmGeGH_g5DaL2rabHev)) and unzip the images under data/scienceqa/images.


## Simignore Zero-shot Inference
We provide the Zero-shot inference procedure for the LLaVA1.5-7b model and the LLaVA1.5-13b model on the *ScienceQA(Image)* dataset.

```bash
bash ./src/Simignore/inference/eval/eval_sqa_latency_inplace.sh
bash ./src/Simignore/inference/eval/eval_sqa_latency_inplace_13b.sh 
```

## Simignore Evaluatio

```bash
bash ./src/Simignore/inference/eval/generate_sqa_results.sh
bash ./src/Simignore/inference/eval/generate_sqa_results_13b.sh 
```
