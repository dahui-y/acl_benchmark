<div align="center">
<h2 align="center">
   <b>OSCBench: Benchmarking Object State Change in Text-to-Video Generation</b>
</h2>

<div>
<a target="_blank" href="https://hanxjing.github.io/">Xianjing&#160;Han</a><sup>1*</sup>,
<a target="_blank" href="https://binzhubz.github.io/">Bin&#160;Zhu</a><sup>2*&#9993;</sup>,
<a target="_blank">Shiqi&#160;Hu</a><sup>1</sup>,
<a target="_blank" href="https://franklin-li.com/">Franklin&#160;Mingzhe&#160;Li</a><sup>3</sup>,
<a target="_blank" href="https://patrickcarrington.com/">Patrick&#160;Carrington</a><sup>3</sup>,
<a target="_blank" href="https://www.comp.nus.edu.sg/cs/people/rogerz/">Roger&#160;Zimmermann</a><sup>1</sup>,
<a target="_blank" href="https://jingjing1.github.io/">Jingjing&#160;Chen</a><sup>4</sup>
<br/>
</div>

<sup>1</sup>National University of Singapore&#160;&#160;&#160;
<sup>2</sup>Singapore Management University&#160;&#160;&#160;
<sup>3</sup>Carnegie Mellon University&#160;&#160;&#160;
<sup>4</sup>Fudan University
<br/>
<sup>&#9993;</sup>Corresponding author
<br/>

<div align="center">
    <a href="https://arxiv.org/abs/2603.11698" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
    <a href="https://huggingface.co/datasets/XianjingHan/OSCBench_Dataset" target="_blank">
    <img src="https://img.shields.io/badge/Hugging%20Face-Dataset-blue" alt="Hugging Face Dataset"></a>
    <a href="https://hanxjing.github.io/OSCBench/" target="_blank">
    <img src="https://img.shields.io/badge/Project-Page-orange" alt="Project Page"></a>
</div>
</div>

## Overview
OSCBench is a benchmark for evaluating whether text-to-video models can generate correct and temporally consistent object state changes.


This repository currently contains prompt resources, action/object taxonomies, frame extraction code, an MLLM-based evaluation script, and a correlation analysis script for comparing automatic judgments with human ratings.

## Setup

1. Install the required dependencies:
```bash
pip install openai opencv-python numpy scipy
```

2. Set up your OpenAI API credentials in `mllm_eval.py`:
```python
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
```


## Pipeline Components

### 1. Video Generation and Frame Sampling

Generate videos from OSCBench prompts using your target text-to-video model, then extract evenly sampled frames for automatic evaluation.

#### Generate Videos

Use prompts from `prompts.txt` or one of the split files under `prompt_splits/`. 


#### Sample Frames

This script samples `20` evenly spaced frames from each video and saves them into one subfolder per video.

Usage:
```bash
python extract_frames.py
```


### 2. Model Evaluation

This script evaluates sampled frames using an MLLM through the OpenAI Responses API. It asks the model to inspect the frames chronologically and return evidence-backed scores for eight dimensions:

- `1a` Subject Alignment
- `1b` Manipulated Object Alignment
- `2a` Action Accuracy
- `3a` Object State Change Accuracy
- `3b` Object Change Continuity and Consistency
- `4a` Scene Alignment
- `5a` Realism
- `5b` Aesthetic

Among these dimensions, `3a` and `3b` directly measure the object state change ability emphasized by OSCBench.

Usage:
```bash
python mllm_eval.py
```


### 3. Results Analysis

This script analyzes the correlation between MLLM-based evaluation and human evaluation, following the benchmark's automatic-evaluation validation setting described on the project page and in the paper.

It computes per-dimension:

1. Kendall's tau
2. Spearman's rho

Usage:
```bash
python result_analyze.py
```


## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{han2026oscbench,
  title={OSCBench: Benchmarking Object State Change in Text-to-Video Generation},
  author={Han, Xianjing and Zhu, Bin and Hu, Shiqi and Li, Franklin Mingzhe and Carrington, Patrick and Zimmermann, Roger and Chen, Jingjing},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={30867--30884},
  year={2026}
}
```
