# [CVPR 2026] FINER: MLLMs Hallucinate under Fine-grained Negative Queries
[![Paper](https://img.shields.io/badge/paper-arXiv-B31B1B.svg)](ADD_ARXIV_LINK_HERE)
[![Project Page](https://img.shields.io/badge/Project-Website-blue.svg)](https://explainableml.github.io/finer-project/)
[![Models](https://img.shields.io/badge/HuggingFace-FINER-FFD700?logo=huggingface&logoColor=yellow)](https://huggingface.co/collections/xiaorui638/finer-models)
[![FINER-Tuning Data](https://img.shields.io/badge/HuggingFace-FINER-FFD700?logo=huggingface&logoColor=yellow)](https://huggingface.co/datasets/xiaorui638/FINER-Tuning-data)

**Authors:** [Rui Xiao](https://www.eml-munich.de/people/rui-xiao), [Sanghwan Kim](https://kim-sanghwan.github.io/), [Yongqin Xian](https://xianyongqin.github.io/), [Zeynep Akata](https://www.eml-munich.de/people/zeynep-akata), [Stephan Alaniz](https://www.telecom-paris.fr/)


## Abstract
Multimodal large language models (MLLMs) struggle with hallucinations, particularly with fine-grained queries, a challenge underrepresented by existing benchmarks that focus on coarse image-related questions. We introduce FIne-grained NEgative queRies (FINER), alongside two benchmarks: FINER-CompreCap and FINER-DOCCI. Using FINER, we analyze hallucinations across four settings: multi-object, multi-attribute, multi-relation, and “what” questions. Our benchmarks reveal that MLLMs hallucinate when fine-grained mismatches co-occur with genuinely present elements in the image. To address this, we propose FINER-Tuning, leveraging Direct Preference Optimization (DPO) on FINER-inspired data. Finetuning four frontier MLLMs with FINER-Tuning yields up to 24.2% gains on hallucinations from our benchmarks, while simultaneously improving performance on eight existing hallucination suites and enhancing general multimodal capabilities across six benchmarks.

## Methodology
We refer to our [project page](https://explainableml.github.io/finer-project/), where we walk you through the paper.

### FINER-Benchmarks
![](assets/FINER-Benchmarks.png "Construction process of FINER benchmarks")

### FINER-Tuning
![](assets/FINER-Tuning.png "Construction process of FINER-Tuning data")

Code coming soon.
