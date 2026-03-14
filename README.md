# [CVPR 2026] FINER: MLLMs Hallucinate under Fine-grained Negative Queries
[![Paper](https://img.shields.io/badge/paper-arXiv-B31B1B.svg)](ADD_ARXIV_LINK_HERE)
[![Project Page](https://img.shields.io/badge/Project-Website-blue.svg)](https://sean-xr.github.io/finer-website/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-FINER-FFD700?logo=huggingface&logoColor=yellow)](ADD_HF_LINK_HERE)

**Authors:** [Rui Xiao](https://www.eml-munich.de/people/rui-xiao), [Sanghwan Kim](https://kim-sanghwan.github.io/), [Yongqin Xian](https://xianyongqin.github.io/), [Zeynep Akata](https://www.eml-munich.de/people/zeynep-akata), [Stephan Alaniz](https://www.telecom-paris.fr/)


## TL;DR
We introduce **FINER**, short for **FI**ne-grained **NE**gative que**R**ies, together with two benchmark studies: **FINER-CompreCap** and **FINER-DOCCI**. FINER evaluates hallucination under four settings: **Multi-obj**, **Multi-attr**, **Multi-rel**, and **Wh** questions. Our benchmark studies reveal that MLLMs often fail when fine-grained mismatches are hidden among otherwise correct image details. 

To improve this, we propose **FINER-Tuning**, a preference-learning approach based on Direct Preference Optimization (DPO). FINER-Tuning uses **minimally edited fine-grained positive and negative queries** so that the model learns to accept correct claims and reject precise but incorrect ones. Across four frontier MLLMs, FINER-Tuning substantially improves performance on FINER, also generalizes to existing hallucination benchmarks, and preserves or improves general multimodal capabilities.

We refer to our [project page](https://img.shields.io/badge/Project-Website-blue.svg)](https://sean-xr.github.io/finer-website/), where we walk you through our storylines.

## Methodology
![](assets/finer_overview.png "Overview of FINER")

### FINER-Benchmarks
FINER studies hallucination under **fine-grained negative queries**. Instead of only asking whether a single object exists, FINER composes questions with multiple semantic elements from the image scene graph, including:
- **objects**
- **attributes**
- **relations**

We construct two benchmarks:

- **FINER-CompreCap**, built from **CompreCap**
- **FINER-DOCCI**, built from **DOCCI**

Each benchmark contains four settings:
1. **Multi-obj**: one object is replaced by a plausible but incorrect negative object
2. **Multi-attr**: one attribute is replaced by a fine-grained negative attribute
3. **Multi-rel**: one relation is replaced by a negative relation
4. **Wh**: a factual “what” question is asked with one incorrect fine-grained condition

To reduce answer priors from simple yes/no responses, FINER uses **multiple-choice questions (MCQs)**. Each negative MCQ is paired with its corresponding positive MCQ, and evaluation is based on **paired accuracy**, which requires the model to answer both sides correctly.

For **FINER-CompreCap**, scene graphs come from the source dataset annotations.  
For **FINER-DOCCI**, we extract scene-graph-like annotations from long human-written captions using a multi-stage pipeline with LLM/MLLM verification and sampled human checking.

### FINER-Tuning
FINER-Tuning is a **DPO-based training strategy** designed to improve MLLMs on fine-grained hallucination.

Starting from long image captions, we first extract fine-grained positive phrases corresponding to the same four settings:
- object summaries
- attribute summaries
- relation summaries
- composed Wh-style descriptions

We then generate corresponding **negative phrases** by minimally editing a single object, attribute, or relation while keeping the rest of the description consistent. From these, we construct:
- **positive queries** with accepted and rejected answers
- **negative queries** with accepted and rejected answers

This produces preference tuples for **Direct Preference Optimization (DPO)**, so the model learns to:
- **accept** visually grounded fine-grained statements
- **reject** precise but unsupported contradictions

Compared with generic hallucination reduction, FINER-Tuning directly targets hallucinations hidden inside otherwise plausible, detailed queries.

## Benchmark Statistics

### FINER-CompreCap
- **Multi-obj:** 6,300 MCQs
- **Multi-attr:** 3,338 MCQs
- **Multi-rel:** 4,280 MCQs
- **Wh:** 3,166 MCQs

### FINER-DOCCI
- **Multi-obj:** 10,000 MCQs
- **Multi-attr:** 28,630 MCQs
- **Multi-rel:** 11,542 MCQs
- **Wh:** 20,944 MCQs

## Main Results
FINER is challenging even for strong frontier MLLMs. Performance drops notably when multiple objects, attributes, or relations are involved, and **Wh** questions remain especially difficult.

FINER-Tuning consistently improves all four evaluated base MLLMs:
- **LLaVA-1.6-7B**
- **Qwen2.5-VL-7B**
- **InternVL-3.5-8B**
- **InternVL-3.5-14B**

On the FINER benchmarks, gains reach up to **24.2% paired accuracy**. These improvements also transfer to a broad set of existing hallucination benchmarks and do not come at the expense of general multimodal performance.

## Repository Structure
```bash
finer/
├── benchmark/          # FINER benchmark files and metadata
├── training/           # FINER-Tuning data generation and training code
├── evaluation/         # evaluation scripts for FINER and other benchmarks
├── assets/             # figures for README / project page
├── scripts/            # example training / inference scripts
└── README.md
