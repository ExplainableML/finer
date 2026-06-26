# FINER-Tuning Training

This folder contains the code for preparing FINER-Tuning data and training MLLMs with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

The expected structure is:

```text
train/
├── generate_training_data/
├── LlamaFactory/
├── prepare_training_data.py
└── sample_training_data/
```

We use LLaMA-Factory for training and provide scripts for preparing the released FINER-Tuning data, running ablations, and sampling customized training data.

---

## 1. Environment

Our training code is based on LLaMA-Factory. Please follow the installation instructions from the local `train/LlamaFactory/README.md` or the official LLaMA-Factory repository.

A typical setup is:

```bash
cd train/LlamaFactory

conda create -n finer_train python=3.10 -y
conda activate finer_train

pip install -e ".[torch,metrics]"
```

Depending on your cluster and CUDA version, you may need to install the correct PyTorch build manually before installing LLaMA-Factory.

---

## 2. Prepare Training Data

We release the FINER-Tuning annotations and image shards on Hugging Face:

```text
xiaorui638/FINER-Tuning-data
```

The script `prepare_training_data.py` will:

1. download the released annotation files and image tar files,
2. extract the image tar files,
3. rewrite image paths inside the JSONL annotation files to local paths,
4. optionally update `LlamaFactory/data/dataset_info.json` so that LLaMA-Factory can directly load the prepared dataset.

### 2.1 Main Experiment Data

From the `train/` directory, run:

```bash
cd train

python prepare_training_data.py \
  --output_dir ./FINER-Tuning-data \
  --llamafactory_dir ./LlamaFactory \
  --dataset_name pixmo_dpo_220k_pon \
  --jsonl_relpath sampled_annotations/main_experiments/first6_pon.jsonl
```

Here:

* `--output_dir` specifies where the released data will be downloaded and prepared.
* `--llamafactory_dir` points to the local LLaMA-Factory directory.
* `--dataset_name` specifies the dataset key in `LlamaFactory/data/dataset_info.json`.
* `--jsonl_relpath` specifies which JSONL annotation file, relative to `--output_dir`, should be registered in LLaMA-Factory.

After running the script, the prepared data will look like:

```text
FINER-Tuning-data/
├── images/
│   ├── *.tar
│   └── ...
├── images_extracted/
│   ├── train-00000-of-00075/
│   ├── train-00001-of-00075/
│   └── ...
├── sampled_annotations/
│   ├── main_experiments/
│   └── ablations/
└── full_annotations/
```

The script updates the `file_name` field of the selected LLaMA-Factory dataset entry. Therefore, after preparation, the dataset can be used in LLaMA-Factory with:

```text
dataset: pixmo_dpo_220k_pon
```

If the image tar files have already been extracted, you can skip extraction:

```bash
python prepare_training_data.py \
  --output_dir ./FINER-Tuning-data \
  --llamafactory_dir ./LlamaFactory \
  --dataset_name pixmo_dpo_220k_pon \
  --jsonl_relpath sampled_annotations/main_experiments/first6_pon.jsonl \
  --skip_extract
```

---

### 2.2 Optional: Ablation Data

We also release annotation files for the ablation studies under:

```text
sampled_annotations/ablations/
```

To prepare an ablation split, run the same script but replace `--jsonl_relpath`.

For example, to train only on object-oriented samples:

```bash
python prepare_training_data.py \
  --output_dir ./FINER-Tuning-data \
  --llamafactory_dir ./LlamaFactory \
  --dataset_name pixmo_dpo_220k_pon \
  --jsonl_relpath sampled_annotations/ablations/train_on_subsets/first10_only_obj.jsonl
```

Other subset ablations include:

```text
sampled_annotations/ablations/train_on_subsets/first10_only_attr.jsonl
sampled_annotations/ablations/train_on_subsets/first10_only_obj.jsonl
sampled_annotations/ablations/train_on_subsets/first10_only_rel.jsonl
sampled_annotations/ablations/train_on_subsets/first10_only_wh.jsonl
```

We also provide ablations for different training data construction strategies:

```text
sampled_annotations/ablations/training_methods/dpo_only_neg.jsonl
sampled_annotations/ablations/training_methods/dpo_pos_and_neg.jsonl
sampled_annotations/ablations/training_methods/sft_only_neg.jsonl
sampled_annotations/ablations/training_methods/sft_pos_and_neg.jsonl
```

The same `dataset_name` can be reused if you only train one experiment at a time. If you want to keep multiple prepared dataset entries simultaneously, create additional entries in `LlamaFactory/data/dataset_info.json` and pass the corresponding name through `--dataset_name`.

---

### 2.3 Generate Your Own Training Data

We also provide scripts for generating and sampling customized FINER-style training data.

The folder `generate_training_data/` contains scripts for generating different types of fine-grained questions

```text
generate_training_data/
├── generate_obj.py
├── generate_attr.py
├── generate_rel.py
├── generate_wh.py
├── params.py
├── prompts/
└── utils/
```

The folder `sample_training_data/` contains:

```text
sample_training_data/sampling_sharegpt.py
```

This script can be used to sample and organize generated data into the ShareGPT-style format used by LLaMA-Factory.

You can modify the sampling configuration to create your own training mixtures, for example by changing the number of samples, the question types, or the positive/negative data composition.

---

## 3. Start Training

We train our models with SLURM and provide example `sbatch` scripts under:

```text
train/LlamaFactory/
```

The provided scripts are:

```text
sbatch_train_internvl35_8b_finer_tuning.sh
sbatch_train_internvl35_14b_finer_tuning.sh
sbatch_train_llava16_7b_finer_tuning.sh
sbatch_train_qwen25vl_finer_tuning.sh
```

To launch training, enter the LLaMA-Factory directory and submit the corresponding script:

```bash
cd train/LlamaFactory

sbatch sbatch_train_internvl35_8b_finer_tuning.sh
```

or, for another model:

```bash
sbatch sbatch_train_internvl35_14b_finer_tuning.sh
sbatch sbatch_train_llava16_7b_finer_tuning.sh
sbatch sbatch_train_qwen25vl_finer_tuning.sh
```

Before launching, please check and modify the SLURM script according to your own cluster:

The dataset name should match the entry prepared in `LlamaFactory/data/dataset_info.json`, for example:

```text
pixmo_dpo_220k_pon
```

---

## 4. Notes

### Dataset registration

The preparation script only updates the selected dataset entry in:

```text
LlamaFactory/data/dataset_info.json
```

It does not modify the LLaMA-Factory training code. Specifically, it updates the `file_name` field of the selected dataset key so that LLaMA-Factory can load the prepared local JSONL file.

### Image paths

The released JSONL annotations originally contain image paths from our internal data storage. During preparation, these paths are rewritten to point to the locally extracted images under:

```text
FINER-Tuning-data/images_extracted/
```

If you encounter image loading errors, first check whether the paths in the prepared JSONL files exist locally.

A quick sanity check is:

```bash
python - <<'PY'
import json
from pathlib import Path

jsonl_path = Path("./FINER-Tuning-data/sampled_annotations/main_experiments/first6_pon.jsonl")

with jsonl_path.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        sample = json.loads(line)
        for img in sample.get("images", []):
            print(img, Path(img).exists())
        if i >= 5:
            break
PY
```

The printed paths should return `True`.

### Switching between experiments

When switching from the main experiment to an ablation file, rerun `prepare_training_data.py` with a different `--jsonl_relpath`. This will update the same LLaMA-Factory dataset key to point to the new annotation file.

For example:

```bash
python prepare_training_data.py \
  --output_dir ./FINER-Tuning-data \
  --llamafactory_dir ./LlamaFactory \
  --dataset_name pixmo_dpo_220k_pon \
  --jsonl_relpath sampled_annotations/ablations/training_methods/dpo_pos_and_neg.jsonl
```

Then launch the corresponding training script again.

---