# TPDOMDR
<img width="8908" height="3507" alt="model small" src="https://github.com/user-attachments/assets/6e64746a-0df1-4193-846c-1a6ac555bb4d" />







## Overview

TPDOMDR is a five-class diabetic retinopathy grading project that combines FLAIR retinal vision-language features with MVCINN multi-view image features.

## Environment

- Ubuntu 20.04 or later
- Python 3.11
- NVIDIA GPU with a CUDA 12.4-compatible driver
- PyTorch 2.6.0
- TorchVision 0.21.0
- Two GPUs with at least 32 GB of memory each are recommended for the default batch size

## Project Structure

```text
TPDOMDR-main/
|-- configs/
|   |-- __init__.py
|   `-- config.py
|-- data/
|   |-- __init__.py
|   `-- unified_dataset.py
|-- models/
|   |-- __init__.py
|   |-- dr_dictionary.py
|   |-- flair_wrapper.py
|   |-- fusion_module.py
|   `-- mvcinn_wrapper.py
|-- utils/
|   |-- __init__.py
|   |-- checkpoint.py
|   |-- metrics.py
|   `-- visualization.py
|-- best.pth
|-- train.py
|-- test.py
|-- README.md
`-- requirements.txt
```

## Server Directory Layout

The Linux configuration automatically uses `/root/DR` when that directory exists. Keep the project, external model repositories, pretrained weights, and datasets in the following layout.

```text
/root/DR/
|-- TPDOMDR-main/
|   |-- best.pth
|   |-- train.py
|   `-- test.py
|-- mysearch/
|   |-- FLAIR-main/
|   |   |-- flair/
|   |   `-- flair_resnet.pth
|   `-- MVCINN-master/
|       `-- MVCINN-code/
|           |-- models/
|           `-- final_0.8010.pth
|-- MFIDDR/
|   |-- train/
|   |-- test/
|   |-- train_fourpic_label.csv
|   `-- test_fourpic_label.csv
|-- DDR/
|   `-- DDR-dataset/
|       `-- DR_grading/
|           |-- train/
|           |-- test/
|           |-- train.txt
|           `-- test.txt
|-- B. Disease Grading/
|   |-- 1. Original Images/
|   |   `-- b. Testing Set/
|   `-- 2. Groundtruths/
|       `-- b. IDRiD_Disease Grading_Testing Labels.csv
`-- APTOS/
    |-- test_images/
    `-- test.csv
```

## Installation

```bash
cd /root/DR/TPDOMDR-main
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Environment Verification

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -c "import sys; sys.path.insert(0, '/root/DR/mysearch/FLAIR-main'); from flair.modeling.model import FLAIRModel; print('FLAIR import successful')"
python -c "import sys; sys.path.insert(0, '/root/DR/mysearch/MVCINN-master/MVCINN-code'); from models.create_models import my_MVCINN; print('MVCINN import successful')"
```

## Training

Run training with the default seed of 1001.

```bash
cd /root/DR/TPDOMDR-main
source .venv/bin/activate
python -u train.py
```

Run training with a custom seed. The generated files use the names `best_seed2002.pth` and `resume_seed2002.pth`.

```bash
python -u train.py --seed 2002
```

Run training in the background.

```bash
nohup python -u train.py > train.log 2>&1 &
tail -f train.log
```

## Evaluation

Evaluate the default `best.pth` checkpoint on all configured test datasets.

```bash
cd /root/DR/TPDOMDR-main
source .venv/bin/activate
python -u test.py --checkpoint best.pth --dataset all
```

Evaluate one dataset.

```bash
python -u test.py --checkpoint best.pth --dataset mfiddr
python -u test.py --checkpoint best.pth --dataset idrid
python -u test.py --checkpoint best.pth --dataset ddr
python -u test.py --checkpoint best.pth --dataset aptos
```

## Outputs

```text
outputs/
|-- checkpoints/
|-- logs/
`-- visualization/
```

The preprocessing scripts, testing scripts, and model weights will all be made publicly available upon acceptance.
