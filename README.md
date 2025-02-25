# Air-DualODE

This repository contains the PyTorch implementation of our paper, **“Air Quality Prediction with Physics-Informed Dual Neural ODEs in Open Systems.”** In this work, we introduce **Air-DualODE** for predicting air quality at both city and national levels. Our model is composed of three key components: **Physics Dynamics, Data-Driven Dynamics,** and **Dynamics Fusion.**

![image-20250225200329134](./fig/Air-DualODE.png)

## Requirement

* python >= 3.9

```shell
pip install -r requirements.txt
```

## Data Preparation

Beijing: https://www.biendata.xyz/competition/kdd_2018/

KnowAir: https://github.com/shuowang-ai/PM2.5-GNN

## Train and Evaluation

```shell
cd Run
```

**Beijing**

```python
python train.py --config_filename ../Model_Config/Beijing/Air-DualODE_config.yaml --des 1
python eval.py --config_filename ../Model_Config/Beijing/Air-DualODE_config.yaml --des 1
```

**KnowAir**

```python
python train.py --config_filename ../Model_Config/KnowAir/Air-DualODE_config.yaml --des 1
python eval.py --config_filename ../Model_Config/KnowAir/Air-DualODE_config.yaml --des 1
```

