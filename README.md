# Air-DualODE

This repository contains the PyTorch implementation of our ICLR'25 paper, **“Air Quality Prediction with Physics-Guided Dual Neural ODEs in Open Systems.”** In this work, we introduce **Air-DualODE** for predicting air quality at both city and national levels. Our model is composed of three key components: **Physics Dynamics, Data-Driven Dynamics,** and **Dynamics Fusion.**

🚩 News (2025.1) Air-DualODE has been accepted by ICLR 2025 (poster).

🚩 News (2025.10) We provide [create_graph.py](https://github.com/decisionintelligence/Air-DualODE/tree/main/Run/create_graph.py) for generating the graph_data.npz file on other air quality datasets. Feel free to test Air-DualODE on different datasets.

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

## Citation

If you find this repo useful, please cite our paper.

```
@article{tian2024air-dualode,
  title={Air quality prediction with Physics-Guided dual neural odes in open systems},
  author={Tian, Jindong and Liang, Yuxuan and Xu, Ronghui and Chen, Peng and Guo, Chenjuan and Zhou, Aoying and Pan, Lujia and Rao, Zhongwen and Yang, Bin},
  journal={ICLR},
  year={2025}
}
```

