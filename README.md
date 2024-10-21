This repository provides the minimal code for running inference of GTT. 

# Getting Started

#### Install dependencies (with python 3.10) 

```shell
pip install -r requirements.txt
```

## Run Experiments

#### Run the zero-shot experiments

```shell
cd src
python test_zeroshot.py --gpu [GPUs] --batch_size [BS] --mode [mode] --data [DS] --uni [uni]
```
Specify mode to one of the following: tiny, small, large.

Specify data to one of the following: m1, m2, h1, h2, electricity, weather, traffic, ill.

Specify uni to 0 or 1, 0: multivariate forecast, 1: univariate forecast

#### Run the fine-tune experiments

```shell
cd experiments
python test_finetune.py --gpu [GPUs] --batch_size [BS] --mode [mode] --data [DS] --uni [uni] --epochs [eps]
```

## Use GTT models for zero-shot forecast on your own data

It is rather straightforward to use GTT models for zero-shot forecast on your own data (even with only CPUs), check the [tutorial](./tutorial.ipynb).


## Cite
Cheng Feng, Long Huang, and Denis Krompass. 2024. General Time Transformer: an Encoder-only Foundation Model for Zero-Shot Multivariate
Time Series Forecasting. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM ’24), October
21–25, 2024, Boise, ID, USA. ACM, New York, NY, USA, 5 pages. https://doi.org/10.1145/3627673.3679931
