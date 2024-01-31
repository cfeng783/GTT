# GTT

## Getting Started

#### Install dependencies (with python 3.10) 

```shell
pip install -r requirements.txt
```

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
