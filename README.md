# Only the Curve Shape Matters: Training Foundation Models for Zero-Shot Multivariate Time Series Forecasting through Next Curve Shape Prediction

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

### Use GTT model for zero-shot forecast for your own data
It is rather straightforward to use GTT models for zero-shot forecast on your own data, check the [tutorial](./tutorial.ipynb).
