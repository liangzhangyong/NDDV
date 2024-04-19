# Neural Dynamic Data Valuation (NDDV)
This repository is the implementation of the paper  "Neural Dynamic Data Valuation"

## Limitations of traditional data valuation methods

- Learning algorithm is unknown prior to valuation
- Stochastic training process => Unstable values
- Model training => Computational burden

## Our Contributions

- A new data valuation method from the perspective of stochastic optimal control
- A novel marginal contribution metric
- Our method requires the training only once

## DCDV

The proposed NDDV method reformulates data valuation methods rooted in cooperative game theory into an optimal control problem within weighted mean-field conditions. This is accomplished by assigning weights to data points, which allows for the incorporation of data point heterogeneity within the mean-field control framework. To achieve this, we utilize a control equation resembling a simple quadratic linear model.

## Requirements

- python==3.10
- torch==2.1.0
- torchvision==0.16.0
- numpy==1.25.2
- matplotlib==3.8.0
- pandas==1.5.3
- pillow==10.0.1
- scikit-learn==1.3.1
- scipy==1.11.3
- tqdm==4.64.1
- tokenizers==0.15.0

## Getting started

To set up an experiment on examples: ./examples_nddv

## Contact Us

```
Liang Zhangyong (liangzhangyong1994@gmail.com)
```

## Related Repositories

[OpenDataVal](https://github.com/opendataval/opendataval) by Kevin Fu Jiang, Weixin Liang, James Zou, Yongchan Kwon.