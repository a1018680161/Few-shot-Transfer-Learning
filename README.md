# Few-shot Transfer Learning for Intelligent Fault Diagnosis of Machine
PyTorch code for paper: [Few-shot Transfer Learning for Intelligent Fault Diagnosis of Machine](https://doi.org/10.1016/j.measurement.2020.108202)

# Data

For PU dataset experiments, please download [PU dataset](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter) and replace the 'root' in 'finetune_generator' amd 'transfer_generator' with your local data path

# Run

meta transfer learning 1 shot:

```
python main_meta_transfer.py -s 1
```

feature transfer 1 shot:

```
python main_finetune.py -s 1 -u 0
```

unfrozen 1 fine-tune 1 shot:

```
python main_finetune.py -s 1 -u 1
```

unfrozen 2 fine-tune 1 shot:

```
python main_finetune.py -s 1 -u 2
```

unfrozen 3 fine-tune 1 shot:

```
python main_finetune.py -s 1 -u 3
```
all fine-tune 1 shot:

```
python main_finetune.py -s 1 -u 4
```

direct training 1 shot:

```
python main_direct.py -s 1
```
you can change -b parameter based on your GPU memory.

## Citing

If you use this code in your research, please use the following BibTeX entry.

```
@article{WU2020108202,
title = "Few-shot Transfer Learning for Intelligent Fault Diagnosis of Machine",
journal = "Measurement",
pages = "108202",
year = "2020",
issn = "0263-2241",
doi = "https://doi.org/10.1016/j.measurement.2020.108202",
url = "http://www.sciencedirect.com/science/article/pii/S0263224120307405",
author = "Jingyao Wu and Zhibin Zhao and Chuang Sun and Ruqiang Yan and Xuefeng Chen",
keywords = "few-shot learning, intelligent diagnosis, transfer learning, meta-learning, rotating machinery",
abstract = "Rotating machinery intelligent diagnosis with large data has been researched comprehensively, while there is still a gap between the existing diagnostic model and the practical application, due to the variability of working conditions and the scarcity of fault samples. To address this problem, few-shot transfer learning method is constructed utilizing meta-learning for few-shot samples diagnosis in variable conditions in this paper. We consider two transfer situations of rotating machinery intelligent diagnosis named conditions transfer and artificial-to-natural transfer, and construct seven few-shot transfer learning methods based on a unified 1D convolution network for few-shot diagnosis of three datasets. Baseline accuracy under different sample capacity and transfer situations are provided for comprehensive comparison and guidelines. What is more, data dependency, transferability, and task plasticity of various methods in the few-shot scenario are discussed in detail, the data analysis result shows meta-learning holds the advantage for machine fault diagnosis with extremely few-shot instances on the relatively simple transfer task. Our code is available at https://github.com."
}
```

## Reference

[LearningToCompare_FSL](https://github.com/floodsung/LearningToCompare_FSL)
