# Surrogate-based cross-correlation (SBCC) 

[![PoF](https://img.shields.io/static/v1?label=Physics of Fluids&message=36,087157&color=B31B1B)](https://doi.org/10.1063/5.0219706)
[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2112.05303&color=B31B1B)](https://arxiv.org/abs/2112.05303)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains code for the submitted paper __Surrogate-based cross-correlation for particle image velocimetry__.
In this work,  the SBCC is proposed to improve the cross-correlation performance via an optimized surrogate filter/image. Our broad SBCC encompasses several existing correlation techniques(PC, SPOF, RPC, etc) as special cases. Besides, the SBCC demonstrates the best robustness  by incorporating other negative context images.

### Motivation 
![movie](https://github.com/yongleex/SBCC/blob/cc21b363e036b3a9e40fed7d51d21e99a59a5a1a/for%20figure/motivation.png)
Inspired by [correlation filters](https://dl.acm.org/doi/book/10.5555/2520035), the surrogate image---supplanting original template images---will produce a more robust and more accurate correlation signal. That says, the surrogate is encouraged to produce a predefined Gaussian shape response to Image 1, and zero response to negative context images. As a result, the response between Image 2 and the surrogate could be accurate and robust.
Our SBCC framework is significantly different from the existing correlation methods by considering other negative context templates. More detailed info is referred to the paper [, Arxiv](https://arxiv.org/abs/2112.05303), [PoF](https://doi.org/10.1063/5.0219706).

## Install dependencies
```
conda install numpy matplotlib opencv seaborn
```



## The experiments
* [Exp1.ipynb](https://github.com/yongleex/SBCC/blob/master/Exp1.ipynb): Visualize the cross-correlation response map for synthetic images;
* [Exp2.ipynb](https://github.com/yongleex/SBCC/blob/master/Exp2.ipynb): Parameter sensitivity analysis;
* [Exp3.ipynb](https://github.com/yongleex/SBCC/blob/master/Exp3.ipynb): Test the performance with different flows;
* [Exp4.ipynb](https://github.com/yongleex/SBCC/blob/master/Exp4.ipynb): Test on the real PIV cases;
* [Exp5.ipynb](https://github.com/yongleex/SBCC/blob/master/Exp5.ipynb): Test computational cost for different images;

### BibTeX
```
@article{lee2024surrogate,
  title={Surrogate-based cross-correlation for particle image velocimetry},
  author={Lee, Yong and Gu, Fuqiang and Gong, Zeyu and Pan, Ding and Zeng, Wenhui},
  journal={Physics of Fluids},
  volume={36},
  number={8},
  year={2024},
  doi={10.1063/5.0219706},
  publisher={AIP Publishing}
}
```

### Questions?
For any questions regarding this work, please email me at [yongli.cv@gmail.com](mailto:yongli.cv@gmail.com), [yonglee@whut.edu.cn](mailto:yonglee@whut.edu.cn).

#### Acknowledgements
Parts of the code in this repository have been adapted from the following repos:

* [OpenPIV/openpiv-python](https://github.com/OpenPIV/openpiv-python)
* [opencv/opencv](https://github.com/opencv/opencv)