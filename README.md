# MLDS2017 FinalProject

這是我們[報告](https://ntumlds.wordpress.com/2017/03/27/r05922018_drliao/)的實驗成果

### Experiments codes, architectures, results
#### ExperimentOne.ipynb
你可以在 ExperimentOne.ipynb 中找到
1. 正常的 MNIST 實驗
2. random labels 的 MNIST 實驗
3. Gaussian noise random input 的 MNIST 實驗

#### evaluate.ipynb
而 evaluate.ipynb 中

我們衡量了各種 DNN 配置的 accuracy


#### ExperimentTwo.ipynb
而未來在 ExperimentTwo.ipynb 裡

會放上我們第二階段報告的實驗結果

預計會多加上 batch size 作為考量

目前實驗正在仔細的規劃中

### Expeirments dataset
我們使用 [MNIST](http://yann.lecun.com/exdb/mnist/)

你可以去官方網站下載

或你可以利用 Keras 自帶的 MNIST dataset

### Expeirments environment

OS: CentOS Linux release 7.3.1611 (Core)

CPU: Intel® Xeon® CPU E3-1230 v3 @ 3.30GHz

GPU: GeForce GTX 980

Memory: 8GB DDR3

Python2.7.5

External libraries:
  * numpy 1.12.0
  * pandas 0.20.2
  * keras 2.0 using tensorflow backend
  * matplotlib 2.0.2
