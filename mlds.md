# MLDS final projects

<b>UNDERSTANDING DEEP LEARNING REQUIRES RETHINKING
GENERALIZATION</b>

1~5 REF: https://arxiv.org/pdf/1611.03530.pdf
1~5 資料
- ICLR workshop 對此論文的反對意見：https://openreview.net/pdf?id=rJv6ZgHYg
- 知乎評論：https://zhuanlan.zhihu.com/p/26567289
- http://www.gooread.com/article/20121236746/
- no free lunch theorem


題目
---

1. “The effective capacity of neural networks is large enough for a brute-force memorization of the entire data set” 
2. “Even optimization on random labels remains easy” 
3. “Neural networks are able to capture the remaining signal in the data, while at the same time fit the noisy part using brute-force”。 
4. “DL中的explicit regularization（l1 l2 weight decay, dropout, data augmentation等）或許可以提高模型的泛化能力，但並非必須，也不是網絡泛化能力的核心保證”：原因是即使不用這些正則方法，神經網絡通常也能在測試集上得到一個不錯的結果，並且泛化能力良好，而使用這些正則方法，也不能保證網絡在訓練時不overfit，這裡可以看出explicit regularization對神經網絡的影響其實是遠不及shallow model的。 
5. “Generically large neural networks can express any labeling of the training data”：原文舉例說可以用包含p = 2n + d個參數的2層ReLU網絡來模擬n個d維樣本的任何標籤可能，但這只是一個網絡表達能力的理論上界，作者沒有繼續討論要達到這個上界需要的條件，以及p、n、d的大小關係對論文核心問題也就是網絡generalization performance的影響。


6. 有關 batch 的問題
    - 假設你知道 testing data 的 distribution，那在 training 的時候，我們是否也能按照這個 distribution 來組成每一個 batch，用跟 testing data 有著相同 distribution 的 batch 來 train，這樣 train 出來的 model 是否更為 robust ? 
    - 還是直接用 batch normalization 就能得到好效果？
    - batch 要怎麼取？直接取一個很小的 batch 是否能夠得到好結果？
    - batch 在訓練過程中能否改變？


可行實驗
---

- 驗證 NN 的記憶性
可以參考 [這篇 paper](https://openreview.net/pdf?id=rJv6ZgHYg) 設計的實驗：
    - 說明：NN 是否靠 brute-force 來記住資料，還是會先學ㄧ些通用的 pattern，來快速降低 training loss，再靠額外的空間來記住 noise。
    - step1: 設置一個參數多於資料的 NN
        - 是否 NN 參數比 data 多就會用 brute-force 來記，還是仍然會以學 pattern 為主？
    - step2: 設置一個參數少於資料的 NN
        - 是否 NN 參數比 data 少，就會傾向學習 pattern，還是仍然會用 brute-force 來記？
    - step3: 用不同的 noise 比例來測試前兩組設置
        - 看看兩組設置是否需要更多的參數/時間才能達到原本的準確度
    - 如果 NN 學到 pattern，那 NN 具有 generalization 能力並不意外，但如果 NN 是靠 brute-force 硬記，那他的 generalization 能力從何而來
    - 假設 NN 是先記 pattern 再用額外空間記住 noise，那麼有加噪音跟沒加的 distribution 長得不同。但為何 testing 的時候 accuracy 卻不會差很多？難道 NN 可以知道 input 是否為 noise?

## 衝鋒
### Reproduce實驗 on cifar10 & imagenet .. mnist(該論文沒做)
#### 驗證題目1, 2
https://github.com/csdai0324/MLDS2017FinalProject
-  FITTING RANDOM LABELS AND PIXELS
    - True labels: the original dataset without modification.
    - Partially corrupted labels: independently with probability p, the label of each image is corrupted as a uniform random class.
    - Random labels: all the labels are replaced with random ones.
    - Shuffled pixels: a random permutation of the pixels is chosen and then the same permutation is applied to all the images in both training and test set.
    - Random pixels: a different random permutation is applied to each image independently.
    - Gaussian: A Gaussian distribution (with matching mean and variance to the original image dataset) is used to generate random pixels for each image.
#### 題目5簡單先做


### IMPLICIT REGULARIZATION: AN APPEAL TO LINEAR MODELS, p.9 
> Unfortunately, this notion of minimum norm is not predictive of generalization performance. For example, returning to the MNIST example, the `2-norm of the minimum norm solution with no preprocessing is approximately 220. With wavelet preprocessing, the norm jumps to 390. Yet the test error drops by a factor of 2. So while this minimum-norm intuition may provide some guidance to new algorithm design, it is only a very small piece of the generalization story

