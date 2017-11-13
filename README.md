# Rethinking Generalization of Deep Learning
###### tags: `deep learning` `generalization`

## 主題 Topic

__~~李宏毅老師常說：「先別管那麼多，硬train一發就對了!」~~__ 


<div align='center'>
    <center>
    <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='https://i.imgur.com/oNFGPCQ.png' padding='5px'></img>
    </center>
    <center>
    <a href='https://xkcd.com/1838/'>Image src</a>
    </center>
</div>
自從 AlexNet 在 ILSVRC2012 大放異彩後，開啟了學界業界對深度神經網路高度重視的濫觴。
深度神經網路被應用到各種領域的 task 中，
驚人的是，深度神經網路幾乎都能取得不輸，甚至超越舊有方法的結果。

深度神經網路能取得如此成績，
很大一部分要歸因於其對數據的擬合能力，
以及在未知數據上重現近似已知數據上的準確度，
我們稱之為泛化 (generalization) 能力。

但，深度神經網絡強大泛化能力的真正原因是什麼？

最近，有一篇論文在 Quora、medium、知乎等等論壇上受到廣泛的討論，這篇論文叫做 Understanding Deep Learning Requires Rethinking Generalization，是 ICLR2017 Best Paper Award 得主之一。它針對深度神經網路的泛化能力做了一連串實驗，並得出了一個顛覆傳統認知的結論：深度神經網路非常的龐大，龐大到足以用 brute-force 的方式記住每一筆 training data！

此結果引起學界一片譁然，畢竟在過往的認知中，深度神經網路應該會去學習 data 中的 pattern。果不其然，在 ICLR2017 workshop 上，另一個團隊發表了這篇論文：Deep Nets Don’t Learn via Memorization，同樣利用一系列實驗，來反駁上一篇論文提出的結論。

一個有意思的事情是，Google Brain 的 Samy Bengio 是第一篇論文的共同作者，
其兄弟 Emmanuel Bengio 則是第二篇論文的共同作者。
究竟，兩位 Bengio 支持的論點誰是誰非?
是否有方法判斷或檢驗兩篇論文提出的結論和實驗?
相信這會是一個很有趣、很有意義的嘗試。

以下迅速說明這篇報告的大綱：
在 section2，我們會盡可能詳盡的說明這兩篇論文提出的觀點以及嘗試過的實驗。
在 section3，會包含我們對這兩篇論文的看法與討論。
我們設計的實驗及其結果會放在 section4。
得到的結論與參考資料分別放在 section5 & section6。

## 資料搜集 Background & Related Work

### Understanding Deep Learning Requires Rethinking Generalization
深度神經網路的參數量通常遠大於訓練樣本的數量，一般而言，越多的參數也代表著越容易發生過擬合。然而，很多神經網路模型所表現出來的泛化(generalization)能力令人驚嘆！

__MIT__ __Chiyuan Zhang__ 的論文[Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/pdf/1611.03530.pdf)在ICLR 2017得到Best Paper Award，該篇論文的主要內容是以實驗結果討論神經網路模型的學習能力以及傳統的泛化觀點並不能解釋的泛化能力。

__該篇論文的貢獻在於：作者提出幾個實驗，並詳盡的探討實驗結果。__

論文中使用了兩個圖片資料集進行實驗: CIFAR10 和 ImageNet。
第一組實驗是把randomization tests套用在幾個標準的模型上(MLP, AlexNet, GoogleNet)，實驗內容主要是將training labels或training image的pixel值作更動再進行訓練。

#### Fitting random labels

論文中使用了兩個圖片資料集進行實驗：CIFAR10 和 ImageNet。
第一組實驗是把 randomization tests 套用在幾個標準的模型上 (MLP, AlexNet, GoogleNet)，實驗內容主要是將 training labels 作更動再進行訓練。
而更動 training labels 的實驗設置如下：

- True labels: the original dataset without modification.
- Partially corrupted labels: independently with probability p, the label of each image is corrupted as a uniform random class.
- Random labels: all the labels are replaced with random ones.

在實驗後，作者觀察到有趣的現象：神經網路可以輕易擬合隨機標籤。
意即就算使用 random labels 的 training data (簡單的說就是亂給解答)，模型依然會收斂，達到 0 的訓練誤差。而測試誤差理所當然會很高，因為 training labels 和 testing labels 之間沒有相關性。

![](https://i.imgur.com/n9WO7X0.png)

換句話說，通過單獨隨機化labels，可以強制模型的泛化誤差(generalization error, training error及test error的差距）變得很大，而不改變模型capacity，hyperparameters或optimizer。

#### Randomized labels test結論
> 1.神經網絡的capacity非常大，大到能夠以暴力記憶整個data set。
>
> 2.就算使用random labels，進行優化依然很容易。實際上，使用 random labels的訓練時間與使用true labels進行訓練相比，僅僅差了一個小的常數因子。
>
> 3.標籤隨機化僅僅是一種數據轉換，學習問題的其他性質仍保持不變。

#### Fitting random pixels

除了對 labels 做文章，作者也試著更動 training data pixels，實驗設置如下：

- Shuffled pixels: a random permutation of the pixels is chosen and then the same permutation is applied to all the images in both training and test set.
- Random pixels: a different random permutation is applied to each image independently.
- Gaussian: A Gaussian distribution (with matching mean and variance to the original image dataset) is used to generate random pixels for each image.

延伸上面所說的random labels實驗，通過random pixels（例如Gaussian noise）產生的image替代true image，並觀察到卷積神經網絡繼續適應零訓練誤差的數據。這代表儘管它們的結構不同，卷積神經網絡依然可以適應隨機噪聲。 

此外，實驗進一步改變randomized image的數量，在無noise和全noise的情況下平滑地內插。 這導致一系列中間學習問題，其中標籤中存在一定程度的signal。隨著noise比例提高，可以觀察到的泛化誤差穩定惡化。

這表示： __神經網絡能夠捕獲資料中的剩餘signal，同時使用brute-force來適應噪聲部分。__

![](https://i.imgur.com/n9WO7X0.png)

#### Implications of randomization test

- Rademacher complexity and VC-dimension
- Uniform stability

#### The role of explicit regularization

通常模型參數很多時，加入explicit regularization是一種防止overfitting的方法，否則hypothesis space會很大。而regularization是一種防止overfitting的方法，能把hypothesis space縮小。

如果模型架構本身泛化能力較差，那我們就能觀察到explicit regularization對於模型的作用有多大。該篇論文設計了一系列對於explicit regularizations方法(例如weight decay, dropout和data augmentation)的實驗，發現這些方法不能充分的解釋神經網路的泛化誤差。

==Note. 在原本的randomization tests沒有使用regularization==

![](https://i.imgur.com/8a24f5b.png)

> Explicit regularization通常能提高泛化性能，但並不是神經網路泛化能力的關鍵。

根據上方表格紀錄的實驗數據，作者給出了觀點：Explicit regularization 通常能提高泛化性能，但並不是神經網路泛化能力的關鍵。
對比傳統的 convex empirical risk minimization，explicit regularization 必須排除 trivial solutions，作者認為 explicit regularization 在神經網路中扮演著完全不同的角色。它做的更多的是調整參數，通常有助於提高模型的最終 test error，但缺少正則化並不一定意味著較大的泛化誤差，如上表所示。

對比傳統的convex empirical risk minimization，explicit regularization必須排除trivial solutions，作者認為explicit regularization在神經網路中扮演著完全不同的角色。它做的更多的是調整參數，通常有助於提高模型的最終test error，但缺少正則化並不一定意味著較大的泛化誤差，如上表所示。

#### Finite sample expressivity

論文用理論結構補充實驗觀察結果，顯示一般的大型神經網路能夠表達任意標籤的資料。

==Note. 見原文appendix. C==

只要使用一個簡單的2層RELU-MLPs網路，其中的參數量 p = 2n + d 可以表達 n 個 d 維樣本的任意標籤。也可以使用一个深度為 k 的網路，其中每層只有 O（n / k) 個參數。

#### The role of implicit regularization

雖然explicit regularizers可能對於泛化可能不是必要的，但並不是所有適合訓練數據的模型都有好的泛化能力。

事實上，在神經網絡中，我們總是選擇SGD作為optimizer。所以作者也做了實驗分析SGD在線性模型中如何作為implicit regularizers。對於線性模型，SGD總是收斂到一個小norm的solution。可見算法本身將solution做了implicit regularization。實際上，在小的data set，Gaussian kernel methods沒經過regularization也可以擁有不錯的泛化能力。

作者在這方面並沒有著墨太多，不過實驗對於SGD, early stop和batch normalization在線性模型中發揮的implicit regularization作用很有啟發性。


![](https://i.imgur.com/DSXbI2s.png)

#### Paper Conclusion

在這篇論文中，提出了一個簡單的實驗框架，用於定義和理解機器學習模型有效 effective capacity 的概念。論文中進行的實驗強調，幾個成功的神經網絡架構的 effective capacity 夠大，可以暴力記憶所有訓練數據。

這種情況對傳統統計學習理論構成了概念上的挑戰，因為傳統的模型複雜度量度無法充分的解釋大型人造神經網絡的泛化能力。

作者認為，我們還沒有發現一個精確且正式方法，在這些方法下，這些巨大的模型夠單純。從實驗得出的另一個見解是，即使最終的模型不能很好的泛化，優化仍然在經驗上的依然容易。這也代表為什麼經驗優化容易的原因必須與真正的泛化原因不同。。

### Deep Nets Don't Learn via Memorization

上面那篇[Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/pdf/1611.03530.pdf)的二作就是Google Brain的 ___Samy Bengio___ ，而在同年的ICLR Workshop，他的兄弟 ___Emmanuel Bengio___ 投了[Deep Nets Don't Learn via Memorization](https://openreview.net/pdf?id=rJv6ZgHYg)這篇論文，對神經網路記憶理論及實驗結果提出異議：DNN不會暴力記住所有資料。

==Note. 這兩人的另一位兄弟就是大名鼎鼎的Yoshua Bengio，真可謂機器學習家族。==

該篇論文使用empirical方法論證了深度神經網絡不會通過記憶訓練數據來實現他們的效能。相反的，這些網路學習一個適合有限數據樣本的簡單可用hypothesis。

為了支持這一觀點，作者認為在學習noise與真實資料集時存在著一定性的差異：
> 1.需要更多的capacity來適應noise。
> 
> 2.random labels的收斂時間更長，但是對於random inputs收斂時間則更短。
> 
> 3.通過收斂時的loss functions的銳度所測量，用真實數據訓練的DNN比使用noise訓練能得到更簡單的函數。 
> 
> 4.證明了對於適當調整的explicit regularization，例如，dropout可以降低noise data set上的DNN訓練性能，而不會影響實際數據的泛化。
> 

#### Learning from data vs. noise


- MNIST
- 2層ReLU-MLPs
- SGD, learning rate 0.01
- 1000epochs 

![](https://i.imgur.com/PiEmCUm.png)

有別於上一篇論文所做的平滑內差實驗，這裡不僅僅改變noise level，也改變模型的capacity。

可以發現隨著noise比例上升，模型要達到相同效能所需要的capacity也增加。這也可能代表模型從真實數據中學到了某種pattern，而對於剩下的noise才使用暴力記憶。

#### Complexity of the learned function

基於很多前人的論證，包括同為ICLR 2017的一篇[oral paper](https://arxiv.org/abs/1609.04836)，loss function的flat minima具有更好的泛化性質。Keskar等人(2017)認為，當batch size小時，SGD能學習到flat minima，這解釋了實驗中所發現的，較小的batch size能產生更好的泛化，並將其與新穎的flatness相關聯。

![](https://i.imgur.com/ECDe3Cb.png)

作者研究了data set中不同數量的noise對local minima學習function的sharpness和flatness。為了表示sharpness，使用（LeCun等人，1993）的[方法](http://www.bcl.hamilton.ie/~barak/papers/nips92-lecun-nofigs.pdf)近似Hessian矩陣最大特徵值的norm。

==Note. Hessian矩陣的特征值代表其在該點附近特征向量方向的凹凸性==

當增加training data中的noise，作者觀察到在loss function的local minima的sharpness增加，為了擬合noise導致需要學習更複雜的function。

![](https://i.imgur.com/OIFplUd.png)

可以看到减小模型capacity或者增加data set的大小會使收斂速度變慢，但對使用真實數據的影響並不明顯，這同樣可能代表模型學到了某種pattern，而不是僅靠暴力記憶。

![](https://i.imgur.com/THzshJ1.png)

圖左可以看到noise對random labels和random input，對Hessian矩陣最大特徵值的norm造成的變化，即sharpness的變化。

#### Effect of regularization on learning

評估explicit regularization在random labels上降低training performance的能力，同時保持實真實資料的泛化性能。

具體來說，在CIFAR-10上用一個真實的或隨機的標籤來訓練一個小型類似Alexnet風格的CNN。比較以下regularizer：
- Dropout（0-.9)
- Input drop out （0-.9）
- Gaussian noise
- Weight decay

上圖右發現drop out最能夠妨礙暴力記憶，而不會降低模型的學習能力。

#### Paper Conclusion

根據經驗表示，noise和真實資料的訓練上存在著差異，這些實驗結果顯示了DNNs用到了簡單的hypotheses，而不是靠著記憶去擬合標籤。

儘管如此，由於DNN具有擬合noise的有效capacity，因此不清楚為什麼它們可以在真實資料上恢復generalizable solutions。作者假設這是因為發現pattern比暴力記憶更容易。

## 討論 Discussion

我們發現了在[Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/pdf/1611.03530.pdf)中，並沒有用MNIST做random labels的實驗，有人在[知乎評論](https://www.zhihu.com/question/56151007)中提出了MNIST樣本的差異性太小而不易擬合training data set，意即手寫數字樣本間的不同很小，太難讓模型以random label去暴力記憶，論文實驗中使用的3xMLPs無法擬合training data set。

而在[Deep Nets Don't Learn via Memorization](https://openreview.net/pdf?id=rJv6ZgHYg)論文中，該組確實以mnist做了實驗，只是使用的noise level是將training data set的部份以gaussian noise input取代，意即出現了有差異性的樣本，同時增加模型的capacity。

### 我們第一個想知道的事

Random labels的mnist data set能不能訓練至收斂呢？

根據[Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/pdf/1611.03530.pdf)提出的 __Finite sample expressivity__：

> 只要使用一個簡單的2層RELU-MLPs網路，其中的參數量p = 2n + d可以表達 n個d維樣本的任意標籤。也可以使用一个深度為k的網路，其中每層只有 O(n/k)個參數。

但這可能只是一個網路表達能力的理論上界，論文中並沒有繼續深入要達到這個上界需要的條件，以及p、n、d的大小關係對generalization performance的影響。

對MNIST training data set，我們計算出能夠模擬任何標籤的參數量p = 2 * 50000 + 784，即p = 100784，遠小於使用MLP 3x512的參數數量。

==Note. 根據上面圖表 MLP 3x512 #params 1,735,178==

既然遠大於理論上需要的參數，那為什麼又有人指出可能無法擬合導致不收斂？

那在[Deep Nets Don't Learn via Memorization](https://openreview.net/pdf?id=rJv6ZgHYg)參數不多的兩層RELU-MLPs能夠收斂，是不是也印證了真實資料、部份gaussian noise與random labels即是因為其資料上的差異性而讓模型有不同學習策略。

這樣看來，DNNs可能是學習patterns再將noise以增加capacity以暴力記憶，比較貼近[Deep Nets Don't Learn via Memorization](https://openreview.net/pdf?id=rJv6ZgHYg)中所說。而[Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/pdf/1611.03530.pdf)所說的全noise training的情況，就是因為無patterns可學，但模型能大到記住所有差異性夠大的noise。

畢竟，我們都期待DNNs不只是單純的靠著暴力記憶，僅僅只是學習一個形狀很奇怪的函數。

### 第二件有趣的事

在資料中提到同為ICLR 2017的一篇oral paper [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836)中，作者發現SGD及其變種在batch size增大的時候會有泛化能力的明顯下降的現象，而沒有找到確切的原因。 

論文的實驗結果顯示了，因為梯度估計的內在噪聲(inherent noise)，batch size越大，越有可能收斂到比較sharp的local minima。batch size越小，越有可能收斂到比較flat的local minima。

作者還討論了幾種empirical strategies，幫助large batch消除泛化差距，並得出了一組未來的研究思路和public questions。

- Can one prove that large-batch (LB) methods typically converge to sharp minimizers of deep learning training functions? (In this paper, we only provided some numerical evidence.)
- What is the relative density of the two kinds of minima?
- Can one design neural network architectures for various tasks that are suitable to the properties of LB methods?
- Can the networks be initialized in a way that enables LB methods to succeed?
- Is it possible, through algorithmic or regulatory means to steer LB methods away from sharp minimizers?

ICLR2017中，持反對意見的論文: [Sharp Minima Can Generalize For Deep Nets](https://arxiv.org/abs/1703.04933)

## 實驗設計 Experiment Design

根據周志華的[機器學習](http://www.tup.tsinghua.edu.cn/upload/books/yz/064027-01.pdf)：

> NFL(No free lunch theorem)定理最重要的寓意，是讓我們清楚地認識到，脫離具體問題，空泛地談論”什麼學習算法更好“毫無意義，因為若考慮所有潛在的問題，則所有的算法一樣好. 要談論算法的相對優劣，必須要針對具體問題；在某些問題上表現好的學習算法，在另一問題上卻可能不盡如人意，學習算法自身的歸納偏好與問題是否相配，往往會起到決定性作用。

所以在實驗中，我們選擇在[Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/pdf/1611.03530.pdf)中沒有使用的mnist，去驗證在網路上流傳使用random labels無法收斂的情況。

mnist資料集包含的共70000張image: 60000張training image和10000張testing image。每個image是一個單一的手寫的數字的數位化的圖片。由0~255的灰階值表達28x28個pixel。

基於討論中提到的capacity，我們設計了簡單的MLP，把隱藏層的單元數量和層數作為可調參數，以相同的learning rate及batch size去實驗，除了看是否收斂或多大的capacity能夠收斂之外，也可以觀察相同的參數量時，比較增加隱藏元數量和加深層數深度有沒有不同。

```python=
batch_size = 1024
learning_rate = 0.01
hidden_size = [16, 128, 512, 1024]
layer = [2, 4, 6, 8, 10]

for size in hidden_size:
    for l in layer:
        train
        ...
        ...
```

先以沒有做random labels的原始mnist實驗，作為對照組，再測試random labels的mnist，觀察兩者結果的不同。

雖然不能肯定，但我們認為[Deep Nets Don't Learn via Memorization](https://openreview.net/pdf?id=rJv6ZgHYg)中所說的，學習pattern後以capacity記憶噪聲是可能的，所以我們重現[Deep Nets Don't Learn via Memorization](https://openreview.net/pdf?id=rJv6ZgHYg)中使用的noise level，即gaussian noise在dataset中所佔的比例作為參數，並將模型深度同時納入觀察。

```python=
gaussian_proportion = [.0, .2, .4, .6, .8, 1.]
hidden_size = [16, 512, 1024]
layer = [2, 4, 8]

for proportion in gaussian_proportion:
    for size in hidden_size:
        for l in layer:
            train
            ...
            ...
```

而對於Finite sample expressivity，即:

> 只要使用一個簡單的2層RELU-MLPs網路，其中的參數量 p = 2n + d 可以表達 n 個 d 維樣本的任意標籤。也可以使用一个深度為 k 的網路，其中每層只有 O（n / k) 個參數。

如討論中提到的，在論文中並沒有以實驗佐證，也沒有繼續深入p、n、d的大小關係對generalization performance的影響。

我們設計了一個實驗將n和d作為可改變的參數，並從normal distribution中sample製造出data set，再實現2層RELU-MLPs且參數為p的網路，和深度為 k 的網路，其中每層只有 O（n / k) 個參數去驗證。

[github](https://github.com/csdai0324/MLDS2017FinalProject)
### 實驗環境 Experiment Environment

- OS: CentOS Linux release 7.3.1611 (Core)
- CPU: Intel(R) Xeon(R) CPU E3-1230 v3 @ 3.30GHz
- GPU: GeForce GTX 980
- Memory: 8GB DDR3
- Python2.7.5
- External libraries:
    - numpy 1.12.0
    - keras 2.0 using tensorflow backend
    - matplotlib 2.0.2
    
### 代辦事項 Todo List

把batch size加入泛化考量，設計解決[On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836)作者提出的pubilc questions的實驗。

- Can one prove that large-batch (LB) methods typically converge to sharp minimizers of deep learning training functions? (In this paper, we only provided some numerical evidence.)
- What is the relative density of the two kinds of minima?
- Can one design neural network architectures for various tasks that are suitable to the properties of LB methods?
- Can the networks be initialized in a way that enables LB methods to succeed?
- Is it possible, through algorithmic or regulatory means to steer LB methods away from sharp minimizers?


## 實驗結果討論 Experiment Result and Discussion

### 正常的 MNIST 實驗結果做為對照組

我們先以正常的，未動過任何手腳的 MNIST dataset 來做實驗，當成對照組。

首先進行三個 warm up 實驗。
我們順手驗證了一個常見的流言：高瘦的網路 (層數多每層神經元少) 會比矮胖的網路 (層數少每層神經元多) 更厲害。

在前兩張圖可以看到，高瘦的網路在參數量差不多甚至更少時都能更快收斂，這成功驗證了流言。話雖如此，在圖三中，我們發現如果每一層的神經元如果太少，會導致梯度不穩定，加深網路並不會讓你得到更好的結果。因此，我們可以對此流言增加一個附加條件：高瘦的網路效果好，但每一層的神經元還是不能太少。

在圖四到圖八以及圖十三到圖十七中，我們 fix 住網路的層數，只改變每一層神經元個數。
在圖九到圖十二以及圖十八到圖二十一中，我們 fix 住每一層神經元個數，只改變層數多寡。
我們發現在幾乎所有的配置下，神經網路都能很好的擬合 training data。

如果依照 Understanding Deep Learning Requires Rethinking Generalization 提出的 Finite sample expressivity 性質中給定的公式，要擬合 MNIST training set 需要 2*60000 + 784 = 120784 個參數 (n=60000, d=784)。我們可以看到，圖四中黃色線代表的神經網路，參數稍少於這個數目，最後成功擬合了數據。因此我們可以說，在 MNIST 的 training set 上，Finite sample expressivity 是成立的！

{%slideshare BrianHuang34/mnist-76914305 %}

### Random labels 的 MNIST 實驗結果

在這個實驗中，我們將原本 MNIST 中的 label 完全打亂。因此，training data 的 image 與 label 間將不再有任何關聯。

我們一樣做三個 warm up 實驗。
在前兩張圖可以看到，高瘦的網路在參數量差不多時，有著碾壓性的表現，而在參數量只有一半時，仍有相似的擬合能力。而圖三則顯示如果每層神經元太少，加深網路的意義不大。
這裡藏著一個值得注意的信號，圖三中兩個網路的準確率幾乎是完全不上升的，與對照組相比，在一樣都只有那麼少參數的情況下，對照組仍然能達到超過 90% 的準確度。
還記得我們在 section3 discussion 中提出的假設：”DNNs 可能是學習 patterns 再將 noise 以增加 capacity 的方式暴力記憶” 嗎？在對照組中，雖然參數少，但因為學到了某些 data 中的 pattern，因此可以達到一定的準確度，並且有可能是因為參數量不足以完整學習 pattern，才沒有達到 training error = 0%。但在 random label (可以想成全部 data 都是 noise) 的情況下，因為沒有 pattern 可學，所以轉成暴力記憶，卻又因參數不夠，導致訓練失敗。

{%slideshare BrianHuang34/random-labels-mnist %}

在圖四到圖八以及圖十三到圖十七中，我們 fix 住網路的層數，只改變每一層神經元個數。
在圖九到圖十二以及圖十八到圖二十一中，我們 fix 住每一層神經元個數，只改變層數多寡。
我們可以看到，在 random labels 的實驗中，模型的收斂情況明顯和對照組不同，並不是所有配置都收斂，神經網路必須要有足夠的 effective capacity 才能順利收斂，而收斂的速度、強度也都與參數數量高度正相關。

再回來看看 Finite sample expressivity 性質，在圖四中的各個網路，即使是參數量最多，將近 200W 的神經網路，都沒辦法很好的擬合數據集。因此我們可以說，在 random labels (noise 很大) 的 MNIST training set 上，Finite sample expressivity 是不成立的！

### Gaussian noise random inputs 的 MNIST 實驗結果

在 random labels 實驗中，透過實驗結果我們發現 DNNs 可能是學習 patterns 再將 noise 以增加 capacity 的方式暴力記憶。
我們配置了九種神經網路，並對 training data 加上不同程度的 Gaussian noise，來觀察神經網路對 noise 的擬合程度，來進一步驗證我們的假設。

讓我們仔細觀察實驗結果，當神經網路參數很少時，對噪聲是幾乎沒有抵抗力的，而參數量越多的神經網路，面對 noise 的表現越好。
我們也可以清楚看到，同樣的神經網路在面對越高的 noise 程度時收斂越慢。
到此，我們幾乎可以很篤定的說，DNNs 會先去學習資料中的 pattern，再用額外的 capacity 去記憶 noise。這樣才能解釋為何收斂時間隨著 noise 升高成正比，因為需要靠暴力去記的東西變多了嘛！

此外，比較圖三、五的神經網路，可以再次應證高瘦的網路比矮胖的網路要好。

{%slideshare BrianHuang34/gaussian-noise-random-inputs-mnist %}

為了加強我們的論證，我們也在 cifar10 中加入了 Gaussian noise 來觀察。
其結果跟在 MNIST 上很相似：同樣的神經網路在面對越高的 noise 程度時收斂越慢。
而因為 cifar10 的圖片比 MNIST 複雜得多，因此在收斂階段會有比較大的震盪。

{%slideshare BrianHuang34/gaussian-noise-random-inputs-cifar10 %}

下圖是這九個神經網路在 MNIST 中對不同 noise level 的 testing accuracy。可以看到圖中產生了一個巨大的 gap，在全 noise 的情況下，所有架構都沒有能力判別，就算是足夠大的 effective capacity 也找不到合適的 hypothesis，即使它們在訓練時是收斂的。
而當 noise 與原始資料共存時，noise 越大，testing accuracy 也越低，但還是與全 noise 存在明顯的 accuracy gap。
讓我們比較 gaussian=1.0 以及 gaussian=0.8，如果依照 Understanding Deep Learning Requires Rethinking Generalization 的說法，DNNs 是靠 brute-force 來記憶，那就無法解釋這 20% 的 noise 為何會造成如此巨大的泛化誤差，我們更願意相信，即使存在噪聲，DNNs 還是會試著尋找 data 裡的 pattern，只是在噪聲越大，真實資料越少的情況下，學到的 pattern 品質也會下降。而在全噪聲時，完全無 pattern 可學，只能靠暴力記憶，如果 capacity 夠大，還是能擬合 training data，但在 testing data 上會因沒有學到 pattern 而得到很差的準確度。

![](https://i.imgur.com/UIm3iwK.png)

而在 cifar10 中，九種神經網路對不同 noise level 的 testing accuracy 如下，與在 MNIST 中的結果相似，全 noise 與部分 noise 也出現了 accuracy gap。只要 data 不是全 noise，神經網路就有能力學到某些 patterns，得到一定程度的泛化能力，進而在 testing 階段提高準確率。而在全部 noise 的情況下，沒有 pattern 可學，雖然在 training 時可以靠 brute-force 記住 noise 達到收斂，但在 testing 階段，神經網路的準確率就跟 random predict 一樣低。

![](https://i.imgur.com/5WnC5KN.png)


## 結論 Conclusion

讓我們重新整理一次得出的結論：

-    高瘦的網路比矮胖的網路更優，但每層神經元個數仍不能太少。
-    DNNs 會先去學習資料中的 pattern，再用額外的 capacity 去記憶 noise。
-    Noise 程度越高，DNNs 學到的 pattern 質量越低，testing accuracy 越低。
-    Capacity 越多，學 pattern & 記憶 noise 的速度都越快。


深度學習儼然成為一門實驗科學，在深度神經網路的學習及泛化上，現今的實驗結果即推論假設依然缺乏理論上的驗證。就算是領域中幾個巨頭也抱持著不同的看法，百家爭鳴，我們所閱讀的資料和論文幾乎都只是從實驗結果去推論，而且得到的結果往往考驗著傳統機器學習的理論價值。

我們認為[Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/pdf/1611.03530.pdf)和[Deep Nets Don't Learn via Memorization](https://openreview.net/pdf?id=rJv6ZgHYg)以及其他許多相關論文中的實驗結果裡並沒有對錯之分，可能因為這門科學還未發展成熟，必定會有不確定因素在訓練過程裡面，影響著實驗著結果。

在理論與實驗之間存在著差異，[讓我們的思想一起發酵：記深度學習中的層級性 甄慧玲](http://www.gooread.com/article/20121236746/)中提出了很有意思的看法，作者是應數及統計物理學博士出身，以hierarchical為出發點解釋神經網路的學習以及泛化。

越是深入了解才越發覺這個題目裡面的坑是如此的大，卻也非常迷人，我們有幸乘著這波浪潮，去慢慢了解一門新興科學，並看到在這門科學一步步的新發展中和傳統理論激起火花。

## 參考資料 Reference

### Papers
- [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/pdf/1611.03530.pdf)
- [Deep Nets Don't Learn via Memorization](https://openreview.net/pdf?id=rJv6ZgHYg)
- [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836)
- [Sharp Minima Can Generalize For Deep Nets](https://arxiv.org/abs/1703.04933)
### Articles
- [知乎](https://zhuanlan.zhihu.com/p/26567289)
- [讓我們的思想一起發酵：記深度學習中的層級性 甄慧玲](http://www.gooread.com/article/20121236746/)
- [no free lunch theorem](https://en.wikipedia.org/wiki/No_free_lunch_theorem)
- [Quora](https://www.quora.com/Why-is-the-paper-“Understanding-Deep-Learning-required-Rethinking-Generalization“-important)
- [Medium](https://medium.com/intuitionmachine/rethinking-generalization-in-deep-learning-ec66ed684ace)
