# 机器学习概述
## 机器学习分类
1. 监督学习：已经有数据，和数据对应的标签。
2. 非监督学习：给定的样本无需输出/标签，让机器自己学习样本中隐含的内部结构。
3. 半监督学习：二者结合。
4. 强化学习：通过打分/评价的形式，类似于监督学习中的标签。

## 机器学习模型
机器学习 = 数据 data + 模型 model + 优化方法 optimal strategy

## 偏差/方差权衡
variance 和 bias，分别对应过拟合和欠拟合

来自 Wikipedia：
> 在监督学习中，如果能将模型的方差与误差权衡好，那么可以认为该模型的泛化性能（对于新数据）将会表现出好的结果。

>偏差刻画的是算法本身的性能。高偏差将会造成欠拟合(Underfitting) [miss the relevant relations between features and target outputs]。换句话说，模型越复杂偏差就越小；而模型越简单，偏差就越大。

>方差用来衡量因训练集数据波动(fluctuations)而造成的误差影响。高方差将会造成过拟合(Overfitting)。

在周志华老师<机器学习>书中是这样阐述的：

>*偏差* 度量了学习算法的期望预测与真实结果的偏离程度，即刻画了算法本身的拟合能力；

>*方差* 度量了同样大小的训练集的变动所导致的学习性能的变化，即刻画了数据扰动所造成的影响；

>*噪声* 则表达了在当前任务上任何学习算法所能达到的期望泛化误差的下界，即刻画了学习问题的本身难度

>偏差-方差分解说明，泛化性能是由学习算法的能力、数据的充分性以及学习任务本身的难度所共同决定的。给定的学习任务，为了取得好的泛化性能，则需使偏差较小，即能够充分拟合数据，并且使方差较小，即使数据扰动产生的影响小。一般来说方差与偏差是有冲突的，这称为方差-偏差窘境。


## 常见机器学习算法概览
### 1. Linear Algorithm 线性算法
1. **Linear Regression 线性回归**：使用最小二乘法 Least Squares 拟合一条直线 → 计算 R<sup>2</sup> → 计算 R<sup>2</sup> 的 p 值。R<sup>2</sup> 表示 x 能多大程度反映 y 的变化，p 值表示可靠程度。拟合直线的过程使用「随机梯度下降」（SGD）

2. **Lasso 回归 和 Ridge 回归**：都可以减少共线性带来的影响，即 X 自变量之间有相互关联。区别可以归结为L2和L1正则化的性质差异。

3. **Polynomial Regression 多项式回归**：能够模拟非线性可分的数据（曲线），线性回归不能做到这一点。但容易过拟合。

4. **Logistic Regression 逻辑回归**：判断 True or False，Y 值为 0-1 表示概率，用于分类。线性回归使用「Residual 偏差」，而逻辑回归使用「maximum likelihood 最大似然」

### 2. Decision Tree 决策树
1. **ID3**: 计算「信息熵」 $Entropy(D)$，值越小，说明样本集合D的纯度就越高，进而选择用样本的某一个属性a来划分样本集合D时，就可以得出用属性a对样本D进行划分所带来的「信息增益」 $Gain(D, a)$，值越大，说明如果用属性a来划分样本集合D，那么纯度会提升。 $$Entropy(t)=-\sum_{k} p\left(c_{k} | t\right) \log p\left(c_{k} | t\right)$$  $$Classificationerror (t)=1-\max _{k}\left[p\left(c_{k} | t\right)\right]$$

2. **C4.5**: 提出Gainratio 「增益率」，解决ID3决策树的一个缺点，当一个属性的可取值数目较多时，那么可能在这个属性对应的可取值下的样本只有一个或者是很少个，那么这个时候它的信息增益是非常高的，这个时候纯度很高，ID3决策树会认为这个属性很适合划分，但是较多取值的属性来进行划分带来的问题是它的泛化能力比较弱。用 $I(·)$ 表示不纯度——可以是熵可以是基尼，信息增益：$$\Delta=I(\text { parent })-\sum_{i=1}^{n} \frac{N\left(a_{i}\right)}{N} I\left(a_{i}\right)$$信息增益率：$$Gainratio =\frac{\Delta}{{Entropy}({parent})}$$

3. **CART(Classification and Regression Tree)**: 通过计算 Gini 基尼系数（尽可能小），判断 impurity 不纯洁度。离散数据用「是否」划分子树，连续数据可以用「两两之间平均值」划分子树。$${Gini}(t)=1-\sum_{k}\left[p\left(c_{k} | t\right)\right]^{2}$$D 分裂为 DL 和 DR，分裂后的信息增益$$Gain(D, A)=\frac{\left|D_{L}\right|}{|D|} \operatorname{Gini}\left(D_{L}\right)+\frac{\left|D_{R}\right|}{|D|} \operatorname{Gini}\left(D_{R}\right)$$

### 3. SVM 支持向量机
SVM：https://blog.csdn.net/liugan528/article/details/79448379

KKT：https://blog.csdn.net/qq_32763149/article/details/81055062

**SVM 分类**：

1. 硬间隔支持向量机（线性可分支持向量机）：当训练数据线性可分时，可通过硬间隔最大化学得一个线性可分支持向量机。
2. 软间隔支持向量机：当训练数据近似线性可分时，可通过软间隔最大化得到一个线性支持向量机。
3. 非线性支持向量机：当训练数据线性不可分时，可通过核方法以及软间隔最大化得一个非线性支持向量机。

**基本原理**：

1. Maximum Margin Classifier：只看边界。

2. Soft Margin Classifier（即 Support Vector Classifier）：允许 misclassification误分类，寻找两个支撑向量来确定分类边界。

3. Kernel Function：非线性SVM，从低维数据开始，通过「核函数」给数据升维，然后找到一个 Support Vector Classifier 将数据分成两组。核函数的选择，支撑向量的选择，都用 cross validation 交叉验证。

4. Kernel Trick: 根据升维的距离进行计算，但是不进行实际的升维。

**具体过程**：

1. 线性可分的情况：对于超平面 $w \cdot x+b=0$ 和 $margin$ 有关系$$ {margin}=\frac{2}{\|w\|}$$
    最大化 $margin$ 等效于最小化 $\frac{1}{2}|w|^{2}$
    
    形成一个拉格朗日乘子α的约束问题 $$\begin{array}{ll}{\min _{w, b}} & {\frac{1}{2}|w|^{2}} {\text {s.t.}} & {y_{i}\left(w \cdot x_{i}+b\right)-1 \geq 0}\end{array}$$
    可以列式 $$L(w, b, \alpha)=\frac{1}{2}|w|^{2}-\sum_{i=1}^{N} \alpha_{i}\left[y_{i}\left(w \cdot x_{i}+b\right)-1\right]$$
    拉格朗日对偶性：解决「凸二次规划」（convex quadratic propgramming）问题，即将原始的约束最优化问题可等价于极大极小的对偶问题（以 w,b 作参数时的最小值，以α作参数时的最大值）
    $$\max _{\alpha} \min _{w, b} \quad L(w, b, \alpha)$$通过求导一系列步骤，转换成\begin{array}{ll}
{\min _{\alpha}} & {\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i}} \\
{\text { s.t. }} & {\sum_{i=1}^{N} \alpha_{i} y_{i}=0} \\
{} & {\alpha_{i} \geq 0, \quad i=1,2, \cdots, N}
\end{array}

2. 线性不可分的情况：对每个样本引入一个松弛变量 $\xi_{i} \geq 0$, 约束条件和目标函数变为

 $$\begin{aligned}
&y_{i}\left(w \cdot x_{i}+b\right) \geq 1-\xi_{i}\\
&\min _{w, b, \xi} \frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{N} \xi_{i}
\end{aligned}$$
    

**部分术语**：

1. KKT 条件：是拉格朗日乘子的泛化，把所有的不等式约束、等式约束和目标函数全部写为一个式子L(a, b, x)= f(x) + a*g(x) + b*h(x)，KKT条件是说最优值必须满足以下条件：（1）L(a, b, x)对x求导为零；（2）h(x) =0; （3）a*g(x) = 0;

2. SMO：Sequential Minimal Optimization用二次规划来求解α，要用到 KKT

3. SVR：支持向量回归

**优点**：
SVM在中小量样本规模的时候容易得到数据和特征之间的非线性关系，可以避免使用神经网络结构选择和局部极小值问题，可解释性强，可以解决高维问题。

**缺点**：
SVM对缺失数据敏感，对非线性问题没有通用的解决方案，核函数的正确选择不容易，计算复杂度高，主流的算法可以达到O(n2)O(n2)的复杂度，这对大规模的数据是吃不消的。

### 4. Naive Bayes Algorithms 朴素贝叶斯
   1. Naive Bayes
   2. Gaussian Naive Bayes
   3. Multinomial Naive Bayes
   4. Bayesian Belief Network (BBN)
   5. Bayesian Network (BN)

朴素贝叶斯基本公式：$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
![](https://i.imgur.com/O9FsXbG.png)

### 5. KNN(k-NearestNeighbor) K 最邻近算法
用于分类
1. 计算测试数据与各个训练数据之间的距离；
2. 按照距离的递增关系进行排序；
3. 选取距离最小的K个点；
4. 确定前K个点所在类别的出现频率；
5. 返回前K个点中出现频率最高的类别作为测试数据的预测分类

![](https://i.imgur.com/85vZVhC.png)

### 6. Clustering Algorithm 聚类算法
1.  k-Means：选取平均值
2.  k-Medians：由选取平均值改为选取中位数
3.  Expectation Maximisation (EM)：有隐含随机变量的概率模型的参数的估计方法，它是一种无监督的算法
4.  Hierarchical Clustering 层次聚类：
```
(1) 将每个对象看作一类，计算两两之间的最小距离；

(2) 将距离最小的两个类合并成一个新类；

(3) 重新计算新类与所有类之间的距离；

(4) 重复(2)、(3)，直到所有类最后合并成一类。
```

### 7. K-means 算法
```
选取k个初始质心(作为初始cluster);
repeat:
    对每个样本点，计算得到距其最近的质心，将其类别标为该质心所对应的cluster;
    重新计算k个cluser对应的质心;
until 质心不再发生变化
```
![](https://i.imgur.com/a6Sp3Ee.png)

### 8. Random Forest 随机森林

### 9. Dimensionality Reduction Algorithms 降维算法

### 10. Gradient Boosting algorithms 梯度提升算法
1.  GBM
2.  XGBoost
3.  LightGBM
4.  CatBoost

### 11. Deep Learning Algorithms 深度学习
1.  Convolutional Neural Network (CNN)
2.  Recurrent Neural Networks (RNNs)
3.  Long Short-Term Memory Networks (LSTMs)
4.  Stacked Auto-Encoders
5.  Deep Boltzmann Machine (DBM)
6.  Deep Belief Networks (DBN)

------------
## 机器学习损失函数
1. 0-1损失函数
$$
L(y,f(x)) =
\begin{cases}
0, & \text{y = f(x)}  \\
1, & \text{y $\neq$ f(x)}
\end{cases}
$$
2. 绝对值损失函数
$$
L(y,f(x))=|y-f(x)|
$$
3. 平方损失函数
$$
L(y,f(x))=(y-f(x))^2
$$
4. log对数损失函数
$$
L(y,f(x))=log(1+e^{-yf(x)})
$$
5. 指数损失函数
$$
L(y,f(x))=exp(-yf(x))
$$
6. Hinge损失函数
$$
L(w,b)=max\{0,1-yf(x)\}
$$

-----------
## 机器学习优化方法

梯度下降是最常用的优化方法之一，它使用梯度的反方向 $ \nabla_\theta J(\theta) $ 更新参数 $ \theta $，使得目标函数$J(\theta)$达到最小化的一种优化方法，这种方法我们叫做梯度更新. 

1. (全量)梯度下降
$$
\theta=\theta-\eta\nabla_\theta J(\theta)
$$
2. 随机梯度下降
$$
\theta=\theta-\eta\nabla_\theta J(\theta;x^{(i)},y^{(i)})
$$
3. 小批量梯度下降
$$
\theta=\theta-\eta\nabla_\theta J(\theta;x^{(i:i+n)},y^{(i:i+n)})
$$
4. 引入动量的梯度下降
$$
\begin{cases}
v_t=\gamma v_{t-1}+\eta \nabla_\theta J(\theta)  \\
\theta=\theta-v_t
\end{cases}
$$
5. 自适应学习率的Adagrad算法
$$
\begin{cases}
g_t= \nabla_\theta J(\theta)  \\
\theta_{t+1}=\theta_{t,i}-\frac{\eta}{\sqrt{G_t+\varepsilon}} \cdot g_t
\end{cases}
$$
6. 牛顿法
$$
\theta_{t+1}=\theta_t-H^{-1}\nabla_\theta J(\theta_t)
$$

    其中:
    $t$: 迭代的轮数

    $\eta$: 学习率

    $G_t$: 前t次迭代的梯度和

    $\varepsilon:$很小的数,防止除0错误

    $H$: 损失函数相当于$\theta$的Hession矩阵在$\theta_t$处的估计

-------
## 机器学习的评价指标
1. MSE(Mean Squared Error)
$$
MSE(y,f(x))=\frac{1}{N}\sum_{i=1}^{N}(y-f(x))^2
$$
2. MAE(Mean Absolute Error)
$$
MSE(y,f(x))=\frac{1}{N}\sum_{i=1}^{N}|y-f(x)|
$$
3. RMSE(Root Mean Squard Error)
$$
RMSE(y,f(x))=\frac{1}{1+MSE(y,f(x))}
$$
4. Top-k准确率
$$
Top_k(y,pre_y)=\begin{cases}
1, {y \in pre_y}  \\
0, {y \notin pre_y}
\end{cases}
$$
5. 混淆矩阵

混淆矩阵|Predicted as Positive|Predicted as Negative
|:-:|:-:|:-:|
|Labeled as Positive|True Positive(TP)|False Negative(FN)|
|Labeled as Negative|False Positive(FP)|True Negative(TN)|

* 真正例(True Positive, TP):真实类别为正例, 预测类别为正例
* 假负例(False Negative, FN): 真实类别为正例, 预测类别为负例
* 假正例(False Positive, FP): 真实类别为负例, 预测类别为正例 
* 真负例(True Negative, TN): 真实类别为负例, 预测类别为负例

* 真正率(True Positive Rate, TPR): 被预测为正的正样本数 / 正样本实际数
$$
TPR=\frac{TP}{TP+FN}
$$
* 假负率(False Negative Rate, FNR): 被预测为负的正样本数/正样本实际数
$$
FNR=\frac{FN}{TP+FN}
$$

* 假正率(False Positive Rate, FPR): 被预测为正的负样本数/负样本实际数，
$$
FPR=\frac{FP}{FP+TN}
$$
* 真负率(True Negative Rate, TNR): 被预测为负的负样本数/负样本实际数，
$$
TNR=\frac{TN}{FP+TN}
$$
* 准确率(Accuracy)
$$
ACC=\frac{TP+TN}{TP+FN+FP+TN}
$$
* 精准率
$$
P=\frac{TP}{TP+FP}
$$
* 召回率
$$
R=\frac{TP}{TP+FN}
$$
* F1-Score
$$
\frac{2}{F_1}=\frac{1}{P}+\frac{1}{R}
$$
* **ROC**

ROC曲线的横轴为“假正例率”，纵轴为“真正例率”. 以FPR为横坐标，TPR为纵坐标，那么ROC曲线就是改变各种阈值后得到的所有坐标点 (FPR,TPR) 的连线，画出来如下。红线是随机乱猜情况下的ROC，曲线越靠左上角，分类器越佳. 


* **AUC(Area Under Curve)**

AUC就是ROC曲线下的面积. 真实情况下，由于数据是一个一个的，阈值被离散化，呈现的曲线便是锯齿状的，当然数据越多，阈值分的越细，”曲线”越光滑. 

<img src="https://gss3.bdstatic.com/-Po3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike80%2C5%2C5%2C80%2C26/sign=b9cb389a68d0f703f2bf9d8e69933a58/f11f3a292df5e0feaafde78c566034a85fdf7251.jpg">

用AUC判断分类器（预测模型）优劣的标准:

- AUC = 1 是完美分类器，采用这个预测模型时，存在至少一个阈值能得出完美预测。绝大多数预测的场合，不存在完美分类器.
- 0.5 < AUC < 1，优于随机猜测。这个分类器（模型）妥善设定阈值的话，能有预测价值.
- AUC < 0.5，比随机猜测还差；但只要总是反预测而行，就优于随机猜测.

## 机器学习模型选择

1. 交叉验证

所有数据分为三部分：训练集、交叉验证集和测试集。交叉验证集不仅在选择模型时有用，在超参数选择、正则项参数 [公式] 和评价模型中也很有用。

2. k-折叠交叉验证

- 假设训练集为S ，将训练集等分为k份:$\{S_1, S_2, ..., S_k\}$. 
- 然后每次从集合中拿出k-1份进行训练
- 利用集合中剩下的那一份来进行测试并计算损失值
- 最后得到k次测试得到的损失值，并选择平均损失值最小的模型

3. Bias与Variance，欠拟合与过拟合

**欠拟合**一般表示模型对数据的表现能力不足，通常是模型的复杂度不够，并且Bias高，训练集的损失值高，测试集的损失值也高.

**过拟合**一般表示模型对数据的表现能力过好，通常是模型的复杂度过高，并且Variance高，训练集的损失值低，测试集的损失值高.

<img src="https://pic3.zhimg.com/80/v2-e20cd1183ec930a3edc94b30274be29e_hd.jpg">

<img src="https://pic1.zhimg.com/80/v2-22287dec5b6205a5cd45cf6c24773aac_hd.jpg">

4. 解决方法

- 增加训练样本: 解决高Variance情况
- 减少特征维数: 解决高Variance情况
- 增加特征维数: 解决高Bias情况
- 增加模型复杂度: 解决高Bias情况
- 减小模型复杂度: 解决高Variance情况

## 机器学习参数调优

1. 网格搜索

一种调参手段；穷举搜索：在所有候选的参数选择中，通过循环遍历，尝试每一种可能性，表现最好的参数就是最终的结果

2. 随机搜索

与网格搜索相比，随机搜索并未尝试所有参数值，而是从指定的分布中采样固定数量的参数设置。它的理论依据是，如果随即样本点集足够大，那么也可以找到全局的最大或最小值，或它们的近似值。通过对搜索范围的随机取样，随机搜索一般会比网格搜索要快一些。

3. 贝叶斯优化算法

贝叶斯优化用于机器学习调参由J. Snoek(2012)提出，主要思想是，给定优化的目标函数(广义的函数，只需指定输入和输出即可，无需知道内部结构以及数学性质)，通过不断地添加样本点来更新目标函数的后验分布(高斯过程,直到后验分布基本贴合于真实分布。简单的说，就是考虑了上一次参数的信息，从而更好的调整当前的参数。
