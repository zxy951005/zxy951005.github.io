# PU Learning

[TOC]

## PU Learning的基本概念

在传统二分类问题中，训练集中所有样本都有明确的0-1标签，即正样本（y=1）和负样本（y=0）。

这些样本满足独立同分布:

x ~ f(x) 即 x ~ $af_+(x) + (1-a)f_-(x)$

$ a = Pr(y=1)$是正类别先验， a通常根据经验给出，例如在信贷欺诈中大概为0.0001。

PU Learning的目标与传统二分类问题一致，都是为了训练一个分类器用以区分正负样本。不同点在于：此时只有少数已标注的正样本，并没有负样本。

## PU Learning的基本假设

PU Learning的问题设定：一个是P集合，另一个是U集合。

有标签和正标签是等价的。一个样例没有标签：1. 是负样本；2. 是正样本。

对PU数据中的学习模型，从以下两个维度设定假设：打标机制、类别分布

### 打标机制

构建选择模型的一些假设：

- 完全随机选择

  > 有标签样本完全是从正样本分布中随机选择的，且与本身的特征属性无关

  $e(x) = Pr(s=1|x,y=1)=Pr(s=1|y=1)=c$

  有标签样本相当于全部从正样本分布中随机获取，且独立同分布。

  也就是: $f_l(x) = f_+(x)$

  $Pr(s=1|y=1)=c$

  => $\frac{Pr(s=1|x)}{Pr(y=1|x)}=c$

  => $Pr(s=1|x)=cPr(y=1|x))$

  - 传统分类器：即 Pr(y=1|x)，以“**是否有正负** $y \in \{0,1\}$ ”作为目标变量。此时所有训练样本都必须有明确的0-1标签。这在PU问题中无法直接构建。

  - 非传统分类器：即 Pr(s=1|x) ，以“**是否被标注**$s \in \{0,1\}$ ”作为目标变量。实际上，这是把无标签样本U当作是负样本。这在PU问题中可以直接构建。

    Pr(y=1|x1) > Pr(y=1|x2)  <=> Pr(s=1|x1) > Pr(s=1|x2)

- 随机选择

  > 有标签样本是从正样本分布中随机选择的，但与本身的特征属性有关。

  $e(x) = Pr(s=1|x,y=1)$

- 概率差距

  > 正负预测概率差距越大，被选中打标的概率也就越大。

  概率差距 $\Delta Pr(x)=Pr(y=1|x)-Pr(y=0|x)$

  如果$\Delta Pr(x)$越高，其被选中的可能性也应该越小。因此，倾向评分函数是概率差距$\Delta Pr(x)$的非负、递减函数，可定义为：

  $e(x) = f(\Delta Pr(x))=f(Pr(y=1)|x)-Pr(y=0|x), \frac{df(t)}{dt}<0$

  由于无法构造传统分类器，因此利用非传统分类器来估计$\Delta Pr(x)$：

  $\Delta \hat{Pr}(x)=Pr(s=1|x) - Pr(s=0|x)$

### 数据假设

### PU Learning的评估指标

PU Learning的基础方法（框架），通常可分为三类：

第一类是两阶段技术（Two-step PU Learning）：先在未标记数据U中识别出一些可靠的负样本，然后在正样本P和这些可靠的负样本上进行监督学习。这是最常用的一种。

第二类是有偏学习（Biased PU Learning）：简单地把未标记样本U作为噪声很大的负样本来处理。

第三类是融入先验类别（Incorporation of the Class Prior）：尝试给正样本P和未标记样本U赋予权重，并给出样本属于正标签的条件概率估计。

## 两阶段技术

> 基于可分性和平滑性假设，所有正样本都与有标签样本相似，而与负样本不同。

为了绕过缺乏负标注的问题，two-stage策略首先挖掘出一些可靠的负例，再将问题转化为一个传统的监督和半监督学习问题。

整体流程一般可分解为以下3个步骤：

- step 1: 从U集合中识别出可靠负样本（Reliable Negative，RN）。
- step 2: 利用P集合和RN集合组成训练集，训练一个传统的二分类模型
- step 3: 根据某种策略，从迭代生成的多个模型中选择最优的模型。

基于平滑性假设，样本属性相似时，其标签也基本相同。换言之，可靠负样本就是那些与正样本相似度很低的样本。问题的关键就是定义相似度，或者说距离（distance）。

### 识别可靠负样本

> 1）The Spy Technique
> 2）1-DNF技术

#### The Spy Technique

spy样本需要有足够量，否则结果可信度低。

- **step 1**：从P中随机选择一些正样本S，放入U中作为间谍样本。此时样本集变为P-S和U+S。其中，从P中划分子集S的数量比例一般为15%。

- **step 2**：使用P-S作为正样本，U+S作为负样本，利用迭代的EM算法进行分类。初始化时，我们把所有无标签样本当作负类（y=0 ），训练一个分类器，对所有样本预测概率Pr(y=1)。

- **step 3**：以spy样本分布的最小值作为阈值，U中所有低于这个阈值的样本认为是RN。

  ![](C:\Users\ZXY\Pictures\spies.png)

  ```python
  class spy:
      def __init__(self, model1, model2):
          self.first_model = model1
          self.second_model = model2
      def fit(self, X, y, spie_rate=0.2, spie_tolerance=0.05):
      	# Step 1. Infuse spies
          spie_mask = np.random.random(y.sum()) < spie_rate
          # Unknown mix + spies
          MS = np.vstack([X[y == 0], X[y == 1][spie_mask]])
          MS_spies = np.hstack([np.zeros((y == 0).sum()), np.ones(spie_mask.sum())])
          # Positive with spies removed
          P = X[y == 1][~spie_mask].values
          # Combo
          MSP = np.vstack([MS, P])
          # Labels
          MSP_y = np.hstack([np.zeros(MS.shape[0]), np.ones(P.shape[0])])
          # Fit first model
          logger.debug('Training first model')
          self.first_model.fit(MSP, MSP_y)
          prob = self.first_model.predict_proba(MS)[:, 1]
          # Find optimal t
          t = 0.001
          while MS_spies[prob <= t].sum()/MS_spies.sum() <= spie_tolerance:
              t += 0.001
          logger.debug('Optimal t is {0:.06}'.format(t))
          logger.debug('Positive group size {1}, captured spies {0:.02%}'.format(
              MS_spies[prob > t].sum()/MS_spies.sum(), (prob > t).sum()))
          logger.debug('Likely negative group size {1}, captured spies {0:.02%}'.format(
              MS_spies[prob <= t].sum()/MS_spies.sum(), (prob <= t).sum()))
          # likely negative group
          N = MS[(MS_spies == 0) & (prob <= t)]
          P = X[y == 1]
          NP = np.vstack([N, P])
          L = np.hstack([np.zeros(N.shape[0]), np.ones(P.shape[0])])
          # Fit second model
          logger.debug('Training second model')
          self.second_model.fit(NP, L)
  ```

  下图为在Blobs数据上使用PU spy的结果：

  ![](C:\Users\ZXY\Pictures\blobs_spies.png)

  

#### 1-DNF技术

- **step 1**：获取PU数据中的所有特征，构成特征集合F。
- **step 2**：对于每个特征，如果其在P集合中的出现频次大于N集合，记该特征为正特征(Positive Feature，PF)，所有满足该条件的特征组成一个PF集合。
- **step 3**：对U中的每个样本，如果其不包含PF集合中的任意一个特征，则将该样本加入RN。

### 训练分类器

在识别出可靠负样本RN后，我们来训练一个分类器，操作步骤描述如下：

```python
# 样本准备：P 和 RN 组成训练集X_train; P给定标签1，RN给定标签0，组成训练集标签y_train
# 用 X_train 和 y_train 训练逻辑回归模型 model
model.fit(X_train, y_train) 

# 用 model 对 Q 进行预测（分类）得到结果 prob
Q = U - RN          # 无标签样本集U中剔除RN
prob = model.predict(Q) 

# 找出 Q 中被判定为负的数据组成集合 W
predict_label = np.where(prob < 0.5, 0, 1).T[0]
negative_index = np.where(predict_label == 0)
W = Q[negative_index]

# 将 W 和 RN 合并作为新的 RN，同时将 W 从 Q 中排除
RN = np.concatenate((RN, W))    # RN = RN + W
Q = np.delete(Q, negative_index, axis=0)  # Q = Q - W 

# 用新的 RN 和 P 组成新的 X_train，对应生成新的 y_train
# 继续训练模型，扩充 RN，直至 W 为空集，循环结束。
# 其中每次循环都会得到一个分类器 model ，加入模型集 model_list 中
```

### 选择最优模型

从每次循环生成的分类器中，制定选择策略，选出一个最佳分类器。

1. 预测误差提升差$\Delta E$

   当$\Delta E<0$时，说明当前轮i，比前一轮i-1模型误差开始升高。

   $\Delta E=Pr(\hat{y}_i ≠ y) - Pr(\hat{y}_{i-1}≠y)$

   以Blobs数据为例：首先用正例数据和未标注数据训练一个随机森林模型；然后把预测为正例数据的得分范围记录，将未标记样本中低于这个范围的记为负例，高于这个范围的几位正例；接着迭代进行训练，每次迭代更新正例得分范围和将负例中的样本高于得分的记为正例，并且计算新标记的正例和负例个数；直到不再新增或者达到迭代上限。

   ![](C:\Users\ZXY\Pictures\Two step.png)

2. F1值提升比

   

3. Vote（Bagging PU Learning）

   以Blobs数据为例：取正例中的一部分作为已知正例，其余作为未标记数据

   ![](C:\Users\ZXY\Pictures\blobs.png)

   每次从无标签样本选一部分，然后训练一个非传统分类器，然后对OOB数据打分；重复N次，最终用总共的打分来划分无标签样本中的数据。

   ![](C:\Users\ZXY\Pictures\PU BAGGING.png)

   

4. 假阴率(FNR>5%)

## 有偏学习

> 有偏PU Learning的思想是，把无标签样本当作带有噪声的负样本。

Unbiased PU Learning

用未标记数据来间接估计$\hat{R}_n^-(g)$。

直接把U当作N会是负类经验风险增大，因此需要平衡U的风险。

$\pi_np_n = p(x) - \pi_np_n$ => $\pi_nR_n^-(g)=R_u^-(g)-\pi_pR_p^-(g)$

=> $\hat{R}_{pu}(g)=\pi_p\hat{R}_p^+(g) - \pi_p\hat{R}_p^-(g)+ \hat{R}_u^-(g)$

在进行模型训练的时候，通常将0-1损失替换为代理损失函数。

若Loss满足$l(t, +1) - l(t, -1)=-t$

则PU的风险公式简化为：$R(g)=\pi_pE_p[-g(X)] + E_u[l(-g(X)]$

这是一个凸问题有最优解

## 融入类别先验

Non-Negative PU Learning

由于Unbiased PU Learning第二项风险可能为负，存在严重过拟合问题，对此：

$\bar{R}_{pu}(g)=\pi_p\hat{R}_p^+(g) + max\{0, \hat{R}_u^-(g) - \pi_p\hat{R}_p^-(g)\} $

然而，我们并不知道正样本的比例，这时可以对原分布部分拟合
$q(x;\theta)=\theta p(x|y=1)$
然后求

$\theta :=\arg \,\min_{0\le\theta \le 1}\int f(\frac{q(x;\theta)}{p(x)})p(x)dx$

![](C:\Users\ZXY\Pictures\拟合.png)

由于缺失了负样本下x的分布信息，部分拟合将过高估计正例样本的比例。该文通过利用惩罚散度来避免负样本缺失造成的误差，对于各个散度的惩罚形式如下：

![](C:\Users\ZXY\Pictures\惩罚.png)

```python
def nnpu_loss(self, y_true, y_pred):
    '''
    y_true: shape[batch_size, 2], 经过labelEncoder处理后的0，1矩阵，0类位于第一列，1类位于第二列,那么这里0类就是已知的标签数据，1类为unlabel数据。
    y_pred: shape[batch_size ,2], 模型预测的分类概率,这里我们以模型预测的第一类概率为准。
     '''
    print("============================== use nnpu_learning !!!! ===========================================")
    p = y_true[:, 0] * y_pred[:, 0]
    u = y_true[:, 1] * y_pred[:, 0]
    p_num = tf.where(K.sum(y_true[:, 0], axis=-1) == 0, 1.0, K.sum(y_true[:, 0], axis=-1))
    u_num = tf.where(K.sum(y_true[:, 1], axis=-1) == 0, 1.0, K.sum(y_true[:, 1], axis=-1))
    t = self.pi * K.sum(p*1, axis=-1)/p_num
    t2 = K.sum(-u, axis=-1)/u_num - self.pi *K.sum(-p, axis=-1)/p_num
    t3 = tf.where(t2 > 0.0, t2, 0.0)
    loss = t + t2
    return -loss

```



下图为CIFAR-10和MNIST上的实验结果, 看的出来PU Learning的效果更好，但是使用该方法的前提是，unlabeled样本中positive样本特征分布与已知的positive样本分布一致！

![](C:\Users\ZXY\Pictures\mnist.png)

![](C:\Users\ZXY\Pictures\mnist.png)

puloss nnpuloss的代码如下：

```python
def puloss(y_true, y_pred):
    return positive_risk(y_true, y_pred) + negative_risk(y_true, y_pred)


def nnpuloss(y_true, y_pred):
    return (positive_risk(y_true, y_pred)
            + tf.nn.relu(negative_risk(y_true, y_pred)))


def pretrain_loss(y_true, y_pred):
    return tf.maximum(positive_risk(y_true, y_pred),
                      negative_risk(y_true, y_pred))


def error(y_true, y_pred):
    global pi_p

    n_positive = tf.maximum(1., tf.reduce_sum(y_true))
    n_unlabeled = tf.maximum(1., tf.reduce_sum(1 - y_true))
    y_positive = (1 - tf.sign(y_pred)) / 2
    y_unlabeled = (1 + tf.sign(y_pred)) / 2
    positive_risk = tf.reduce_sum(pi_p * y_true / n_positive * y_positive)
    negative_risk = tf.reduce_sum(
        ((1 - y_true) / n_unlabeled - pi_p * y_true / n_positive) * y_unlabeled)
    return positive_risk + negative_risk


def positive_risk(y_true, y_pred):
    global pi_p

    loss_func = tf.nn.sigmoid

    n_positive = tf.maximum(1., tf.reduce_sum(y_true))
    r_plus = loss_func(-y_pred)
    return tf.reduce_sum(pi_p * y_true / n_positive * r_plus)


def negative_risk(y_true, y_pred):
    global pi_p

    loss_func = tf.nn.sigmoid

    n_positive = tf.maximum(1., tf.reduce_sum(y_true))
    n_unlabeled = tf.maximum(1., tf.reduce_sum(1 - y_true))
    r_minus = loss_func(y_pred)
    return tf.reduce_sum(
        ((1 - y_true) / n_unlabeled - pi_p * y_true / n_positive) * r_minus)
```

## 参考文献

1. 《Convex Formulation for Learning from Positive and Unlabeled Data》 ICML (2015)
2. 《Positive-Unlabeled Learning with Non-Negative Risk Estimator》NIPS 11 (2017)
3. 《A bagging SVM to learn from positive and unlabeled examples》
4. 《Learning From Positive and Unlabeled Data: A Survey》