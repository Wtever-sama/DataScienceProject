## SVM

### extract features

- 数据预处理
	- 使用 stop_symbol.py 和 stopwords.txt 进行停用词处理
	- 使用 `sci-kit learn` 中的 `TfidfVectorizer` 将所有清洗后的文本处理成向量
	- 将所有处理后的文本向量作为 `X_full`, 对应的 label （1: baseball，-1: hockey）作为`y` ，使用 `train_test_split` 打乱顺序划分训练集和测试集, 得到的X的特征数量是 16040, 训练集样本量总共1592，测试集399
```bash
# print(X_train.shape, X_test.shape, len(y_train), len(y_test))
(1592, 16040) (399, 16040) 1592 399
```

### main functions

- 由于对于核函数 `rbf` ，`sklearn.svm.SVC` 中默认使用 SMO 优化器，所有在代码中没有显示定义 `alpha1`, `alpha2`和更新，但是 SMO 的更新原理如下：
	- 目标是找到一个超平面  $w\cdot x+b=0$ 使得不同的样本和超平面之间 间隔最大，并且样本被正确分类
	- 约束条件是 正样本间隔边界 $w'x+b=1$, 负样本 $w'x+b=-1$, 也就是 $y_i(w'\cdot x_i+b)\geq 1$
	- 使用松弛变量 $\xi_i$, 有 $y_i(wx+b)\geq 1-\xi_i$
- 构造拉格朗日函数：$$L=\frac{1}{2}\|w\|^2-\alpha(\sum_{i=1}^{n}y_i(w'x_i+b)-1+\xi_i)+C\sum_{i=1}^{n}\xi_i-\sum \xi_i\cdot \mu_i$$
- 使用 $w \quad b\quad \xi_i$ 对于 L 分别求导，令其为 0，得出的等式代入 L 得出目标函数 $$W(α)=\sum_{i=1}^{n}\alpha_i​−\frac{1}{2}​\sum_{i=1}^{n}​\sum_{j=1}^{n}​\alpha_i​\alpha_j​y_i​y_j​(xi​⋅xj​)$$
- 固定其他 $\alpha$, 对于其中不满足
	1. $\alpha_i\geq0$
	2. $\alpha_i =0$ 或者 $y_i(wx_i+b)=1$ 
	3. $\sum_{i=1}^{n}\alpha_iy_i=0$
同时满足上述条件才行，对于任意两不满足的参数，有 $$\alpha_1 y_1+\alpha_2y_2=\zeta$$，代入目标函数，求导之后得到 $$\alpha_2'=\alpha_2+\frac{y_2({error}_1-{error}_2)}{K_{11}+K_{22}-2K_{12}}$$
- 由于$\alpha_2$必须在$[0, C]$范围内，需对$\alpha_2'=\alpha_2^{\text{new, unclipped}}$进行裁剪，得到$\alpha_2^{\text{new}}$：

	- 若$y_1 \neq y_2$：裁剪区间为$[L, H] = [\max(0, \alpha_2^{\text{old}} - \alpha_1^{\text{old}}), \min(C, C + \alpha_2^{\text{old}} - \alpha_1^{\text{old}})]$；

	- 若$y_1 = y_2$：裁剪区间为$[L, H] = [\max(0, \alpha_2^{\text{old}} + \alpha_1^{\text{old}} - C), \min(C, \alpha_2^{\text{old}} + \alpha_1^{\text{old}})]$；

	- 裁剪规则：

$\alpha_2^{\text{new}} = \begin{cases} H & \text{若}\ \alpha_2^{\text{new, unclipped}} > H \\ \alpha_2^{\text{new, unclipped}} & \text{若}\ L \leq \alpha_2^{\text{new, unclipped}} \leq H \\ L & \text{若}\ \alpha_2^{\text{new, unclipped}} < L \end{cases}$

- **更新$\alpha_1$**：根据步骤 2 中的$\alpha_1$与$\alpha_2$的关系，代入$\alpha_2^{\text{new}}$计算：

$\alpha_1^{\text{new}} = \alpha_1^{\text{old}} + y_1 y_2 (\alpha_2^{\text{old}} - \alpha_2^{\text{new}})$

- **更新偏置b**：b的更新需确保超平面满足支持向量的条件（$y_i f(x_i) = 1$），通常取两个样本更新后的b的平均值（或根据边界条件计算），避免b波动过大：

$b_1^{\text{new}} = y_1 - \sum_{j=3}^n \alpha_j y_j K(x_j, x_1) - \alpha_1^{\text{new}} y_1 K_{11} - \alpha_2^{\text{new}} y_2 K_{21}$

$b_2^{\text{new}} = y_2 - \sum_{j=3}^n \alpha_j y_j K(x_j, x_2) - \alpha_1^{\text{new}} y_1 K_{12} - \alpha_2^{\text{new}} y_2 K_{22}$

$b^{\text{new}} = \frac{b_1^{\text{new}} + b_2^{\text{new}}}{2}$


### how to implement script

cd 到文件夹目录
```
cd Path/to/you/dir/TextClassifySVM
```

目录架构
```
TextClassifySVM/
	|_data_process.py # 文本处理脚本，处理成模型能够直接处理的 X_train, X_test, y_train, y_test
	|_models.py # SVC 模型构建，可以手动调参
	|_train.ipynb # 集成脚本，从数据处理到模型加载，到训练和评估
	|_metrics.py # 评估脚本，包含预测准确度、查准率、查全率、召回率
	|_stop_symbol.py # 需要去除的标点符号
	|_stopwords.txt # 需要去除的英文停用词
	|_Dataset_classification/ # 数据集
		|_baseball # baseball 的文本
		|_hockey # hockey 的文本
```

直接运行 `train.ipynb` 的单元格，逐个运行，可以到看结果

使用核函数为 `rbf`, 参数 `C=1.`, `gamma`设置为`auto`.

加载数据 (首次加载数据大约 1 分钟)

```bash
Loading baseball data: 100%|██████████| 995/995 [00:00<00:00, 1457.15it/s] 
Loading baseball data: 100%|██████████| 995/995 [00:00<00:00, 1457.15it/s] 
Loading hockey data: 100%|██████████| 996/996 [00:00<00:00, 1077.39it/s]
```

`cv_scores`

```bash
array([0.4984326 , 0.4984326 , 0.80188679, 0.82389937, 0.76415094])
```

`average_score`: 0.67736
### result

`metrics_result`

```json
{
	'accuracy': 0.8170426065162907, 
	'precision': 0.8661451422674332, 
	'recall': 0.8170426065162907, 
	'f1_score': 0.8107989836684859
}
```

