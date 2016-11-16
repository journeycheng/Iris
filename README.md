# Iris_species
## 目录
- 1 导入相关库
- 2 数据初认识
   - 导入数据
   - 数据整体信息
   - 查看Species的种类及数量
- 3 数据可视化
- 4 数据清洗
- 5 机器学习

## 1 导入相关库

```python
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
sns.set(style="white", color_codes=True)
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn import svm, neighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.liner_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
```

## 2 数据初认识

- 导入数据
```python
iris_df = pd.read_csv("Iris.csv")
```

- 数据整体信息
```python
> iris_df.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 6 columns):
Id               150 non-null int64
SepalLengthCm    150 non-null float64
SepalWidthCm     150 non-null float64
PetalLengthCm    150 non-null float64
PetalWidthCm     150 non-null float64
Species          150 non-null object
dtypes: float64(4), int64(1), object(1)
memory usage: 7.1+ KB
```
   - 没有缺失值
   - 共6列数据，其中SepalLengthCm、SepalWidthCm、PetalLengthCm和PetalWidthCm是属性值，Species是类别

- 显示前5行数据
``` python
> print iris_df.head()

   Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
0   1            5.1           3.5            1.4           0.2  Iris-setosa
1   2            4.9           3.0            1.4           0.2  Iris-setosa
2   3            4.7           3.2            1.3           0.2  Iris-setosa
3   4            4.6           3.1            1.5           0.2  Iris-setosa
4   5            5.0           3.6            1.4           0.2  Iris-setosa
```
- 查看Species的种类及数量
```python
> print iris_df["Species"].value_counts()

Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
Name: Species, dtype: int64
```
## 3 数据可视化
- 所有样本的萼片长度和萼片宽度的散点图
```python
iris_df.plot(kind="scatter",x="SepalLengthCm",y="SepalWidthCm")
plt.show()
```
![](raw/figure_1.png?raw=true)

- 在散点图的基础上添加单变量的直方图
```python
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris_df, size=5)
plt.show()
```
![](raw/figure_2.png?raw=true)

- 用不同颜色表明不同的品种
```python
sns.FacetGrid(iris_df, hue="Species", size=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()
plt.show()
```
![](raw/figure_3.png?raw=true)

- boxplot
```python
sns.boxplot(x="Species", y="PetalLengthCm", data=iris_df)
plt.show()
```
![](raw/figure_4.png?raw=true)

- boxplot加上散点图
```python
sns.boxplot(x="Species", y="PetalLengthCm", data=iris_df)
sns.stripplot(x="Species", y="PetalLengthCm", data=iris_df, jitter=True, edgecolor="gray")
```
![](raw/figure_5.png?raw=true)

- violinplot
```python
sns.violinplot(x="Species", y="PetalLengthCm", data=iris_df, size=6)
plt.show()
```
![](raw/figure_6.png?raw=true)

- KDE图（Kernel Density Estimation）
```python
sns.FacetGrid(iris_df, hue="Species", size=6).map(sns.kdeplot, "PetalLengthCm").add_legend()
plt.show()
```
![](raw/figure_7.png?raw=true)

- 两两变量决定的分布关系，直方图形式
```python
# delete the column Id , axis=1 for columns and 0 for rows
sns.pairplot(iris_df.drop("Id", axis=1), hue="Species", size=3)
plt.show()
```
![](raw/figure_8.png?raw=true)

- 两两变量决定的分布，KDE形式
```python
sns.pairplot(iris_df.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde")
plt.show()
```
![](raw/figure_9.png?raw=true)

- 不同品种鸢尾花的所有属性分布的箱线图
```python
iris_df.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))
plt.show()
```
![](raw/figure_10.png?raw=true)

- 调和曲线图

也称作Andrews Curves。
把属性值当做傅立叶级数的参数，就可以将高维空间中的样本点对应到二维平面上的一条曲线。
比如，样本数据为x=(x1,x2,x3...,),对应的曲线是：

$$
f_x(t) = \frac{x_1}{\sqrt{2}}+x_2\sin t + s_3 \cos t + x_4\sin 2t + ... 
$$
```python
from pandas.tools.plotting import andrews_curves
andrews_curves(iris_df.drop("Id", axis=1), "Species")
plt.show()
```
![](raw/figure_11.png?raw=true)

- 把每个样本的不同属性点连成直线
```python
from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(iris_df.drop("Id", axis=1), "Species")
plt.show()
```
![](raw/figure_12.png?raw=true)

- 忘记了叫什么图
```python
from pandas.tools.plotting import radviz
radviz(iris_df.drop("Id", axis=1), "Species")
plt.show()
```
![](raw/figure_13.png?raw=true)

## 4 数据清洗

- 删除Id列，对分析无影响
```python
iris_df = iris_df.drop('Id', axis = 1)
```
- 把Species列的值修改为对应的数字
```python
iris_df.loc[iris_df['Species'] == 'Iris-setosa', 'Species'] = 0
iris_df.loc[iris_df['Species'] == 'Iris-versicolor', 'Species'] = 0
iris_df.loc[iris_df['Species'] == 'Iris-virginica', 'Species'] = 0

# 将object类型转换为int类型
iris_df['Species'] = iris_df['Species'].astype('int')
```
- 将数据集分成训练集和测试集，比例为7:3
```python
iris_train, iris_test = train_test_split(iris_df, test_size=.3, random_state = 1)

iris_train_y = iris_train['Species']
iris_train_x = iris_train.drop('Species', axis = 1)
iris_test_y = iris_test['Species']
iris_test_x = iris_test.drop('Species', axis = 1)

# 可以查看train和test
print iris_train_x.shape, iris_test_x.shape
```

