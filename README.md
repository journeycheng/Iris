# Iris_species
## 1.数据可视化
### 1.1 导入相关库
```python
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
sns.set(style="white", color_codes=True)
import matplotlib.pyplot as plt
```

### 1.2 导入数据
```python
iris_df = pd.read_csv("Iris.csv")
```

### 1.3 数据初认识
- 1.3.1 数据整体信息
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
- 1.3.2 显示前5行数据
``` python
> print iris_df.head()

   Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
0   1            5.1           3.5            1.4           0.2  Iris-setosa
1   2            4.9           3.0            1.4           0.2  Iris-setosa
2   3            4.7           3.2            1.3           0.2  Iris-setosa
3   4            4.6           3.1            1.5           0.2  Iris-setosa
4   5            5.0           3.6            1.4           0.2  Iris-setosa
```
- 1.3.3 查看Species的种类及数量
```python
> print iris_df["Species"].value_counts()

Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
Name: Species, dtype: int64
```
### 1.4 数据图
```python
iris.plot(kind="scatter",x="SepalLengthCm",y="SepalWidthCm")
```
![](raw/figure_1.png?raw=true)

```python
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)
```
![](raw/figure_2.png?raw=true)

```python
sns.FacetGrid(iris, hue="Species", size=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()
```
![](raw/figure_3.png?raw=true)

```python
sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
```
![](raw/figure_4.png?raw=true)

```python
sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")
```
![](raw/figure_5.png?raw=true)

```python
sns.violinplot(x="Species", y="PetalLengthCm", data=iris, size=6)
```
![](raw/figure_6.png?raw=true)

```python
sns.FacetGrid(iris, hue="Species", size=6).map(sns.kdeplot, "PetalLengthCm").add_legend()
```
![](raw/figure_7.png?raw=true)

```python
# delete the column Id , axis=1 for columns and 0 for rows
sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)
```
![](raw/figure_8.png?raw=true)

```python
sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde")
```
![](raw/figure_9.png?raw=true)

```python
iris.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))
```
![](raw/figure_10.png?raw=true)

```python
from pandas.tools.plotting import andrews_curves
andrews_curves(iris.drop("Id", axis=1), "Species")
```
![](raw/figure_11.png?raw=true)

```python
from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(iris.drop("Id", axis=1), "Species")
```
![](raw/figure_12.png?raw=true)

```python
from pandas.tools.plotting import radviz
radviz(iris.drop("Id", axis=1), "Species")
```
![](raw/figure_13.png?raw=true)

```python
plt.show()
```
