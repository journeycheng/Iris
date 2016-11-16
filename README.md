# Iris_species
## 1.Data Visualization
```python
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
sns.set(style="white", color_codes=True)
import matplotlib.pyplot as plt
```
```python
iris = pd.read_csv("Iris.csv")
iris.head()
iris["Species"].value_counts()

# e.g. 1
iris.plot(kind="scatter",x="SepalLengthCm",y="SepalWidthCm")

# e.g. 2
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)

# e.g. 3
sns.FacetGrid(iris, hue="Species", size=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()

# e.g. 4
sns.boxplot(x="Species", y="PetalLengthCm", data=iris)

# e.g. 5
sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")

# e.g. 6
sns.violinplot(x="Species", y="PetalLengthCm", data=iris, size=6)

# e.g.7
sns.FacetGrid(iris, hue="Species", size=6).map(sns.kdeplot, "PetalLengthCm").add_legend()

# e.g.8
# delete the column Id , axis=1 for columns and 0 for rows
sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)

# e.g. 9
sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde")

# e.g.10
iris.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))

# e.g.11
from pandas.tools.plotting import andrews_curves
andrews_curves(iris.drop("Id", axis=1), "Species")

# e.g. 12
from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(iris.drop("Id", axis=1), "Species")

# e.g. 13
from pandas.tools.plotting import radviz
radviz(iris.drop("Id", axis=1), "Species")

plt.show()
```
