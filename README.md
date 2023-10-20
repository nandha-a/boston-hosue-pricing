### boston-hosue-pricing

### Tools Requirements
1. [GitHub Account](https://fithub.com)
2. [VS code IDE](https://code.visualstudio.com/)
3. [Heroku Account](https://heroku.com)
4. [GitCli](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)

### Packages Used
1. Numpy
2. Pandas
3. Matplotlib.pyplot
4. Seaborn
5. sklearn
6. Pickle

### Algoritms and preprocessor
1. Linear Regression
2. Random Forest Regressor
3. Ada Boost Regressor
4. Standard Scaler

### Data Description
Boston House Price data set consists of 506 entires and 14 features, this dataset has been removed from scikit-learn since version 1.2. you can fetch the dataset from the original source:

    import pandas as pd
    import numpy as np

    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

Features of this dataset is as follows crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat, medv.
There is no null values present in this dataset. The five point summary of this data as follows
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>crim</th>
      <td>506.0</td>
      <td>3.613524</td>
      <td>8.601545</td>
      <td>0.00632</td>
      <td>0.082045</td>
      <td>0.25651</td>
      <td>3.677083</td>
      <td>88.9762</td>
    </tr>
    <tr>
      <th>zn</th>
      <td>506.0</td>
      <td>11.363636</td>
      <td>23.322453</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>12.500000</td>
      <td>100.0000</td>
    </tr>
    <tr>
      <th>indus</th>
      <td>506.0</td>
      <td>11.136779</td>
      <td>6.860353</td>
      <td>0.46000</td>
      <td>5.190000</td>
      <td>9.69000</td>
      <td>18.100000</td>
      <td>27.7400</td>
    </tr>
    <tr>
      <th>chas</th>
      <td>506.0</td>
      <td>0.069170</td>
      <td>0.253994</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>nox</th>
      <td>506.0</td>
      <td>0.554695</td>
      <td>0.115878</td>
      <td>0.38500</td>
      <td>0.449000</td>
      <td>0.53800</td>
      <td>0.624000</td>
      <td>0.8710</td>
    </tr>
    <tr>
      <th>rm</th>
      <td>506.0</td>
      <td>6.284634</td>
      <td>0.702617</td>
      <td>3.56100</td>
      <td>5.885500</td>
      <td>6.20850</td>
      <td>6.623500</td>
      <td>8.7800</td>
    </tr>
    <tr>
      <th>age</th>
      <td>506.0</td>
      <td>68.574901</td>
      <td>28.148861</td>
      <td>2.90000</td>
      <td>45.025000</td>
      <td>77.50000</td>
      <td>94.075000</td>
      <td>100.0000</td>
    </tr>
    <tr>
      <th>dis</th>
      <td>506.0</td>
      <td>3.795043</td>
      <td>2.105710</td>
      <td>1.12960</td>
      <td>2.100175</td>
      <td>3.20745</td>
      <td>5.188425</td>
      <td>12.1265</td>
    </tr>
    <tr>
      <th>rad</th>
      <td>506.0</td>
      <td>9.549407</td>
      <td>8.707259</td>
      <td>1.00000</td>
      <td>4.000000</td>
      <td>5.00000</td>
      <td>24.000000</td>
      <td>24.0000</td>
    </tr>
    <tr>
      <th>tax</th>
      <td>506.0</td>
      <td>408.237154</td>
      <td>168.537116</td>
      <td>187.00000</td>
      <td>279.000000</td>
      <td>330.00000</td>
      <td>666.000000</td>
      <td>711.0000</td>
    </tr>
    <tr>
      <th>ptratio</th>
      <td>506.0</td>
      <td>18.455534</td>
      <td>2.164946</td>
      <td>12.60000</td>
      <td>17.400000</td>
      <td>19.05000</td>
      <td>20.200000</td>
      <td>22.0000</td>
    </tr>
    <tr>
      <th>black</th>
      <td>506.0</td>
      <td>356.674032</td>
      <td>91.294864</td>
      <td>0.32000</td>
      <td>375.377500</td>
      <td>391.44000</td>
      <td>396.225000</td>
      <td>396.9000</td>
    </tr>
    <tr>
      <th>lstat</th>
      <td>506.0</td>
      <td>12.653063</td>
      <td>7.141062</td>
      <td>1.73000</td>
      <td>6.950000</td>
      <td>11.36000</td>
      <td>16.955000</td>
      <td>37.9700</td>
    </tr>
    <tr>
      <th>medv</th>
      <td>506.0</td>
      <td>22.532806</td>
      <td>9.197104</td>
      <td>5.00000</td>
      <td>17.025000</td>
      <td>21.20000</td>
      <td>25.000000</td>
      <td>50.0000</td>
    </tr>
  </tbody>
</table>
</div>

The independent variables are correlated with each other, that means this dataset has Multicolinearity inside the data. By droping some features that are highly correlated with each other features it can be reduced.

### Preprocessing
For this dataset Standard Scaler is used for preprocessing process. Standardize features by removing the mean and scaling to unit variance. Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data. Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Mean and standard deviation are then stored to be used on later data using :meth:`transform`.

The standard score of a sample `x` is calculated as:

    z = (x - u) / s

### Model Bulding