---
layout: post
title: Land Classification
description: Classifying different parts of land using Machine Learning
image: assets/images/landclass.png
nav-menu: true
---

There are multiple satellites that capture the data about the amount of light intensity reflected at different frequencies from the Earth at a very granular geographic level. Some of this information can be used to classify the Earth into different buckets - built-up, barren, green or water. The training data contains different parameters classified into the 4 classes. 

# Column Description

* Numeric columns <b>X1 to X6 and I1 to I6</b> define characteristics about the land piece
* <b>ClusterID</b> is a categorical column which clusters a type of land together
* <b>target</b> is the output categorical column which needs to be found for the test dataset
    * 1 = Green Land
    * 2 = Water
    * 3 = Barren Land
    * 4 = Built-up 
    
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

```python
df = pd.read_csv('land_train.csv') # loading the dataset
df.head()
```

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
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>I1</th>
      <th>I2</th>
      <th>I3</th>
      <th>I4</th>
      <th>I5</th>
      <th>I6</th>
      <th>clusterID</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>207</td>
      <td>373</td>
      <td>267</td>
      <td>1653</td>
      <td>886</td>
      <td>408</td>
      <td>0.721875</td>
      <td>-1.023962</td>
      <td>2.750628</td>
      <td>0.530316</td>
      <td>0.208889</td>
      <td>0.302087</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>194</td>
      <td>369</td>
      <td>241</td>
      <td>1539</td>
      <td>827</td>
      <td>364</td>
      <td>0.729213</td>
      <td>-1.030143</td>
      <td>2.668501</td>
      <td>0.546537</td>
      <td>0.203306</td>
      <td>0.300930</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>214</td>
      <td>385</td>
      <td>264</td>
      <td>1812</td>
      <td>850</td>
      <td>381</td>
      <td>0.745665</td>
      <td>-1.107047</td>
      <td>3.000315</td>
      <td>0.546156</td>
      <td>0.181395</td>
      <td>0.361382</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>212</td>
      <td>388</td>
      <td>293</td>
      <td>1882</td>
      <td>912</td>
      <td>402</td>
      <td>0.730575</td>
      <td>-1.077747</td>
      <td>3.006150</td>
      <td>0.530083</td>
      <td>0.156835</td>
      <td>0.347172</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>249</td>
      <td>411</td>
      <td>332</td>
      <td>1773</td>
      <td>1048</td>
      <td>504</td>
      <td>0.684561</td>
      <td>-0.941562</td>
      <td>2.713079</td>
      <td>0.494370</td>
      <td>0.205742</td>
      <td>0.257001</td>
      <td>6</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

```python
dfs = df.sample(frac=0.1).reset_index(drop=True) # shuffles the data and takes a fraction of it
dfs.shape
```
    (48800, 14)

# Column level preprocessing
### Scatter Plot

```python
# scatter plot of a part of data
pd.plotting.scatter_matrix(dfs[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6']], figsize=(17,10));
```

![png](output_5_0.png)

### Correlation Plot

```python
import seaborn as sns
sns.set(style="white")

# finding correlation
corr = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6']].corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0.5, square=True, linewidths=.5, cbar_kws={"shrink": .5});
```

![png](output_7_0.png)

From the scatter plot and correlation plot, we can conclude that-
- The $ X $ features are almost linearly correlated with each other. Moreover, they show similar correlation with $ I $ features. Therefore, we can merge them into one (average).
- $ I5 $ is almost constant and shows less variance (see the histogram). Therefore, it can be dropped.
- $ I1 $ and $ I4 $ are highly correlated. They can be merged (average).
- $ I2 $ shows inverse correlation with $ I1 $. It can be dropped.

Remaining columns: $ X $<sub>avg</sub>, $ I $<sub>14</sub>, $ I $<sub>3</sub>, $ I $<sub>6</sub>. 

#### Note: Highly correlated columns are dropped since they provide no extra information to the model and hampers the performance.

### Selecting features

```python
# preprocessing columns
df['X'] = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']].mean(axis=1) # merging all X features in one
df = df.drop(['X1', 'X2', 'X3', 'X4', 'X5', 'X6'], axis=1)      # dropping the rest
df['I14'] = df[['I1', 'I4']].mean(axis=1)                       # averaging I1 and I4
df = df.drop(['I1', 'I4', 'I2', 'I5'], axis=1)                  # dropping the rest
df.head()
```

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
      <th>I3</th>
      <th>I6</th>
      <th>clusterID</th>
      <th>target</th>
      <th>X</th>
      <th>I14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.750628</td>
      <td>0.302087</td>
      <td>6</td>
      <td>1</td>
      <td>632.333333</td>
      <td>0.626095</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.668501</td>
      <td>0.300930</td>
      <td>6</td>
      <td>1</td>
      <td>589.000000</td>
      <td>0.637875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.000315</td>
      <td>0.361382</td>
      <td>6</td>
      <td>1</td>
      <td>651.000000</td>
      <td>0.645910</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.006150</td>
      <td>0.347172</td>
      <td>6</td>
      <td>1</td>
      <td>681.500000</td>
      <td>0.630329</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.713079</td>
      <td>0.257001</td>
      <td>6</td>
      <td>1</td>
      <td>719.500000</td>
      <td>0.589465</td>
    </tr>
  </tbody>
</table>
</div>

```python
# df.to_csv('land_train_pruned.csv', index=False) # saving the data
```

# Row level preprocessing

```python
df.describe()
```

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
      <th>I3</th>
      <th>I6</th>
      <th>clusterID</th>
      <th>target</th>
      <th>X</th>
      <th>I14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>487998.000000</td>
      <td>487998.000000</td>
      <td>487998.000000</td>
      <td>487998.000000</td>
      <td>487998.000000</td>
      <td>487998.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.602914</td>
      <td>0.098514</td>
      <td>4.180263</td>
      <td>1.062301</td>
      <td>2567.438750</td>
      <td>0.295633</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.106819</td>
      <td>0.147183</td>
      <td>1.645535</td>
      <td>0.350805</td>
      <td>1840.983848</td>
      <td>0.250844</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.177197</td>
      <td>-0.297521</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>-83.666667</td>
      <td>-0.848529</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.694444</td>
      <td>0.016150</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1015.000000</td>
      <td>0.067379</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.267495</td>
      <td>0.056333</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1839.000000</td>
      <td>0.188264</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.430122</td>
      <td>0.161845</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>4015.666667</td>
      <td>0.515738</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.663032</td>
      <td>0.662566</td>
      <td>8.000000</td>
      <td>4.000000</td>
      <td>10682.333333</td>
      <td>5.755439</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.target.value_counts().plot('bar');
```

![png](output_15_0.png)

Samples per example: <br>
1: 472987 (97%) <br>
2: 0 <br>
3: 14630 <br>
4: 381 <hr>
This clearly shows the imbalance in classes. While training on such data even if the model predits <b>class 1</b> everytime, the accuracy will be 97%.
Therefore, we need to balance the classes and for this, the approach used is given below.

- Due to very less samples in Class 4, it can be removed.
- Class 1 can be divided randomly into 10 subparts (Samples in each part - 47300).
- Class 3 can be duplicated 2 times (Samples - 43890).
- Trian 10 separate classification models with one part of Class 1 each along with same Class 3 samples in each model.
- Take the average of all models while testing.

### Normalizing features

```python
from sklearn import preprocessing

x = df[['I3', 'I6', 'I14', 'X']].values # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_n = pd.DataFrame(x_scaled, columns=['I3', 'I6', 'I14', 'X'])
df_norm = df[['clusterID', 'target']].join(df_n)

df_norm.head()
```

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
      <th>clusterID</th>
      <th>target</th>
      <th>I3</th>
      <th>I6</th>
      <th>I14</th>
      <th>X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>1</td>
      <td>0.501320</td>
      <td>0.624535</td>
      <td>0.223294</td>
      <td>0.066506</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>1</td>
      <td>0.487258</td>
      <td>0.623330</td>
      <td>0.225077</td>
      <td>0.062481</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>1</td>
      <td>0.544073</td>
      <td>0.686296</td>
      <td>0.226294</td>
      <td>0.068240</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>1</td>
      <td>0.545072</td>
      <td>0.671495</td>
      <td>0.223935</td>
      <td>0.071073</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>1</td>
      <td>0.494891</td>
      <td>0.577575</td>
      <td>0.217747</td>
      <td>0.074602</td>
    </tr>
  </tbody>
</table>
</div>

### Splitting data into 10 parts 

```python
# separating classes
df_1 = df_norm[df_norm.target==1]
df_1 = df_1.sample(frac=1).reset_index(drop=True) # shuffling rows
df_3 = df_norm[df_norm.target==3]

df_1.shape[0], df_3.shape[0]
```
    (472987, 14630)


```python
df_1_split = np.split(df_1, np.arange(47300, 472987, 47300),axis=0) # splits dataframe (of Target=1) in 10 
[i.shape for i in df_1_split]
```
    [(47300, 6),
     (47300, 6),
     (47300, 6),
     (47300, 6),
     (47300, 6),
     (47300, 6),
     (47300, 6),
     (47300, 6),
     (47300, 6),
     (47287, 6)]

```python
df_3_dup = pd.concat([df_3]*3, ignore_index=True) # duplicate data (of Target=3) 3 times
df_3_dup.shape
```
    (43890, 6)

```python
for i in range(len(df_1_split)):
    df_1_split[i] = df_1_split[i].append(df_3_dup, ignore_index=True) # merge Target 1 and 3

[i.shape for i in df_1_split]
```
    [(91190, 6),
     (91190, 6),
     (91190, 6),
     (91190, 6),
     (91190, 6),
     (91190, 6),
     (91190, 6),
     (91190, 6),
     (91190, 6),
     (91177, 6)]

```python
df_1_split[9].target.value_counts().plot('bar');
```

![png](output_29_0.png)

```python
# saving all parts
# for i in range(len(df_1_split)):
#     df_1_split[i].to_csv(f'land_train_split_{i+1}.csv', index=False)
```

# Preprocessing Test Data

```python
df_test = pd.read_csv('land_test.csv')
df_test.shape
```
    (2000000, 13)

```python
df_test.head()
```

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
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>I1</th>
      <th>I2</th>
      <th>I3</th>
      <th>I4</th>
      <th>I5</th>
      <th>I6</th>
      <th>clusterID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>338</td>
      <td>554</td>
      <td>698</td>
      <td>1605</td>
      <td>1752</td>
      <td>1310</td>
      <td>0.393834</td>
      <td>-0.350045</td>
      <td>1.565423</td>
      <td>0.311659</td>
      <td>0.304781</td>
      <td>-0.043789</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>667</td>
      <td>976</td>
      <td>1187</td>
      <td>1834</td>
      <td>1958</td>
      <td>1653</td>
      <td>0.214167</td>
      <td>-0.181467</td>
      <td>1.050679</td>
      <td>0.196439</td>
      <td>0.164085</td>
      <td>-0.032700</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>249</td>
      <td>420</td>
      <td>402</td>
      <td>1635</td>
      <td>1318</td>
      <td>736</td>
      <td>0.605302</td>
      <td>-0.712650</td>
      <td>2.268984</td>
      <td>0.441984</td>
      <td>0.293497</td>
      <td>0.107348</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>111</td>
      <td>348</td>
      <td>279</td>
      <td>1842</td>
      <td>743</td>
      <td>328</td>
      <td>0.736917</td>
      <td>-1.162062</td>
      <td>3.074176</td>
      <td>0.551699</td>
      <td>0.080725</td>
      <td>0.425145</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>349</td>
      <td>559</td>
      <td>642</td>
      <td>1534</td>
      <td>1544</td>
      <td>989</td>
      <td>0.409926</td>
      <td>-0.406678</td>
      <td>1.607795</td>
      <td>0.323984</td>
      <td>0.212753</td>
      <td>-0.003249</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>

```python
# preprocessing columns
df_test['X'] = df_test[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']].mean(axis=1) # merging all X features in one
df_test = df_test.drop(['X1', 'X2', 'X3', 'X4', 'X5', 'X6'], axis=1)      # dropping the rest
df_test['I14'] = df_test[['I1', 'I4']].mean(axis=1)                       # averaging I1 and I4
df_test = df_test.drop(['I1', 'I4', 'I2', 'I5'], axis=1)                  # dropping the rest
```

```python
x_test = df_test[['I3', 'I6', 'I14', 'X']].values # returns a numpy array
x_test_scaled = min_max_scaler.transform(x_test)  # applying scaler
df_test_n = pd.DataFrame(x_test_scaled, columns=['I3', 'I6', 'I14', 'X'])
df_test_norm = df_test[['clusterID']].join(df_test_n)
df_test_norm.shape
```
    (2000000, 5)

```python
df_test_norm.head()
```

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
      <th>clusterID</th>
      <th>I3</th>
      <th>I6</th>
      <th>I14</th>
      <th>X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>0.298382</td>
      <td>0.264280</td>
      <td>0.181902</td>
      <td>0.104635</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>0.210245</td>
      <td>0.275830</td>
      <td>0.159576</td>
      <td>0.135875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>0.418850</td>
      <td>0.421701</td>
      <td>0.207780</td>
      <td>0.081460</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>0.556720</td>
      <td>0.752709</td>
      <td>0.226051</td>
      <td>0.064292</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>0.305637</td>
      <td>0.306505</td>
      <td>0.184054</td>
      <td>0.094727</td>
    </tr>
  </tbody>
</table>
</div>

```python
# df_test_norm.to_csv('land_test_preprocessed.csv')
```

# Model

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
```

```python
# different classifiers
num_models = [KNeighborsClassifier(n_neighbors=5), svm.SVC(), LogisticRegression(), GaussianNB(),
              AdaBoostClassifier(), BernoulliNB(), MLPClassifier(),
              RandomForestClassifier(), DecisionTreeClassifier('entropy')]
```

```python
# training 9 different model on different data splits
# taking average of 9 predictions while testing
# testing set is the 10th data split

models = []
for i in range(9):
    df_train = pd.read_csv('land_train_split_{}.csv'.format(i+1))
    df_train = df_train.sample(frac=1).reset_index(drop=True)

    x = df_train.drop('target', axis=1).values
    y = df_train.target.values
    
    model = num_models[7]
    models.append(model.fit(x, y))
    
    print(f'Model {i+1} trained.', end='\r')
```
    Model 9 trained.


```python
# loading the test set
df_test = pd.read_csv('land_train_split_10.csv')
df_test = df_test.sample(frac=1).reset_index(drop=True)

x_test = df_test.drop('target', axis=1).values
y_test = df_test.target.values
```

```python
# making predictions
predictions = []
for i in range(9):
    predictions.append(list(models[i].predict(x_test)))
predictions = np.array(predictions)
```

```python
# getting the mojority Class

pred_maj = []
for i in range(predictions.shape[1]):
    (values, counts) = np.unique(predictions[:,i], return_counts=True)
    pred_maj.append(values[np.argmax(counts)])
pred_maj = np.array(pred_maj)
```

```python
print('Accuracy: {:.2f}%'.format(np.mean(pred_maj==y_test)*100))

precision, recall, f1 = precision_score(y_test, pred_maj), recall_score(y_test, pred_maj), f1_score(y_test, pred_maj)
print('Precision: {:.2f}\nRecall: {:.2f}\nF1 score: {:.2f}'.format(precision, recall, f1))

cm = confusion_matrix(y_test, pred_maj, labels=[1,3])
sns.heatmap(cm, xticklabels=['False', 'True'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix');
```
    Accuracy: 97.58%
    Precision: 0.98
    Recall: 0.97
    F1 score: 0.98

![png](output_46_1.png)

# Results

| Classifier | Accuracy | F1 Score |
| ---------- | -------- | -- |
| Decision Tree (Entropy) | 99.11% | 0.99 |
| Decision Tree (Gini) | 99.15% | 0.99 |
| Random Forest | 99.23% | 0.99 |
| KNN | 98.53% | 0.99 |
| Logistic Regression | 91.38% | 0.92 |
| Gaussian Naive Bayes | 89.83% | 0.90 |
| Bernoulli Naive Bayes | 51.86% | 0.68 |
| Adaboost Classifier | 95.87% | 0.90 |
| ANN | 89.83% | 0.90 |
| All | 97.58% | 0.98 |


# Getting predictions for test set

```python
df_test_set = pd.read_csv("land_test_preprocessed.csv")
df_test_set.drop('Unnamed: 0', inplace=True, axis=1)
df_test_set.head()
```

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
      <th>clusterID</th>
      <th>I3</th>
      <th>I6</th>
      <th>I14</th>
      <th>X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>0.298382</td>
      <td>0.264280</td>
      <td>0.181902</td>
      <td>0.104635</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>0.210245</td>
      <td>0.275830</td>
      <td>0.159576</td>
      <td>0.135875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>0.418850</td>
      <td>0.421701</td>
      <td>0.207780</td>
      <td>0.081460</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>0.556720</td>
      <td>0.752709</td>
      <td>0.226051</td>
      <td>0.064292</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>0.305637</td>
      <td>0.306505</td>
      <td>0.184054</td>
      <td>0.094727</td>
    </tr>
  </tbody>
</table>
</div>




```python
# predictions
predictions = []
for i in range(9):
    predictions.append(list(models[i].predict(df_test_set.values)))
predictions = np.array(predictions)
```


```python
# getting the mojority Class
pred_maj = []
for i in range(predictions.shape[1]):
    (values, counts) = np.unique(predictions[:,i], return_counts=True)
    pred_maj.append(values[np.argmax(counts)])
pred_maj = np.array(pred_maj)

pred_maj.shape
```
    (2000000,)

```python
df_sub = pd.DataFrame(pred_maj, columns=['target'])
df_sub.head()
```

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
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>

```python
df_sub.to_csv('submission.csv', index=False)
```

# Analysis

```python
dfa = pd.read_csv('land_train_split_1.csv')
dfa.groupby(dfa.target).describe().T
```

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
      <th>target</th>
      <th>1</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="8" valign="top">I14</th>
      <th>count</th>
      <td>47300.000000</td>
      <td>43890.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.172823</td>
      <td>0.178143</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.038544</td>
      <td>0.010856</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.126043</td>
      <td>0.128314</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.138245</td>
      <td>0.172255</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.153272</td>
      <td>0.177356</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.208025</td>
      <td>0.183939</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.313051</td>
      <td>0.236896</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">I3</th>
      <th>count</th>
      <td>47300.000000</td>
      <td>43890.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.303832</td>
      <td>0.300403</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.192337</td>
      <td>0.065502</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.005789</td>
      <td>0.024574</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.145690</td>
      <td>0.265956</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.234960</td>
      <td>0.289333</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.455870</td>
      <td>0.326778</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.908952</td>
      <td>0.874047</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">I6</th>
      <th>count</th>
      <td>47300.000000</td>
      <td>43890.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.418548</td>
      <td>0.235797</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.151682</td>
      <td>0.076489</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.077088</td>
      <td>0.029555</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.330767</td>
      <td>0.190215</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.371948</td>
      <td>0.218487</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.483560</td>
      <td>0.258432</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.947382</td>
      <td>0.818775</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">X</th>
      <th>count</th>
      <td>47300.000000</td>
      <td>43890.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.251571</td>
      <td>0.131445</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.172705</td>
      <td>0.032215</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.015047</td>
      <td>0.061382</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.101817</td>
      <td>0.112948</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.202729</td>
      <td>0.126339</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.388372</td>
      <td>0.142888</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.992523</td>
      <td>0.519661</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">clusterID</th>
      <th>count</th>
      <td>47300.000000</td>
      <td>43890.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.127294</td>
      <td>5.833698</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.634507</td>
      <td>0.906823</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.000000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
estimator = models[0].estimators_[5]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = ['clusterID', 'I3', 'I6', 'I14', 'X'],
                class_names = ['1', '3'],
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
import pydot
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('somefile.png')
```

```python
# taking sum of all GINI indexes
a = models[0].feature_importances_
for i in range(8):
    a += models[i+1].feature_importances_

a
```
    array([2.10152634, 0.58177282, 3.40019577, 2.06218839, 0.85431669])

The model cannot be visualized because of its size, so I have taken the sum of <b>gini index</b> of attributes, which are shown below.<br>
clusterID: 2.10 <br>
I3: 0.58 <br>
I6: 3.40 <br>
I14: 2.06 <br>
X: 0.85 <br>
<hr>
Clearly, $ I6 $ classifies the data best.
