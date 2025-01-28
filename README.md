# Sampling Methods Evaluation

This project focuses on evaluating various **sampling methods** for handling imbalanced datasets, with **accuracy** as the primary evaluation metric. It explores the effectiveness of different resampling techniques and machine learning algorithms on a credit card fraud detection dataset.

---

## Features

- **Comprehensive Sampling Techniques**: Includes undersampling, oversampling, and hybrid sampling methods.
- **Imbalanced Data Handling**: Uses SMOTE, RandomUnderSampler, and SMOTEENN for creating balanced datasets.
- **Model Evaluation**: Assesses multiple machine learning models on resampled data.
- **Primary Metric**: Focuses on **accuracy** as the key performance indicator.

---
## Input Data
- A credit card dataset:
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>

## Sampling Methods

### 1. No Sampling
- **Description**: The baseline method where the original imbalanced dataset is used without any resampling.
- **Purpose**: To compare the performance of other techniques against the unmodified dataset.

### 2. RandomUnderSampler
- **Description**: Balances the dataset by randomly removing samples from the majority class.
- **Advantages**:
  - Reduces dataset size, speeding up training.
  - Simple and effective for small datasets.
- **Limitations**: Risk of losing important majority class information.

### 3. SMOTE (Synthetic Minority Oversampling Technique)
- **Description**: Generates synthetic samples for the minority class by interpolating between existing samples.
- **Advantages**:
  - Preserves majority class data.
  - Effective for imbalanced datasets with sufficient minority samples.
- **Limitations**: May create overlapping samples, increasing class confusion.

### 4. SMOTEENN
- **Description**: A hybrid method combining SMOTE for oversampling and Edited Nearest Neighbors (ENN) for cleaning noisy samples.
- **Advantages**:
  - Improves dataset quality by removing noise.
  - Enhances model performance on complex datasets.
- **Limitations**: Computationally expensive for large datasets.

---

## Machine Learning Models

### 1. Logistic Regression

### 2. Random Forest Classifier

### 3. EasyEnsembleClassifier

### 4. XGBoost

### 5. CatBoost
---

## Output Data
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sampling Technique</th>
      <th>Model</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sampling1</td>
      <td>Model1</td>
      <td>0.931034</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sampling1</td>
      <td>Model2</td>
      <td>0.982759</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sampling1</td>
      <td>Model3</td>
      <td>0.982759</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sampling1</td>
      <td>Model4</td>
      <td>0.974138</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sampling1</td>
      <td>Model5</td>
      <td>0.991379</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sampling2</td>
      <td>Model1</td>
      <td>0.939655</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sampling2</td>
      <td>Model2</td>
      <td>0.982759</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sampling2</td>
      <td>Model3</td>
      <td>0.982759</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sampling2</td>
      <td>Model4</td>
      <td>0.956897</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Sampling2</td>
      <td>Model5</td>
      <td>0.982759</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Sampling3</td>
      <td>Model1</td>
      <td>0.870690</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Sampling3</td>
      <td>Model2</td>
      <td>0.982759</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Sampling3</td>
      <td>Model3</td>
      <td>0.939655</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Sampling3</td>
      <td>Model4</td>
      <td>0.922414</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Sampling3</td>
      <td>Model5</td>
      <td>0.939655</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Sampling4</td>
      <td>Model1</td>
      <td>0.853448</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Sampling4</td>
      <td>Model2</td>
      <td>0.965517</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Sampling4</td>
      <td>Model3</td>
      <td>0.948276</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Sampling4</td>
      <td>Model4</td>
      <td>0.948276</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Sampling4</td>
      <td>Model5</td>
      <td>0.956897</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Sampling5</td>
      <td>Model1</td>
      <td>0.896552</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Sampling5</td>
      <td>Model2</td>
      <td>0.991379</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Sampling5</td>
      <td>Model3</td>
      <td>0.974138</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Sampling5</td>
      <td>Model4</td>
      <td>0.956897</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Sampling5</td>
      <td>Model5</td>
      <td>0.974138</td>
    </tr>
  </tbody>
</table>
</div>

---

## Conclusion

The combination of diverse sampling techniques and powerful machine learning models allows for a comprehensive evaluation of strategies to handle imbalanced datasets. By focusing on accuracy, the project aims to identify the most effective approach for real-world applications.

---

