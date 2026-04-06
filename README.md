# Pumpkin Price Classification using Logistic Regression

## 1️⃣ Project Overview
This project predicts pumpkin prices using historical pricing and characteristics.  
We focus on **Linear Regression** to capture the relationship between pumpkin features and price.

### Goal
- Predict pumpkin prices based on **City, Package, Variety, Date, Item Size, Low Price, High Price, and Origin**.  
- Understand which features affect pricing the most.


### ⚠️ Note:
'Low Price' is highly correlated with 'High Price', which may introduce data leakage.
However, it was intentionally included to help the model capture price range behavior.

### Dataset
- Rows: 1757  
- Columns: 26 ('City Name', 'Package', 'Variety', 'Date', 'Low Price', 'High Price',e.g.)  
- This project uses the dataset from [Microsoft ML-For-Beginners](https://github.com/microsoft/ML-For-Beginners).

### Dataset Preview
<img width="1635" height="324" alt="Screenshot 2026-04-05 142122" src="https://github.com/user-attachments/assets/6b21d29c-a390-4af3-af95-4cd37943703b" />

### Data Info
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1757 entries, 0 to 1756
Data columns (total 26 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   City Name        1757 non-null   object 
 1   Type             45 non-null     object 
 2   Package          1757 non-null   object 
 3   Variety          1752 non-null   object 
 4   Sub Variety      296 non-null    object 
 5   Grade            0 non-null      float64
 6   Date             1757 non-null   object 
 7   Low Price        1757 non-null   float64
 8   High Price       1757 non-null   float64
 9   Mostly Low       1654 non-null   float64
 10  Mostly High      1654 non-null   float64
 11  Origin           1754 non-null   object 
 12  Origin District  131 non-null    object 
 13  Item Size        1478 non-null   object 
 14  Color            1141 non-null   object 
 15  Environment      0 non-null      float64
 16  Unit of Sale     162 non-null    object 
 17  Quality          0 non-null      float64
 18  Condition        0 non-null      float64
 19  Appearance       0 non-null      float64
 20  Storage          0 non-null      float64
 21  Crop             0 non-null      float64
 22  Repack           1757 non-null   object 
 23  Trans Mode       0 non-null      float64
 24  Unnamed: 24      0 non-null      float64
 25  Unnamed: 25      103 non-null    object 
dtypes: float64(13), object(13)
memory usage: 357.0+ KB

### Missing Values
City Name             0
Type               1712
Package               0
Variety               5
Sub Variety        1461
Grade              1757
Date                  0
Low Price             0
High Price            0
Mostly Low          103
Mostly High         103
Origin                3
Origin District    1626
Item Size           279
Color               616
Environment        1757
Unit of Sale       1595
Quality            1757
Condition          1757
Appearance         1757
Storage            1757
Crop               1757
Repack                0
Trans Mode         1757
Unnamed: 24        1757
Unnamed: 25        1654
dtype: int64

---

## Data Preparation

Before training any model, we must clean the data.

### Feature Selection

We selected only useful columns:
- Date  
- City Name  
- Package  
- Variety  
- Origin  
- Item Size  
- Low Price  
- High Price  

### Why?

Because many columns had:
- Because the supervisor requested this.
- Too many missing values  
- No useful information
  
'''python
df = df[['Date', 'City Name', 'Package', 'Variety', 'Origin', 'Item Size', 'Low Price', 'High Price']]
'''

---

### Handling Missing Values

- Text columns → filled with '"Unknown"'  
- Numeric columns → filled with **mean values**

### Why?

Machine Learning models cannot handle missing values.

'''python
text_columns = ['City Name', 'Package', 'Variety', 'Origin', 'Item Size']
for col in text_columns:
    df[col] = df[col].fillna('Unknown')

df['High Price'] = df['High Price'].fillna(df['High Price'].mean())
df['Low Price'] = df['Low Price'].fillna(df['Low Price'].mean())
'''

---

## Feature Engineering

### Item Size Encoding

We converted sizes into numbers:

- sml → 1  
- med → 2  
- med-lge → 3  
- lge → 4  
- xlge → 5  
- jbo → 6  
- exjbo → 7
  
### Why?

Models only understand numbers, not text.

'''python
levels = ['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']
size_mapping = {size: i+1 for i, size in enumerate(levels)}

df['Item_Size_Num'] = df['Item Size'].map(size_mapping)

print(df[['Item Size', 'Item_Size_Num']].head(5))
'''
### Results
Item Size  Item_Size_Num
0       lge            4.0
1       lge            4.0
2       med            2.0
3       med            2.0
4       lge            4.0

---

### Handling Missing Size

We replaced missing values using **median**.

### Why?

Median is better when data contains outliers.

---

### Removing Outliers

We used **IQR method** on 'High Price'.

### Why?

Outliers can negatively affect model performance.
'''python
Q1 = df['High Price'].quantile(0.25)
Q3 = df['High Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['High Price'] >= lower_bound) & (df['High Price'] <= upper_bound)]
'''

---

### Date Processing

We converted `Date` and extracted:

- Year  
- Month  
- Day  

### Why?

To allow the model to learn **seasonal patterns**.

'''python
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
'''

---

### Filtering Pumpkin Varieties

We selected only:

- HOWDEN TYPE  
- PIE TYPE  
- MINIATURE  
- FAIRYTALE
  
### Why?

This is what the supervisor wants because he believes that the other types will cause a bias problem, and these are the most important and common types.

---
### Handling Imbalanced Data

We increased FAIRYTALE samples by ~52%.

### Why?

Because it had fewer samples compared to other varieties.

'''python
fairytale_rows = df[df['Variety'] == 'FAIRYTALE']
n_add = int(len(fairytale_rows) * 0.52)
fairytale_extra = fairytale_rows.sample(n=n_add, replace=True, random_state=42)
df = pd.concat([df, fairytale_extra], ignore_index=True)
'''
### Results

Variety
HOWDEN TYPE    542
PIE TYPE       467
MINIATURE      310
FAIRYTALE      200
Name: count, dtype: int64

---

## Target Variable

We created a new column:

- 1 → price above average  
- 0 → price below average  

### Why?

To convert the problem into a **classification task**.

'''python
threshold = df['High Price'].mean()
df['Price_Category'] = df['High Price'].apply(lambda x: 1 if x > threshold else 0)
'''

---

## Data Visualization

We explored the data using:

- Price distribution
<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/344ea826-b2df-4e0c-9d74-9a5e19987245" />

- City distribution
<img width="571" height="554" alt="image" src="https://github.com/user-attachments/assets/d01ed1b4-cfc5-4468-a65d-85cc38645502" />

- Variety distribution
<img width="571" height="544" alt="image" src="https://github.com/user-attachments/assets/7092042d-1a0f-4f35-8c8e-1436975bff81" />

- Correlation heatmap  
<img width="608" height="528" alt="image" src="https://github.com/user-attachments/assets/da2b53b5-9d54-4d54-b946-ba338451e8a7" />

---

## Preparing Data for Model

We selected features:

- City Name  
- Package  
- Variety  
- Origin  
- Item_Size_Num  
- Year  
- Month  
- Day  
- Low Price  

Then applied **One-Hot Encoding**.

### Why?

Because Logistic Regression requires numerical input.
'''python
X = df[['City Name', 'Package', 'Variety', 'Origin', 'Item_Size_Num', 'Year', 'Month', 'Day', 'Low Price']]
X = pd.get_dummies(X, columns=['City Name', 'Package', 'Variety', 'Origin'], drop_first=True)
y = df['Price_Category']
'''

---

## Train / Validation / Test Split

We split the data into:

- 70% Training  
- 15% Validation  
- 15% Testing  

### Why?

- Training → learn  
- Validation → tune  
- Testing → final evaluation
  
'''python
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split( X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
'''

---

## Logistic Regression Model

We trained the model using:

LogisticRegression(max_iter=1000)

### Why?

Simple and effective for classification
Works well with structured data

'''python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
'''

---

## Model Evaluation

We evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score

### Results

- Accuracy: 0.956140350877193
- Precision: 0.9366197183098591
- Recall: 0.9925373134328358
- F1 Score: 0.9637681159420289

---

### Conclusion

Data cleaning was essential due to missing values
Feature engineering improved model performance
Handling imbalance made the model more fair
Logistic Regression successfully classified pumpkin prices








