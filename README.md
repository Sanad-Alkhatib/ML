# Pumpkin Price Prediction using Linear Regression

## 1️⃣ Project Overview
This project predicts pumpkin prices using historical pricing and characteristics.  
We focus on **Linear Regression** to capture the relationship between pumpkin features and price.

### Goal
- Predict pumpkin prices based on **City, Package, Variety, Date, Item Size, Low Price, High Price, and Origin**.  
- Understand which features affect pricing the most.  

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

df = df[['Date', 'City Name', 'Package', 'Variety', 'Origin', 'Item Size', 'Low Price', 'High Price']]

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

