# EX-06-Feature-Transformation

## AIM
To Perform the various feature transformation techniques on a dataset and save the data to a file. 

# Explanation
Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

 
# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Transformation techniques to all the feature of the data set
### STEP 4
Save the data to the file

# CODE
### CODE FOR Data_to_Transform.csv
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

df=pd.read_csv("Data_To_Transform.csv")
df

df.skew()

np.log(df["Highly Positive Skew"])

np.reciprocal(df["Moderate Positive Skew"])

np.sqrt(df["Highly Positive Skew"])

np.square(df["Highly Negative Skew"])

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df

df["Moderate Positive Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Positive Skew"])
df

df["Moderate Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df

df["Highly Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Highly Negative Skew"])
df

df.skew()

from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew_1'],line='45')
plt.show()

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()

df["Moderate Positive Skew_1"]=qt.fit_transform(df[["Moderate Positive Skew"]])
sm.qqplot(df['Moderate Positive Skew'],line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew_1'],line='45')
plt.show()

df["Highly Positive Skew_1"]=qt.fit_transform(df[["Highly Positive Skew"]])
sm.qqplot(df['Highly Positive Skew'],line='45')
plt.show()

sm.qqplot(df['Highly Positive Skew_1'],line='45')
plt.show()

df
```
# OUPUT
### OUTPUT FOR Data_to_Transform.csv
![output](./1.png)
![output](./2.png)
![output](./3.png)
![output](./4.png)
![output](./5.png)
![output](./6.png)
![output](./7.png)
![output](./8.png)
![output](./9.png)
![output](./10.png)
![output](./11.png)
![output](./12.png)
![output](./13.png)
![output](./14.png)
![output](./15.png)
![output](./16.png)
![output](./17.png)
![output](./18.png)

### CODE FOR titanic_dataset.csv
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

df=pd.read_csv("titanic_dataset.csv")
df

df.drop("Name",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])
df

df.info()

from sklearn.preprocessing import OrdinalEncoder

embark=["C","S","Q"]
emb=OrdinalEncoder(categories=[embark])
df["Embarked"]=emb.fit_transform(df[["Embarked"]])

from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
df

np.log(df["Age"])

np.reciprocal(df["Fare"])

np.sqrt(df["Embarked"])

df["Age _boxcox"], parameters=stats.boxcox(df["Age"])
df

df["Pclass _boxcox"], parameters=stats.boxcox(df["Pclass"])
df

df["Fare _yeojohnson"], parameters=stats.yeojohnson(df["Fare"])
df

df["SibSp _yeojohnson"], parameters=stats.yeojohnson(df["SibSp"])
df

df["Parch _yeojohnson"], parameters=stats.yeojohnson(df["Parch"])
df

df.skew()

from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)


df["Age_1"]=qt.fit_transform(df[["Age"]])
sm.qqplot(df['Age'],line='45')
plt.show()

sm.qqplot(df['Age_1'],line='45')
plt.show()

df["Fare_1"]=qt.fit_transform(df[["Fare"]])
sm.qqplot(df["Fare"],line='45')
plt.show()

sm.qqplot(df['Fare_1'],line='45')
plt.show()

df["Parch_1"]=qt.fit_transform(df[["Parch"]])
sm.qqplot(df["Parch"],line='45')
plt.show()

sm.qqplot(df['Parch_1'],line='45')
plt.show()

df
```
### OUTPUT
### OUTPUT FOR titanic_dataset.csv
![output](./19.png)
![output](./20.png)
![output](./21.png)
![output](./22.png)
![output](./23.png)
![output](./24.png)
![output](./25.png)
![output](./26.png)
![output](./27.png)
![output](./28.png)
![output](./29.png)
![output](./30.png)
![output](./31.png)
![output](./32.png)
![output](./33.png)
![output](./34.png)
![output](./35.png)
![output](./36.png)
![output](./37.png)
![output](./38.png)
![output](./39.png)
### RESULT
The various feature transformation techniques has been performed on the given datasets and the data are saved to a file.