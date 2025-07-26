import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from  sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df = pd.read_csv('smoking.csv')


df

df_x_train = pd.read_csv('x_train.csv')


df_x_test = pd.read_csv('x_test.csv')


df_y_train = pd.read_csv('y_train.csv')
df_y_test = pd.read_csv('y_test.csv')


df_x_train

df_x_test

df.isnull().sum()

df.duplicated()

df.select_dtypes([int,float]).columns

cols =['ID', 'age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)',
       'eyesight(right)', 'hearing(left)', 'hearing(right)', 'systolic',
       'relaxation', 'fasting blood sugar', 'Cholesterol', 'triglyceride',
       'HDL', 'LDL', 'hemoglobin', 'Urine protein', 'serum creatinine', 'AST',
       'ALT', 'Gtp', 'dental caries', 'smoking']
for col in cols:
  plt.boxplot(df[col])
  plt.title(col)
  plt.show()


for i in df.select_dtypes([int,float]):
  q1 = df[i].quantile(0.25)
  q3 = df[i].quantile(0.75)
  IQR = q3-q1
  uf = q3 + (1.5*IQR)
  lf = q1 - (1.5*IQR)
  if lf>0 or uf>0:
   df[i] = df[i].clip(lf,uf)


for i in df.select_dtypes([int,float]):
  plt.boxplot(df[i])
  plt.title(i)
  plt.show()

def IQR(x):
  q1 = x.quantile(0.25)
  q3 = x.quantile(0.75)
  IQR = q3-q1
  lf = q1 - (1.5*IQR)
  uf = q3 + (1.5*IQR)
  print("lower_fence",lf)
  print("upper_fence",uf)


IQR(df["dental caries"])

lb = LabelEncoder()

df["gender"] = lb.fit_transform(df["gender"])
df["oral"] = lb.fit_transform(df["oral"])
df["tartar"] = lb.fit_transform(df["tartar"])


df.select_dtypes([object]).columns

dff = pd.get_dummies(df,columns=['gender', 'oral', 'tartar' ],
                drop_first=True)
dff.columns

dff

dff.drop(columns=['ID'],inplace=True)

df.drop(columns=['oral'],inplace=True)

X = dff.drop(columns="smoking")
y = dff["smoking"]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=88)


lr = LogisticRegression()

from sklearn.ensemble import RandomForestClassifier
rm = RandomForestClassifier(random_state = 77)

# Drop rows with NaN values from X_train and y_train
X_train.dropna(inplace=True)
y_train.dropna(inplace=True)

rm.fit(X_train,y_train)
print("train_score",rm.score(X_train,y_train))
print("test_score",rm.score(X_test,y_test))

from sklearn.model_selection import GridSearchCV
para = {"n_estimators": [450,500,550],
        "criterion": ["gini","entropy"],
        "max_depth": [50,60],
        "min_samples_split": [110,120],
        "min_samples_leaf": [80,90,100,110,120]}

rft = RandomForestClassifier(random_state = 77)
grid = GridSearchCV(rft,param_grid=para,cv=5,verbose = 1,n_jobs = -1)
grid.fit(X_train,y_train)

# it takes too much time in grid search