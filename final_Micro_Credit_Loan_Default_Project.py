#!/usr/bin/env python
# coding: utf-8

# # About The Analysis Topic

# A Microfinance Institution (MFI) is an organization that offers financial services to low income populations. MFS becomes very useful when targeting especially the unbanked poor families living in remote areas with not much sources of income. The Microfinance services (MFS) provided by MFI are Group Loans, Agricultural Loans, Individual Business Loans and so on. 
# Many microfinance institutions (MFI), experts and donors are supporting the idea of using mobile financial services (MFS) which they feel are more convenient and efficient, and cost saving, than the traditional high-touch model used since long for the purpose of delivering microfinance services. Though, the MFI industry is primarily focusing on low income families and are very useful in such areas, the implementation of MFS has been uneven with both significant challenges and successes.
# Today, microfinance is widely accepted as a poverty-reduction tool, representing $70 billion in outstanding loans and a global outreach of 200 million clients.
# We are working with one such client that is in Telecom Industry. They are a fixed wireless telecommunications network provider. They have launched various products and have developed its business and organization based on the budget operator model, offering better products at Lower Prices to all value conscious customers through a strategy of disruptive innovation that focuses on the subscriber. 
# They understand the importance of communication and how it affects a person’s life, thus, focusing on providing their services and products to low income families and poor customers that can help them in the need of hour. 
# They are collaborating with an MFI to provide micro-credit on mobile balances to be paid back in 5 days. The Consumer is believed to be defaulter if he deviates from the path of paying back the loaned amount within the time duration of 5 days. For the loan amount of 5 (in Indonesian Rupiah), payback amount should be 6 (in Indonesian Rupiah), while, for the loan amount of 10 (in Indonesian Rupiah), the payback amount should be 12 (in Indonesian Rupiah). 
# The sample data is provided to us from our client database. It is hereby given to you for this exercise. In order to improve the selection of customers for the credit, the client wants some predictions that could help them in further investment and improvement in selection of customers. 
# 

# # About Target Variable

# Build a model which can be used to predict in terms of a probability for each loan transaction, whether the customer will be paying back the loaned amount within 5 days of insurance of loan. In this case, Label ‘1’ indicates that the loan has been payed i.e. Non- defaulter, while, Label ‘0’ indicates that the loan has not been payed i.e. defaulter.  

# # About Dataset

# - label = Flag indicating whether the user paid back the credit amount within 5 days of issuing the loan{1:success, 0:failure}
# - msisdn = mobile number of user
# - aon = age on cellular network in days
# - daily_decr30 = Daily amount spent from main account, averaged over last 30 days (in Indonesian Rupiah)
# - daily_decr90 = Daily amount spent from main account, averaged over last 90 days (in Indonesian Rupiah)
# - rental30 = Average main account balance over last 30 days
# - rental90 = Average main account balance over last 90 days
# - last_rech_date_ma = Number of days till last recharge of main account
# - last_rech_date_da = Number of days till last recharge of data account
# - last_rech_amt_ma = Amount of last recharge of main account (in Indonesian Rupiah)
# - cnt_ma_rech30 = Number of times main account got recharged in last 30 days
# - fr_ma_rech30 = Frequency of main account recharged in last 30 days
# - sumamnt_ma_rech30 = Total amount of recharge in main account over last 30 days (in Indonesian Rupiah)
# - medianamnt_ma_rech30 = Median of amount of recharges done in main account over last 30 days at user level (in Indonesian Rupiah)
# - medianmarechprebal30 = Median of main account balance just before recharge in last 30 days at user level (in Indonesian Rupiah)
# - cnt_ma_rech90 = Number of times main account got recharged in last 90 days
# - fr_ma_rech90 = Frequency of main account recharged in last 90 days
# - sumamnt_ma_rech90 = Total amount of recharge in main account over last 90 days (in Indonasian Rupiah)
# - medianamnt_ma_rech90 = Median of amount of recharges done in main account over last 90 days at user level (in Indonasian Rupiah)
# - medianmarechprebal90 = Median of main account balance just before recharge in last 90 days at user level (in Indonasian Rupiah)
# - cnt_da_rech30 = Number of times data account got recharged in last 30 days
# - fr_da_rech30 = Frequency of data account recharged in last 30 days
# - cnt_da_rech90 = Number of times data account got recharged in last 90 days
# - fr_da_rech90 = Frequency of data account recharged in last 90 days
# - cnt_loans30 = Number of loans taken by user in last 30 days
# - amnt_loans30 = Total amount of loans taken by user in last 30 days
# - maxamnt_loans30 = maximum amount of loan taken by the user in last 30 days
# - medianamnt_loans30 = Median of amounts of loan taken by the user in last 30 days
# - cnt_loans90 = Number of loans taken by user in last 90 days
# - amnt_loans90 = Total amount of loans taken by user in last 90 days
# - maxamnt_loans90 = maximum amount of loan taken by the user in last 90 days
# - medianamnt_loans90 = Median of amounts of loan taken by the user in last 90 days
# - payback30 = Average payback time in days over last 30 days
# - payback90 = Average payback time in days over last 90 days
# - pcircle = telecom circle
# - pdate = date

# # Importing Essential Libraries

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# # Reading the dataset

# In[5]:


df=pd.read_csv("F:\\priyanka_micro_credit\\Micro-Credit-Project\\Micro_Credit_Project\\dataset.csv")
df.head(10)


# # Data Pre-Processing

# In[6]:


#We dont require first column as it is only index column.
df.drop('Unnamed: 0', inplace=True, axis=1)


# In[7]:


#Structure of data
df.shape


# In[8]:


df.info()


# Observation : 
#     - We are having missing data.
#     - The variables are of three data types int, float or object. 
#    

# In[9]:


#Checking the mobile no columns
df['msisdn'].nunique()


# In[10]:


df.msisdn.value_counts()


# So, we can observe that there are various unique values that are present in this column so we will be removing it.

# In[11]:


df.drop('msisdn', inplace=True, axis=1)


# In[12]:


#Checking the telecom circle columns
df['pcircle'].nunique()


# In[13]:


#All the entiries in the column are same so it is of no use to us.
df.drop('pcircle', inplace=True, axis=1)


# In[14]:


#Checking the date columns
df['pdate'].nunique()


# So, as we can see that all the entries in the column are not unique so we will not be removing it and we will be using label encoder in later stages.

# In[15]:


#Changing object to datetime
df["pdate"]= pd.to_datetime(df["pdate"])


# In[16]:


df.info()


# In[17]:


df.describe()


# Observation
# - Apart from 3 columns(msisdn(mobile no), pcircle(telecom circle) & pdate) every variable in dataset is numerical (ordinal, binary or continuous) in nature. So we will be treating these variables when we further progress in our analysis.
# - Apart from rental30 and rental90 no other variable can be negative in nature so we will be removing those negative rows in further steps of analysis.

# In[18]:


df.shape


# In[19]:


#We will be removing the rows where cnt_loans90(People who haven't loans in past 90 days)<=0 because those rows are of no use to us.
df = df[df['cnt_loans90']>0]


# In[20]:


print('People who have not took loan in past 90 days',df.shape)


# In[21]:


df.columns


# In[22]:


df.shape


# In[23]:


#Age on network can never be negative so we will be removing it.
df = df[df['aon']>=0]
df.shape


# In[24]:


#Daily decrease on main account in 30 days of network should never be negative so we will be removing it.
df = df[df['daily_decr30']>=0]
df.shape


# In[25]:


#Daily decrease on main account in 90 days of network should never be negative so we will be removing it.
df = df[df['daily_decr90']>=0]
df.shape


# In[26]:


# Let's check the negative values present in our dataset

(df.drop(['pdate','label','rental30','rental90'],axis=1) >= 0).all()


# In[27]:


#Number of days till last recharge of main account of network should never be negative so we will be removing it.
df = df[df['last_rech_date_ma']>=0]
df.shape


# In[28]:


#Number of days till last recharge of data account of network should never be negative so we will be removing it.
df = df[df['last_rech_date_da']>=0]
df.shape


# In[29]:


#Median of amount of recharges done in main account over last 90 days at user level (in Indonesian Rupiah) of network should never be negative so we will be removing it.
df = df[df['medianmarechprebal90']>=0]
df.shape


# In[30]:


#Median of amount of recharges done in main account over last 30 days at user level (in Indonesian Rupiah) of network should never be negative so we will be removing it.
df = df[df['medianmarechprebal30']>=0]
df.shape


# In[31]:


# Let's recheck the negative values present in our dataset

(df.drop(['pdate','label','rental30','rental90'],axis=1) >= 0).all()


# Now we have no negative values so we will be proceeding further in our analysis.

# In[32]:


# Lets split the year and month from the Date for better visulatization

df['Year']=df['pdate'].dt.year
df['Month']=df['pdate'].dt.month
df.Month = df.Month.map({1:'JAN',2:'FEB',3:'MAR',4:'APR',5:'MAY',6:'JUN',7:'JUL',8:'AUG',9:'SEPT',10:'OCT',11:'NOV',12:'DEC'})
df['Day']=df['pdate'].dt.day


# In[33]:


# Let's drop the unnecessary column
df.drop(['pdate'],axis=1,inplace=True)


# In[34]:


df.head(10)


# In[35]:


df['Year'].nunique


# In[36]:


#As there is no unique values in Year column so we will be removing it
df.drop(['Year'],axis=1,inplace=True)


# In[37]:


#Lets check no of defalulters and non defaulters
total_customers = df.shape[0]
print("Total customers: {}".format(total_customers))
non_defaulter_customers = df[df["label"] == 1].shape[0]
print("Non defaulter customer: {}".format(non_defaulter_customers))
defaulter_customers = df[df["label"] == 0].shape[0]
print("Defaulter customer : {}".format(defaulter_customers))
non_defaulter_percent = (non_defaulter_customers/total_customers)*100
print("Non Defaulter Customer percentage : {0:.2f}%".format(non_defaulter_percent))
defaulter_percent = (defaulter_customers/total_customers)*100
print("Deafulter Customer : {0:.2f}%".format(defaulter_percent))


# # EXPLORATORY DATA ANALYSIS

# In[38]:


df.describe().T


# Observation
# - We can observe that in most of the variables the mean is varying very much from its median so there will be a high no of outliers present in it.
# - Also observe that apaprt from some of the variables the standard deviation is also quite high. 

# # Univariate Analysis

# In[39]:


df.hist(figsize=(20,20))
plt.show()


# By this we can observe data is not normally distributed(Mostly positively skewed).

# In[40]:


plt.figure(figsize=(10,6))
sns.countplot(df["label"])
plt.show()


# In[41]:


plt.figure(figsize=(10,6))
sns.countplot(df["Month"])
plt.show()


# Given dataset is only for 3 months (July, August, and June).

# In[42]:


cols=['last_rech_amt_ma','cnt_ma_rech90', 'fr_ma_rech90',
       'cnt_da_rech90', 'fr_da_rech90', 'cnt_loans30',
      'amnt_loans90', 'maxamnt_loans90', 'medianamnt_loans90', ]
for i in cols:
    plt.subplots(figsize=(30,8))
    sns.countplot(i,data=df)
    plt.show()


# # Bi-Variate Analysis

# In[43]:


#Label vs aon
sns.barplot(x=df['label'],y=df['aon'])


# In[44]:


#Label vs Daily Amount Reduced from Balance Over 30 days
plt.subplots(figsize=(12,8))
sns.barplot(x='label',y='daily_decr30',data=df)
plt.title('Average Daily Amount Reduced from Balance Over 30 days')
plt.xlabel("Defaulter Or Non-Defaulter")
plt.ylabel("Daily Amount Spent(30 days)")
plt.xticks(rotation=0);


# In[45]:


#Label vs Daily Amount Reduced from Balance Over 90 days
plt.subplots(figsize=(12,8))
sns.barplot(x='label',y='daily_decr90',data=df)
plt.title('Average Daily Amount Reduced from Balance Over 90 days')
plt.xlabel("Defaulter Or Non-Defaulter")
plt.ylabel("Daily Amount Spent(90 days)")
plt.xticks(rotation=0);


# In[46]:


#Label vs Average main account balance over last 30 days
plt.subplots(figsize=(12,8))
sns.barplot(x='label',y='cnt_ma_rech30',data=df)
plt.title('No.of times main account got recharged in last 30 days')
plt.xlabel("Defaulter Or Non-Defaulter")
plt.ylabel("Main account got recharged(30 days) count")
plt.xticks(rotation=0);


# In[47]:


#Label vs Average main account balance over last 30 days
plt.subplots(figsize=(12,8))
sns.barplot(x='label',y='rental30',data=df)
plt.title('Average main account balance over last 30 days')
plt.xlabel("Defaulter Or Non-Defaulter")
plt.ylabel("Main account balance(30 days)")
plt.xticks(rotation=0);


# Observation:
#     - Defaulters have  max average balance of 2000,repayers has an avg main balance over 2500.

# In[48]:


#Label vs No of loans taken over last 30 days
plt.subplots(figsize=(12,8))
sns.barplot(x='label',y='cnt_loans30',data=df)
plt.title('Number of loans taken by user in last 30 days')
plt.xlabel("Defaulter Or Non-Defaulter")
plt.ylabel("No of loans taken(30 days)")
plt.xticks(rotation=0);


# OBSERVATION:
# - Defaulters has been given a max 1 loan.
# - Non-Defaulters had taken a max of 3 loans.

# In[49]:


#Label vs No of loans taken over last 90 days
plt.subplots(figsize=(12,8))
sns.barplot(x='label',y='cnt_loans90',data=df)
plt.title('Number of loans taken by user in last 90 days')
plt.xlabel("Defaulter Or Non-Defaulter")
plt.ylabel("No of loans taken(90 days)")
plt.xticks(rotation=0);


# OBSERVATION:
# - Defaulters has been given a max 15 loan.
# - Non-Defaulters had taken a max of 18 loans.

# In[50]:


#Label vs Average payback time in last 30 days
plt.subplots(figsize=(12,8))
sns.barplot(x='label',y='payback30',data=df)
plt.title('Average payback time in days over last 30 days')
plt.xlabel("Defaulter Or Non-Defaulter")
plt.ylabel("Average payback time(30 days)")
plt.xticks(rotation=0);


# OBSERVATION:
# - For Defaulters the average payback time in last 30 days is 2.5.
# - For Non-Defaulters the average payback time in last 30 days is 3.5.

# In[51]:


#Label vs Average payback time in last 90 days
plt.subplots(figsize=(12,8))
sns.barplot(x='label',y='payback90',data=df)
plt.title('Average payback time in days over last 90 days')
plt.xlabel("Defaulter Or Non-Defaulter")
plt.ylabel("Average payback time(90 days)")
plt.xticks(rotation=0);


# OBSERVATION:
# - For Defaulters the average payback time in last 90 days is 3.
# - For Non-Defaulters the average payback time in last 90 days is 4.5.

# In[52]:


corr = df.corr()
corr


# In[53]:


plt.figure(figsize=(40,35))
sns.heatmap(corr,annot=True,cmap="winter")


# Highly Correlated
# - daily_decr30
# - daily_decr90
# - cnt_loan30
# - amnt_loan30
# 
# Least Correlated
# - aon
# - maxamnt_loans30
# - last_rech_date_da
# - cnt_loans90

# In[54]:


df.shape


# In[55]:


#We will be dropping the columns that are highly and less correlated with each other inorder to avoid multicolinearity problem

df.drop(columns=["daily_decr30","fr_ma_rech30","payback30","rental30","medianamnt_loans30","amnt_loans30",
                "fr_da_rech30","cnt_da_rech30","sumamnt_ma_rech30","fr_ma_rech30","cnt_ma_rech30"],axis=1, inplace = True)


# In[56]:


df.shape


# # Plotting Outliers

# In[57]:


df.plot(kind='box',subplots=True,layout=(14,3),figsize=(12,30))
plt.show()


# Observation:
# - We can observe from the data that we have a lot of outliers present in the datasets if we ought to remove it we will be removing a significant amount of our data.
# - So as a final verdict we will not be removing any of the outliers present in it.

# In[58]:


df.info()


# As month is categorical data we will be changing it to numerical using Label Encoder.

# # Encoding

# In[59]:


#Let's encode our dataset

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
mappings=list()

for column in range(len(df.columns)):
    df[df.columns[column]]=le.fit_transform(df[df.columns[column]])
    mappings_dict={index: label for index, label in enumerate(le.classes_)}
    mappings.append(mappings_dict)


# # Checking and Treating Skewness

# In[60]:


df.skew()


# In[61]:


for col in df.columns:
    if df[col].skew()>0.55:
        df[col]=np.log1p(df[col])


# In[62]:


#Splitting dataset into Independent and Target variable
df_x=df.drop(columns="label")
y=pd.DataFrame(df["label"])


# In[63]:


df_x.shape, y.shape


# # Scaling The Data

# In[64]:


#Scaling the input variable
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(df_x)
x=pd.DataFrame(x,columns=df_x.columns)


# In[65]:


x.describe()


# # Model Building

# In[66]:


#Coverting to train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)


# In[67]:


from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve,roc_auc_score,auc


# In[68]:


models= [LogisticRegression(),DecisionTreeClassifier(),GaussianNB()]
for m in models:
    m.fit(x_train,y_train)
    print("Score of ",m," :",m.score(x_train,y_train))
    predm=m.predict(x_test)
    print('Scores')
    print("Accuracy Score : ",accuracy_score(y_test,predm))     
    print("--------------------------------------------------------------------------------------------")
    print("\n")


# In[69]:


#Cross Validation
from sklearn.model_selection import cross_val_score
models= [LogisticRegression(),DecisionTreeClassifier(),GaussianNB()]
rocscore=[]
for m in models:
    score=cross_val_score(m,x,y,cv=2,scoring="accuracy")
    print("Score of ",m," is :",score)
    print("Mean Score : ",score.mean())
    print("Standard Deviation : ",score.std())
    false_positive_rate,true_positive_rate, thresholds=roc_curve(y_test,predm)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    print('roc auc score : ', roc_auc)
    rocscore.append(roc_auc*100)
    print('\n')
    print('Classification Report:\n',classification_report(y_test,predm))
    print('\n')
    print('Confusion Matrix:\n',confusion_matrix(y_test,predm))
    print('\n')
    print("--------------------------------------------------------------------------------------------")
    print("\n")


# # We are getting the best score using LogisticRegression after Cross-Validation.

# We will take a look at Ensemble Methods to boost our scores.

# # Ensemble Methods

# In[70]:


from sklearn.ensemble import GradientBoostingClassifier,ExtraTreesClassifier
emodels= [GradientBoostingClassifier(),AdaBoostClassifier(),ExtraTreesClassifier()]
for m in emodels:
    m.fit(x_train,y_train)
    print("Score of ",m," :",m.score(x_train,y_train))
    predm=m.predict(x_test)
    print('Scores')
    print("Accuracy Score : ",accuracy_score(y_test,predm))     
    print("--------------------------------------------------------------------------------------------")
    print("\n")


# # Now we can observe that using ExtraTreeClassifier we have boosted up our score by 99%.

# In[71]:


#Cross Validation
from sklearn.model_selection import cross_val_score
emodels= [GradientBoostingClassifier(),AdaBoostClassifier(),ExtraTreesClassifier()]
rocscore=[]
for m in emodels:
    false_positive_rate,true_positive_rate, thresholds=roc_curve(y_test,predm)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    print('roc auc score : ', roc_auc)
    rocscore.append(roc_auc*100)
    print('\n')
    print('Classification Report:\n',classification_report(y_test,predm))
    print('\n')
    print('Confusion Matrix:\n',confusion_matrix(y_test,predm))
    print('\n')
    print("--------------------------------------------------------------------------------------------")
    print("\n")


# In[72]:


emodels= [LogisticRegression(),DecisionTreeClassifier(),GaussianNB(),GradientBoostingClassifier(),AdaBoostClassifier(),ExtraTreesClassifier()]
for name in emodels:
    false_positive_rate,true_positive_rate, thresholds=roc_curve(y_test,predm)
    roc_auc=auc(false_positive_rate, true_positive_rate)    
    plt.figure(figsize=(10,40))
    plt.subplot(911)
    plt.title(name)
    plt.plot(false_positive_rate,true_positive_rate,label='AUC = %0.2f'% roc_auc)
    plt.plot([0,1],[0,1],'r--')
    plt.legend(loc='lower right')
    plt.ylabel('True_positive_rate')
    plt.xlabel('False_positive_rate')
    print('\n\n')


# In[73]:


#Extra Trees Classifier is the best model so we will find out it's best parameter using GridSearchCV
from sklearn.model_selection import GridSearchCV
et=ExtraTreesClassifier()
parameters = {'n_estimators': [50,100,200,300],'max_features': ['auto', 'sqrt', 'log2'],'max_depth': [None, 5, 10],'min_samples_split': [4, 6],
          'min_samples_leaf': [1, 2]}
GridSearchCV(et,parameters)


# In[74]:


def grid_cv(mod,parameters,scoring):
    clf = GridSearchCV(mod,parameters,scoring, cv=2,verbose=1,refit=True,n_jobs=-1)
    clf.fit(x,y)
    print(clf.best_params_)
    print(clf.best_score_)


# As it was taking a lot of time in order to find the best parameters we take randoml take the paramters
# - {'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 300}

# In[75]:


#Using ExtraTreeClassifier method with best parameters
etc=ExtraTreesClassifier(n_estimators=300,max_depth=None, min_samples_leaf= 1, max_features= 'log2',min_samples_split=4)
etc.fit(x_train,y_train)
print("Score of ",etc," :",etc.score(x_train,y_train))
predetc=etc.predict(x_test)
print('Scores')
print("Accuracy Score : ",accuracy_score(y_test,predetc))
print("Classification Report : \n",classification_report(y_test,predetc))
print("Confusion_matrix : ",confusion_matrix(y_test,predetc))
print("--------------------------------------------------------------------------------------------")
print("\n")


# In[76]:


y_probs = etc.predict_proba(x_test)
y_probs = y_probs[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
fpr
plt.plot([0,1],[0,1])
plt.plot(fpr,tpr,label='ExtraTreesClassifier')
plt.xlabel('false positive rate')
plt.ylabel('True positive rate')
plt.title('ExtraTreesClassifier')
plt.show()
print('roc_auc_score = ',roc_auc_score(y_test, y_probs))


# # Saving the Model

# In[77]:


import pickle
filename= "MicroCredit.pkl"
pickle.dump(etc,open(filename,'wb'))

