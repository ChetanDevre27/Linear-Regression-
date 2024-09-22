import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import classification_report
claimants=pd.read_csv("C:/1-python/ML/LR/claimants.csv")
# there are CLAMG and LOSS are having continus data rest

c1=claimants.drop('CASENUM',axis=1)
c1.head(11)
c1.describe()

# lets check whether there are any null value
c1.isna().sum()
# there are several null values
#if we will  dropna()

c1.dtypes
mean_value=c1.CLMAGE.mean()
mean_value

# now impute the same
c1.CLMAGE=c1.CLMAGE.fillna(mean_value)
c1.CLMAGE.isna().sum()
# hence all null value of CLMAGE has been filled by mean value

mode_CLMSEX=c1.CLMSEX.mode()
mode_CLMSEX
c1.CLMSEX=c1.CLMSEX.fillna((mode_CLMSEX)[0])
c1.CLMSEX.isna().sum()

# CLMINSUR is also categorical data hence mode impution is applied
mode_CLMINSUR=c1.CLMINSUR.mode()
mode_CLMINSUR
c1.CLMINSUR=c1.CLMINSUR.fillna((mode_CLMINSUR)[0])
c1.CLMINSUR.isna().sum()

#SEATBELT is categorical data hence go for mode imputation

mode_SEATBELT=c1.SEATBELT.mode()
mode_SEATBELT
c1.SEATBELT=c1.SEATBELT.fillna((mode_SEATBELT)[0])
c1.SEATBELT.isna().sum()

# now the person we met an accident will hire the atternev or no

#lets build the model
logit_model=sm.logit('ATTORNEY ~CLMAGE+LOSS+CLMINSUR+CLMSEX+SEATBELT',data=c1).fit()
logit_model.summary()
#in logistic regression we do not have R sqr value only check p=value
#SEATBELT is statistically insignificant ignore and proceed
logit_model.summary2()
#


# NOw lets us go for prediction
pred=logit_model.predict(c1.iloc[:,1:])
# here we are applying all row cols from 1 as cols 0 is ATTORNEY
#target Value
#lets check the preformance of the performance of the model
fpr,tpr,thresholds=roc_curve(c1.ATTORNEY,pred)
# we are applying actual values and predicted value so as to get
# false positive rate, true positive rate and threshold

optimal_idx=np.argmax(tpr-fpr)
optimal_threshold=thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i=np.arange(len(tpr))
roc=pd.DataFrame({'fpr' :pd.Series(fpr,index=i),'tpr' : pd.Series(tpr,index=i),'1-fpr' : pd.Series(1-fpr,index=i),'tf' : pd.Series(tpr-(1-fpr),index=i),'thresholds' : pd.Series(thresholds,index=i)})


plt.plot(fpr,tpr)
plt.xlabel("False positive rate");plt.ylabel("Trur positive rate ")
roc.iloc[(roc.tf-0).abs().argsort()[:1]]
roc_auc=auc(fpr,tpr)
print("Area under the curve:%f"% roc_auc)

# tpr vs 1-fpr
# plot topr vs 1-fpr
fig, ax=pl.subplot()
pl.plot(roc['tpr'],color='red')
pl.plot(roc['1-fpr'],color='blue')
pl.xlabel('True positive rate')
pl.ylabel('Reciver operating Charactaristic')
ax.set_xticklabes([])
# 



# filter all thr cells with zero
c1['pred']=np.zeros(1340)
c1.loc[pred>optimal_threshold,'pred']=1

# lets check the classification report
classification=classification_report(c1['pred'], c1['ATTORNEY'])
classification

# splitting the data into train and test

train_data, test_data=train_test_split(c1,test_size=0.3)

model=sm.logit('ATTORNEY ~CLMAGE+LOSS+CLMINSUR+CLMSEX+SEATBELT',data=train_data).fit()
model.summary()
model.summary2()


# lets go for prediction
test_pred=logit_model.predict(test_data)
test_data['test_pred']=np.zeros(402)
test_data.loc[test_pred>optimal_threshold,'test_pred']=1

# confusion matrix
confusion_matrix=pd.crosstab(test_data.test_pred,test_data.ATTORNEY)
confusion_matrix
accuracy_test=(143+151)/(402)
accuracy_test
# 0.7313432835820896 this is going to chane with everytime when you run

# classification report
classification_test=classification_report(test_data['test_pred'],test_data['ATTORNEY'])
classification_test
#acuracy=0.73

# ROC curve and AUC
fpr,tpr,threshold=metrics.roc_curve(test_data['ATTORNEY'], test_pred)

#plot ROC curve
plt.plot(fpr,tpr);plt.xlabel('false positive rate');plt.ylabel("true positive rate")

roc_auc_test=metrics.auc(fpr,tpr)
roc_auc_test

# prediction on train data
train_pred=logit_model.predict(train_data)
train_data['train_pred']=np.zeros(938)
train_data.loc[train_pred>optimal_threshold,'train_pred']=1

# confusion matrix
confusion_matrix=pd.crosstab(train_data.train_pred,train_data.ATTORNEY)
confusion_matrix
accuracy_train=(345+347)/(938)
accuracy_train

# classification report
classification_train=classification_report(train_data['train_pred'],train_data['ATTORNEY'])
classification_train
#acuracy=0.69

# Roc curve and AUC curve 
fpr,tpr,threshold=metrics.roc_curve(train_data['ATTORNEY'], train_pred)

#plot ROC curve
plt.plot(fpr,tpr);plt.xlabel('false positive rate');plt.ylabel("true positive rate")

roc_auc_train=metrics.auc(fpr,tpr)
roc_auc_test







