import pandas as pd
import numpy as np
import seaborn as sns
wcat=pd.read_csv("C:/1-python/ML/LR/wc-at.csv")
wcat
# Exploratry data anylysis
# measure the central Tendency
# 2. Measure of dispresion
# 3. Third moment buisness decision
# 4. Fourth buisness moment decision

wcat.info()
wcat.describe()

# graphical repesentaion
import matplotlib.pyplot as plt
plt.bar(height=wcat.AT, x=np.arange(1,110,1 ))
plt.distplot(wcat.AT)
sns.boxplot(wcat.AT)
# data is right skewed
# scatter plot

plt.scatter(x=wcat['Waist'], y=wcat['AT'], color='green')
# direction: positive , liniarity:modrate, strength: poor

# calculate Correlation coef
np.corrcoef(wcat.Waist, wcat.AT)

# check direction using cover factor
cov_output=np.cov(wcat.Waist,wcat.AT)[0,1]
cov_output

# now apply LR model
import statsmodels.formula.api as smf
#all ML model implimented using sklearn
# but for this used statmodel
# backend calcu is bita-0 and bita-1
model=smf.ols('AT~Waist',data=wcat).fit()
model.summary()


#Regresion Line
pred1=model.predict(pd.DataFrame(wcat['Waist']))
plt.scatter(wcat.Waist, wcat.AT)
plt.plot(wcat.Waist,pred1,'r')
plt.show()

# error calculation
res1=wcat.AT-pred1
np.mean(res1)
# it must be zero and here it 10 ^14=~0

res_sqr1= res1*res1
msel1=np.mean(res_sqr1)
rmsel=np.sqrt(msel1)
rmsel
#₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪
plt.scatter(x=np.log(wcat['Waist']), y=wcat['AT'], color='brown')
np.corrcoef(np.log(wcat.Waist),wcat.AT)

# r value is 0.82<0.85 hence modrate liniarity
model2=smf.ols('AT~np.log(Waist)',data=wcat).fit()
model2.summary()

# again check the R-square value=0.67 which is less than 0.8
# p value is 0 less than 0.05
pred2=model2.predict(pd.DataFrame(wcat['Waist']))

plt.scatter(wcat.Waist, wcat.AT)
plt.plot(wcat.Waist,pred2,'r')
plt.legend(['Predicted line','Observed data'])

# error calculation
res2=wcat.AT-pred2
res_sqr2= res2*res2
msel2=np.mean(res_sqr2)
rmsel2=np.sqrt(msel2)
rmsel2
# value of emse is 32.49688490932127
#₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪
plt.scatter(x=wcat['Waist'], y=np.log(wcat['AT']), color='orange')
np.corrcoef(wcat.Waist,np.log(wcat.AT))
# r value is 0.84<0.85 hence modrate linarity
model3=smf.ols('np.log(AT)~Waist',data=wcat).fit()
model3.summary()

pred3=model3.predict(pd.DataFrame(wcat['Waist']))
pred3_at=np.exp(pred3)

plt.scatter(wcat.Waist, np.log(wcat.AT))
plt.plot(wcat.Waist,pred3,'r')
plt.legend(['Predicted line','Observed data'])



# error calculation
res3=wcat.AT-pred3_at
res_sqr3= res3*res3
msel3=np.mean(res_sqr3)
rmsel3=np.sqrt(msel3)
rmsel3
# value of rmse is 38.529001758071416

#₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪
# polinomial Transformation
model4=smf.ols('np.log(AT)~Waist+I(Waist*Waist)',data=wcat).fit()
# Y is log(AT) and X=Waist
model4.summary()
#R-squared=0.779<0..85 there is scope of improvement
#p=0.000<0.05 hence acceptable
#bita 0 = -7.8241
# bita 1= 0.2289

pred4=model4.predict(pd.DataFrame(wcat.Waist))
pred4
pred4_at=np.exp(pred4)
pred4_at

#₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪

#regression line

plt.scatter(wcat.Waist, np.log(wcat.AT))
plt.plot(wcat.Waist,pred4,'r')
plt.legend(['Predicted line','Observed data model3'])
plt.show()

#error calculation
res4=wcat.AT-pred4_at
res_sqr4= res4*res4
mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4
#32.24
# Among of all modedl model 4 is best model
#₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪
data={'model':pd.Series(['SLR','Log_model','Exp_model','Ploy_model'])}
data
table_rmse=pd.DataFrame(data)
table_rmse

#₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪₪
# We have to genralize the best model
from sklearn.model_selection import train_test_split
train,test=train_test_split(wcat,test_size=0.2)
plt.scatter(train.Waist,np.log(train.AT))
plt.scatter(test.Waist,np.log(test.AT))
final_model=smf.ols('np.log(AT)~Waist+I(Waist*Waist)',data=wcat).fit()

final_model.summary()

test_pred =final_model.predict(pd.DataFrame(test))
test_pred_at=np.exp(test_pred)
test_pred_at

train_pred=final_model.predict(pd.DataFrame(train))
train_pred_at=np.exp(train_pred)
train_pred_at

# Evaluation on test data
test_err=test.AT-test_pred_at
test_sqr=test_err*test_err
test_mse=np.mean(test_sqr)
test_rmse=np.sqrt(test_mse)
test_rmse

# Evaluation on train data

train_res=train.AT-train_pred_at
train_sqr=train_err*train_err
train_mse=np.mean(train_sqr)
train_rmse=np.sqrt(train_mse)
train_rmse




