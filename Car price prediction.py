import pandas as pd
import numpy as np
import seaborn as sns
cars=pd.read_csv('C:/1-python/ML/LR/Cars.csv')
cars
#1. measure central tendancy
#2. masure the disperasion
#3.third moment buisness decision
#4.Fourth moment buisness decision

cars.describe()
# graphical repres
import matplotlib.pyplot as plt
plt.bar(height=cars.HP,x=np.arange(1,82,1 ))
sns.displot(cars.HP)

# data is right skewd
plt.boxplot(cars.HP)
# there are sevral outlier in hp
# similar process with other col

sns.distplot(cars.MPG)
# data is slighty distributed


sns.distplot(cars.VOL)
plt.boxplot(cars.VOL)
# data slightly right distributed

sns.displot(cars.SP)
plt.boxplot(cars.SP)
#sevral outliers

sns.displot(cars.WT)
plt.boxplot(cars.WT)
# There are several outliers

# lets plot join plot , join plot to show scatter
#histogram

sns.jointplot(x=cars['HP'],y=cars['MPG'])

# now lets plot count plot
plt.figure(1,figsize=(16,10))
sns.countplot(cars['HP'])
#92 HP value occured by 7 times

#QQplot
from scipy import stats 
import pylab
stats.probplot(cars.MPG,dist='norm',plot=pylab)
plt.show()
#MPG data is normally distributed
# there are 10 scatter plots need ti plotted one by 
# to plot,so we can use pair plots

import seaborn as sns
sns.pairplot(cars.iloc[:,:])
# lineaarity: 
# direction:
# strength: 

    
#now check r value bet var
cars.corr()
# you can check the SP and HP
#value id 0.97 and same way you can check Wt and VOl it has got 0.999 which is great

# noe although we observed strongly corr pair
import statsmodels.formula.api as smf
ml1=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
ml1.summary()

# R square value observed 0.771 < 0.85
# p- value of WT and VOL is 0.814 and 0.556 which is very high it mean it is greater than 0.05 Wt and VOL col
# we need to ignore
# or delete. Instead deleting 81 entries

# identify there any influential value
# to check can used influential index

import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
# 76 is the value which has got outliers
#3 go to the data frame and check 76th entry
# delete that entry
cars_new=cars.drop(cars.index[[76]])

# again aplly regression to cars_new
ml_new= smf.ols('MPG~WT+VOL+HP+SP',data=cars_new).fit()
ml_new.summary()
# R square value is 0.819 but p value are same hence not 

rsq_hp=smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp=1/(1-rsq_hp)
vif_hp

rsq_wt=smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_wt=1/(1-rsq_wt)

rsq_vol=smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_vol=1/(1-rsq_vol)

rsq_sp=smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquaredsi
vif_sp=1/(1-rsq_sp)
#vif_wt=639.53 and vif_vol=638.80 hence vif_wt is greater , thum rule should be not greater than 1

#storing the value in datafram
d1={'Variables':['HP','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
vif_frame=pd.DataFrame(d1)
vif_frame


# lets drop wt and apply corrle to remailing three
final_ml=smf.ols('MPG~VOL+SP+HP',data=cars).fit()
final_ml.summary()
# R sqr value is 0.770 and p value 0.00,0.012<0.05

# prediction
pred=final_ml.predict(cars)
pred

#QQplot
res=final_ml.resid
sm.qqplot(res)
plt.show()

stats.probplot(res,dist='norm',plot=pylab)
plt.show()

# 
sns.residplot(x=pred,y=cars.MPG,lowess=True)
plt.xlable('Fitted')
plt.ylabel('Residual')
plt.title('Fitted Vs Residual')

# splitting the data into train and test data
from sklearn.model_selection 