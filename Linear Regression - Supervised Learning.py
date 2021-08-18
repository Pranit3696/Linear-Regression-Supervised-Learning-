#!/usr/bin/env python
# coding: utf-8

# # Pranit Prabhakaran
# 
# ### Registration Id : WYUP6LJMNT

# #  Task1 : Prediction using Supervised Learning

# Problem Discription : Predict the percentage of an student based on the no. of study hours. 

# ### Importing all the Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error,mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading Dataset

# In[2]:


# Reading data from remote link

url = "http://bit.ly/w-data"
stud_data = pd.read_csv(url)
print("Data imported successfully")

stud_data.head(10)


# ### Data Visualization

# In[3]:


# Plotting the distribution of scores

stud_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ### Data Preparation

# In[4]:


X = stud_data.iloc[:, :-1].values  
y = stud_data.iloc[:, 1].values  


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# ### Training the Model

# In[6]:


Lr = LinearRegression()  
Lr.fit(X_train, y_train) 

print("Training complete.")


# ### Regression Line Plot

# In[7]:


# Plotting the regression line
line = Lr.coef_*X+Lr.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# ### Prediction on Test Data

# In[8]:


print(X_test) # Testing data - In Hours
y_pred = Lr.predict(X_test) # Predicting the scores
y_pred[0:5]


# In[9]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# ### Evaluating the Model

# In[10]:


print('Root mean squared error(RMSE) is:'),np.sqrt(mean_squared_error(y_test,y_pred))
print('Mean Absolute Error:',mean_absolute_error(y_test,y_pred))


# ### Predicted score if a student studies for 9.25 hrs/ day

# In[11]:


# You can also test with your own data
hours = 9.25
pred = Lr.predict(np.array(hours).reshape(-1,1))
print("No of Hours = ",hours)
print("Predicted Score = ",pred[0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




