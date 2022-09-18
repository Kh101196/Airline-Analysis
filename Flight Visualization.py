#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import statsmodels
import matplotlib.pyplot as plt
import math


# In[2]:


# Read in data 
flight = pd.read_csv("flight.csv")
flight.head()


# In[3]:


# Firstly, the coach price is surveyed
coach_price_arr = flight["coach_price"]
print("The maximum value of coach price "+ str(coach_price_arr.max()))
print("The minimum value of coach price "+ str(coach_price_arr.min()))
print("The avearage value of coach price "+ str(coach_price_arr.mean()))


# In[77]:


plt.figure(figsize=(10,8))
sns.histplot(coach_price_arr, bins=20, element="step", stat="proportion", color="orange")
plt.title("The distribution of Coach Price")
plt.xlabel("Coach Price")
plt.ylabel("The Proportion of Clients")


# In[5]:


# the data centralizes in the range from ~150 to ~550
# most clients chose the stadard price oscillate from 350 to 450
list_most_sale = []
for i in coach_price_arr:
    if i >= 350 and i <= 450:
        list_most_sale.append(i)
print(len(list_most_sale)/len(flight)) 
# the groups of clients above occupy of more 0.5% in the proportion, otherwise more than half compared to other prices


# In[6]:


# The boxplot can be plot to compare among ranges of values
plt.figure(figsize=(8,5))
sns.boxplot(data=flight, x=coach_price_arr, palette = "pastel", width=0.9)


# In[7]:


# mean value nearly 400 that already calculated is ~376.58 (the vertical line in the box)
# The box ranges from ~330 to ~415
# The prices that is below 150 and above 550 is really scarce


# In[8]:


flight["hours"].unique()


# In[9]:


# The maximum time length taken to take a flight is 8 hours
# We can survey to confirm that there is a strong relationship between ticket's price and time length


# In[10]:


# Visualize the coach ticket prices for flights that are 8 hours long
price_8_hours = flight[flight["hours"]==8]
plt.figure(figsize=(10,8))
sns.histplot(data = price_8_hours, x="coach_price", stat="proportion", bins=50, alpha=0.2)
plt.title("The distribution of coach price")
plt.xlabel("Coach Price")
plt.ylabel("The Proportion of flight")


# In[11]:


print("The mean value: " + str(np.mean(price_8_hours["coach_price"])))
print("The median value: " + str(np.median(price_8_hours["coach_price"])))


# In[12]:


# In most of the 8hrs's flight trips, the clients chose the offers whose prices lies in the range from nearly $400 to nearly $500
# There is some outliers which lies below $200 and nearly $600
# If you get the trip prolonging 8 hours, the cost of $500 is reasonable


# In[13]:


# We continue to consider whether other factors contribute to expand the costs in the 8hrs's flight besides the petrol and airstaff's costs
hue_list = ["inflight_meal", "inflight_entertainment", "inflight_wifi", "weekend", "redeye"]
# They are all binary factors
plt.figure(figsize=(10,8))
for i in range(len(hue_list)):
    plt.subplot(2,3,1)
    sns.histplot(data = price_8_hours, x="coach_price", stat="density", element="step", bins=50, alpha=0.2)
    plt.subplot(2,3,i+2)
    plt.subplots_adjust(wspace=0.7)
    sns.histplot(data = price_8_hours, x="coach_price", hue=hue_list[i], element="step", stat="density", bins=50, alpha=0.2)


# In[14]:


# Most of these flight usually get entertaiment and wifi service. However, it is not usual that customer is offer the meal
# The much more price is paid, the more likely the high quality services and meals are offered
# If the trips need more time to travel, of course they are often schedule on the weekend
# Most of these trips not travel overnight that may relates to the safety issues


# In[21]:


# Let survey how are flight delay times distributed?
# Data needs to be subset that the ditribution can be seen because this array of delay have outliers
plt.figure(figsize=(10,8))
sns.histplot(flight.delay[flight.delay<1400])
plt.title("The distribution of delay time")
plt.xlabel("Delay Time")
plt.ylabel("The Number of Flight")


# In[22]:


# delay time have the highest frequency is 10 min
# sometimes, delay time is revolved around 20 to 30 minutes
# rarely, it delays more than 50 mins


# In[23]:


# Find the relationship betwween coach price and first class price, investigate whether it is a linear relationship
plt.figure(figsize=(10,8))
for i in range(len(hue_list)):
    plt.subplot(2,3,1)
    sns.scatterplot(data=flight, x="coach_price", y="firstclass_price")
    plt.subplot(2,3,i+2)
    sns.scatterplot(data=flight, x="coach_price", y="firstclass_price", hue=hue_list[i], palette="pastel")
    plt.subplots_adjust(hspace=0.5, wspace=0.7)


# In[65]:


# The relationship can bee seen more visually by lmplot. 
# Instead plot all data, one subset can be randomly drawn to see the trend. In this case, 10% of dataset is randomly drawn
perc = 0.1
flight_sub = flight.sample(n = int(flight.shape[0]*perc))
sns.lmplot(x = "coach_price", y = "firstclass_price", data = flight_sub, line_kws={'color': 'pink'}, lowess=True)
plt.show()
plt.clf()


# In[25]:


# We can see there is obvious positve linear relationship between coach price and first class price
# It is likely to realize the more higher both price options are, more inflight entertainment and wifi are offered
# While relationship with inflight meal is ambiguous
# Especially, we can confirm that both price option are influenced the most by weekend
# The price is booked in the weekend is always more expensive than normal days 
# The difference between firstclass and coach price looks like more larger than weekdays


# In[57]:


# The next relationship is investigated is between the number of passenger and length of flight

sns.lmplot(data=flight_sub,x="hours",y="passengers")


# In[63]:


# The simple plot above show that it becomes overfitting when plotting several discrete values
# There are many values are missed because the length of dataset is 129780 flights
# In order to handle overplotting of discreteness in smaller dataset, small variations are added to local of each point

sns.lmplot(data=flight_sub, x="hours", y="passengers", x_jitter=0.25, scatter_kws={"s": 10, "alpha":0.3}, fit_reg = False)


# In[75]:


# How about coach prices differ for redeyes and non-redeyes on each day of the week?
plt.figure(figsize=(15,8))
sns.boxplot(data=flight, x="day_of_week", y="coach_price", hue="redeye", palette="pastel")
plt.title("compare the prices among weekdays")
plt.xlabel("weekdays")
plt.ylabel("coach price")


# In[78]:


# Generally, whether the flight is taken at the weekend or on the weekdays, the average price of overnight flight is always lower
# As being seen before, the average prices is more expensive at the weekend
# The average costs overnight or in normal time are nearly the same among weekdays and at the weekend

