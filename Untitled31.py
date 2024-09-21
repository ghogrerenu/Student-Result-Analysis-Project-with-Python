#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv("Expanded_data_with_more_features.csv")
print(df.head())


# In[48]:


print(df)


# # Basic Exploration

# In[5]:


df.describe()


# In[6]:


df.info


# In[9]:


df.isnull().sum()


# # Drop unnamed column

# In[11]:


df = df.drop("Unnamed: 0",axis = 1)
print(df.head())


# # Gender distribution

# In[37]:


plt.figure(figsize=(5,5))
x=sns.countplot(data = df, x="Gender")
x.bar_label(x.containers[0])
plt.title("Gender distribution")
plt.show()


# In[18]:


# from the above chart we analysed that :
# female are more as compared to male


# # Examining the Relationship Between Parental Education Level and Student Academic Performance

# In[32]:


gb = df.groupby("ParentEduc").agg({"MathScore":"mean","ReadingScore":"mean","WritingScore":"mean"})
print(gb)


# In[39]:


plt.figure(figsize=(5,5))
sns.heatmap(gb , annot = True)
plt.title("Relationship between Parent's Education and Student's Score")
plt.show()


# In[29]:


# from the above chart we concluded that the education of the parents have a good impact on their child score


# # Examining the Relationship Between Parental Marital Status and Student Academic Performance

# In[35]:


gm = df.groupby("ParentMaritalStatus").agg({"MathScore":"mean","ReadingScore":"mean","WritingScore":"mean"})
print(gm)


# In[40]:


plt.figure(figsize=(5,5))
sns.heatmap(gm , annot = True)
plt.title("Relationship between Parent's Marital Status and Student's Score")
plt.show()


# In[ ]:


# from the above chart we concluded that the ParentMaritalStatus of the parents have a no impact on their child score


# # outliers based on subject

# In[41]:


sns.boxplot(data=df,x="MathScore")
plt.show()


# In[42]:


sns.boxplot(data=df,x="ReadingScore")
plt.show()


# In[43]:


sns.boxplot(data=df,x="WritingScore")
plt.show()


# In[44]:


# so above chart  we have concluded that the maths is difficult subject campared to Reading and Writing


# In[45]:


print(df["EthnicGroup"].unique())


# # Distribution of Ethnic Group

# In[47]:


groupA = df.loc[(df["EthnicGroup"] == "group A")].count()
print(groupA)


# # Count of Students vs. Weekly Study Hours

# In[60]:


# Count plot for Weekly Study Hours
plt.figure(figsize=(10, 6))
sns.countplot(x='WklyStudyHours', data=df)
plt.title('Count of Students vs. Weekly Study Hours')
plt.xlabel('Weekly Study Hours')
plt.ylabel('Count of Students')
plt.show()


# In[61]:


# conclusion - The majority of students prefer studying in the 5-10 hour range per week


# # Impact of Birth Order on Academic Performance

# In[62]:


# Group the data by 'IsFirstChild' and calculate the average scores
first_child_group = df.groupby('IsFirstChild')[['MathScore', 'ReadingScore', 'WritingScore']].mean()

# Plotting the results
first_child_group.plot(kind='bar', figsize=(10, 6))
plt.title('Average Scores Based on Whether Student is First Child')
plt.ylabel('Average Score')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


# conclusion - There's no significant difference between first and non-first children in academic performance, but students generally score higher in reading than in writing and math."


# In[ ]:


# Based on the analysis of the dataset, strategies to improve student scores include enhancing parental engagement, providing targeted support for math teaching effective study habits, offering diverse reading materials, encouraging peer collaboration, and adapting interventions based on performance.

