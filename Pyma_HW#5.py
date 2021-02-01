#!/usr/bin/env python
# coding: utf-8

# # Pymaceuticals Inc.
# ---
# 
# ### Observations and Insights
# - *Your observations and insights here* ...
# 

# In[1]:


# Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import numpy as np
from scipy.stats import linregress
from scipy.stats import sem

# Study data files
mouse_metadata_path = "data/Mouse_metadata.csv"
study_results_path = "data/Study_results.csv"

# Read the mouse data and the study results
mouse_metadata = pd.read_csv(mouse_metadata_path)
study_results = pd.read_csv(study_results_path)

# Combine the data into a single dataset
combined_df=pd.merge(mouse_metadata,study_results,on="Mouse ID")

# Display the data table for preview
combined_df.head()


# In[2]:


# Checking the number of mice.
number_mice=combined_df["Mouse ID"].value_counts()
number_mice


# In[3]:


# Getting the duplicate mice by ID number that shows up for Mouse ID and Timepoint. 
#To find duplicates on specific columns use subset
duplicate_mice=combined_df.loc[combined_df.duplicated(subset=['Mouse ID','Timepoint']),'Mouse ID'].unique()
duplicate_mice


# In[4]:


clean_df=combined_df.drop_duplicates(subset=['Mouse ID'],keep=False)
clean_df.head()


# In[5]:


#Get all the data for the duplicate mouse ID
combined_df.loc[combined_df['Mouse ID']=="g989"]


# In[6]:


#Create a clean DataFrame by dropping the duplicate mouse by it's ID
combined_df_clean=combined_df[combined_df["Mouse ID"].isin(duplicate_mice)==False]
combined_df_clean

#Checking the number of mice in the clean DataFrame
clean_mice=len(combined_df_clean["Mouse ID"].unique())
clean_mice


# ## Summary Statistics

# In[7]:


# Generate a summary statistics table of mean, median, variance, standard deviation, and SEM of the tumor volume for each regimen
# Use groupby and summary statistical methods to calculate the following properties of each drug regimen: 
# mean, median, variance, standard deviation, and SEM of the tumor volume. 
# Assemble the resulting series into a single summary dataframe.

tumor_volume_mean=combined_df_clean.groupby('Drug Regimen').mean()["Tumor Volume (mm3)"]
tumor_volume_median=combined_df_clean.groupby('Drug Regimen').median()["Tumor Volume (mm3)"]
tumor_volume_variance=combined_df_clean.groupby('Drug Regimen').var()["Tumor Volume (mm3)"]
tumor_volume_std=combined_df_clean.groupby('Drug Regimen').std()["Tumor Volume (mm3)"]
tumor_volume_sem=combined_df_clean.groupby('Drug Regimen').sem()["Tumor Volume (mm3)"]

summary_df= pd.DataFrame({'Mean Tumor Volume':tumor_volume_mean,
                          'Median Tumor Volume':tumor_volume_median,
                          'Tumor Volume Variance':tumor_volume_variance,
                          'Tumor Volume Ste. Dev.':tumor_volume_std,
                          'Tumor Volume Std. Err.':tumor_volume_sem})
summary_df


# In[8]:


# Generate a summary statistics table of mean, median, variance, standard deviation, and SEM of the tumor volume for each regimen

# Using the aggregation method, produce the same summary statistics in a single line
stats_summary=combined_df_clean.groupby('Drug Regimen')
summary_df_2=stats_summary.agg(['mean','median','var','std','sem'])["Tumor Volume (mm3)"]
summary_df_2


# ## Bar and Pie Charts

# In[9]:


# Generate a bar plot showing the total number of measurements taken on each drug regimen using pandas.
number_mice=combined_df_clean.groupby(['Drug Regimen']).count()['Mouse ID']

#Plot using Pandas
plot_pandas=number_mice.plot.bar(figsize=(4,3), color='b',fontsize = 7)

# Give our graph axis labels
plt.xlabel("Drug Regimen")
plt.ylabel("Number of Unique Mice Tested")

# Saves an image of our chart so that we can view it in a folder
plt.savefig("../Images/Pandas_Plot.png")
plt.show()


# In[10]:


# Generate a bar plot showing the total number of measurements taken on each drug regimen using using pyplot.

#X axis length 
x_axis = np.arange(len(number_mice))

#Plot using Pyplot
plt.bar(x_axis, number_mice, color='b', alpha=0.5, align="center")

#Align tick locations and x axis headers
tick_locations = [value for value in x_axis]
plt.xticks(tick_locations,['Capomulin', 'Ramicane','Ketapril','Naftisol','Zoniferol','Placebo','Stelasyn', 'Infubinol','Ceftamin', 'Propriva'], rotation='vertical')

# Sets the x limits of the current chart
plt.xlim(-0.75, len(x_axis)-0.25)

# Sets the y limits of the current chart
plt.ylim(0, max(number_mice)+20)

# Label the chart  
plt.xlabel("Drug Regimen")
plt.ylabel("Number of Unique Mice Tested")

# Saves an image of our chart so that we can view it in a folder
plt.savefig("../Images/Pyplot.png")
plt.show()


# In[11]:


# Generate a pie plot showing the distribution of female versus male mice using pandas

#Data Distribution by sex
sex_data=combined_df_clean['Sex'].value_counts()

#Plot the chart (Pandas)
pie_plot=sex_data.plot.pie(autopct="%1.1f%%", startangle=90, title='Distribution by Sex')
pie_plot


# In[12]:


# Generate a pie plot showing the distribution of female versus male mice using plyplot

#Data Distribution by sex
sex_data=combined_df_clean['Sex'].value_counts()

# Automatically finds the percentages of each part of the pie chart
plt.pie(sex_data,labels=sex_data.index.values,autopct="%1.1f%%",startangle=140)

#Assign a Title and Display Chart 
plt.title('Distribution by Sex')
plt.show


# ## Quartiles, Outliers and Boxplots

# In[13]:


# Calculate the final tumor volume of each mouse across four of the treatment regimens:Capomulin, Ramicane, Infubinol, and Ceftamin
# Start by getting the last (greatest) timepoint for each mouse
greatest_tp=pd.DataFrame(combined_df_clean.groupby('Mouse ID')['Timepoint'].max().sort_values()).reset_index().rename(columns=
     {'Timepoint':'Greatest Timepoint'})
greatest_tp

# Merge this group df with the original dataframe to get the tumor volume at the last timepoint
new_greatest_tp=pd.merge(combined_df_clean,greatest_tp,on='Mouse ID')
new_greatest_tp.head()


# In[14]:


# Put treatments into a list for for loop (and later for plot labels)
treatment_list = ["Capomulin", "Ramicane", "Infubinol", "Ceftamin"]

# Create empty list to fill with tumor vol data (for plotting)
tumor_vol_list = []

# Calculate the IQR and quantitatively determine if there are any potential outliers. 
for drug in treatment_list:
    drug_df=new_greatest_tp.loc[new_greatest_tp['Drug Regimen']==drug]
    
    # Create subset 
    vol_data=drug_df.loc[drug_df['Timepoint']==drug_df['Greatest Timepoint']]
    
    #Store Final Volume Values
    final_vol=vol_data['Tumor Volume (mm3)']
    tumor_vol_list.append(final_vol)
    
    #Pandas to give quartile calculations
    #To get the quartiles we can pass in .25, .5, .75
    quartiles = final_vol.quantile([.25,.5,.75])
    lowerq = quartiles[0.25]
    upperq = quartiles[0.75]
    iqr = upperq-lowerq
    print(f'IQR for {drug}: {iqr}')
    
    # Determine outliers using upper and lower bounds
    lower_bound = lowerq - (1.5*iqr)
    upper_bound = upperq + (1.5*iqr)
    print(f"Values below {lower_bound} could be outliers.")
    print(f"Values above {upper_bound} could be outliers.")
    


# In[15]:


# Generate a box plot of the final tumor volume of each mouse across four regimens of interest

#Set Markers
markers=dict(marker='o', markerfacecolor='b', markersize=10, markeredgecolor='black')

#Plot Boxplot
plt.boxplot(tumor_vol_list,flierprops=markers)

plt.title('Final Tumor Volume Across Four Regimens')
plt.ylabel('Final Tumor Volume(mm3)')
plt.xticks([1, 2, 3, 4], ['Capomulin', 'Ramicane', 'Infubinol', 'Ceftamin'])
plt.show()


# ## Line and Scatter Plots

# In[16]:


# Generate a line plot of tumor volume vs. time point for a mouse treated with Capomulin
# Use .loc to filter data to find mouse trated with Capomulin
capomulin_df=combined_df_clean.loc[combined_df_clean['Drug Regimen']=='Capomulin']
capomulin_df

#Selecting a mouse that has been treated with capomulin; mouse s185
test_mouse=capomulin_df.loc[capomulin_df['Mouse ID']=='S185']

#Line plot of tumor volume vs. time point 
plt.plot(test_mouse['Timepoint'],test_mouse['Tumor Volume (mm3)'])
plt.title('Capomulin for Mouse S185')
plt.xlabel('Time Point')
plt.ylabel('Tumor Volume')
plt.show


# In[17]:


# Generate a scatter plot of average tumor volume vs. mouse weight for the Capomulin regimen
# Use .loc to filter data to find mouse trated with Capomulin
capomulin_df=combined_df_clean.loc[combined_df_clean['Drug Regimen']=='Capomulin']

#Find average (mean) for tumor volume
average_vol=pd.DataFrame(capomulin_df.groupby('Mouse ID')['Tumor Volume (mm3)'].mean().sort_values()).reset_index().rename(columns=
    {'Tumor Volume (mm3)':'Average Tumor Volume'})
                                            
#Merge & Drop Duplicate                                                                                                                            
avergae_vol=pd.merge(capomulin_df,average_vol,on='Mouse ID') 
final_ave_df=average_vol[['Weight (g)','Average Tumor Volume']].drop_duplicates()
final_ave_df
                     
#Create x and y 
x=final_ave_df['Weight (g)']
y=final_ave_df['Average Tumor Volume']
                     
#Create Scatter Plot
plt.scatter(x,y)
plt.xlabel('Weight (g)')
plt.ylabel('Average Tumor Volume (mm3)')
plt.show



# ## Correlation and Regression

# In[ ]:


# Calculate the correlation coefficient and linear regression model 
# for mouse weight and average tumor volume for the Capomulin regimen
# Compute the Pearson correlation coefficient mouse weights and average tumor volume 

correlation = st.pearsonr(x,y)
print(f"The correlation between both factors is {round(correlation[0],2)}")

# Add the linear regression equation 
(slope, intercept, rvalue, pvalue, stderr) = linregress(x,y)
regress_values = x * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.scatter(x,y)
plt.plot(x,regress_values,"r-")
plt.annotate(line_eq,(6,10),fontsize=15,color="red")
plt.xlabel('Weight (g)')
plt.ylabel('Average Tumor Volume (mm3)')
plt.show()

