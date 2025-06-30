### 2025-06-26_Bike-UAE_Survey-Data-Analysis.py
### Max Moran

# Import libraries
import os

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
from sklearn.preprocessing import MultiLabelBinarizer

import re

# Set Working Directory to folder with data
path = r"C:\Users\might\Documents\UTSA\Summer_2025\STA-6943_Internship\bike-UAE" #r in front is for the back slashes
os.chdir(path)

# Import data
df = pd.read_excel('Bike-UAE_Basic-Cleaned.xlsx')
print(df.head(),"\n")

### Basic Visualization
# Gender Distribution Plot
plt.figure(figsize=(6, 6))
gender_counts = df['Gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Gender Distribution')
plt.axis('equal')
plt.show()

# Salary Distribution Plot
plt.figure(figsize=(5, 12))
salary_counts = df['Salary'].value_counts().sort_index()
salary_counts.plot(kind='bar')
plt.title('Salary Distribution')
plt.xticks(rotation=45)
plt.show()

# Age Distribution Plot
plt.figure(figsize=(5, 11))
age_counts = df['Age'].value_counts().sort_index()
age_counts.plot(kind='bar')
plt.title('Age Distribution')
plt.xticks(rotation=45)
plt.show()


### Duration Distribution
# Examine initial boxplot
sns.boxplot(x=df['Duration'])
plt.title('Initial Survey Duration Box Plot')
plt.xlabel('Duration (minutes)')
plt.show()

# Find Quantiles
Q1 = df['Duration'].quantile(0.25)
Q3 = df['Duration'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the DataFrame
dfDurationIQR = df[(df['Duration'] >= lower_bound) & (df['Duration'] <= upper_bound)]
print("Temporarily removed",len(df['Duration'])-len(dfDurationIQR['Duration']),"outliers")

# Examine updated boxplot
sns.boxplot(x=dfDurationIQR['Duration'])
plt.title('Narrowed Survey Duration Box Plot')
plt.xlabel('Duration (minutes)')
plt.show()


dfDurationIQR["Duration"].hist(bins=100)
plt.xlabel('Duration (minutes)')
plt.ylabel('Number of Respondents')
plt.title('Survey Completion Time Distribution')
plt.show()

# Check influence of Gender on Owning a Bike
table = pd.crosstab(df['Gender'], df['OwnBike'])
print(table)
chi2, p, dof, expected = chi2_contingency(table)

print()
print("Degrees of Freedom:", dof)
print()
print("Chi-Square Statistic:", chi2)
print("p-value:", p)


