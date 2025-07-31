### 2025-06-26_Bike-UAE_Survey-Data-Analysis.py
### Max Moran

# Import libraries
import os

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
from scipy import stats
from sklearn.preprocessing import MultiLabelBinarizer

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

import re

# Set Working Directory to folder with data
path = r"C:\Users\might\Documents\UTSA\Summer_2025\STA-6943_Internship\bike-UAE" #r in front is for the back slashes
os.chdir(path)

# Import data
df = pd.read_excel('Bike-UAE_Basic-Cleaned.xlsx')
#print(df.head(),"\n")

'''

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
print("p-vlue:", p)

'''

####################################################################

### Gender vs Safety ###

# Let's start by checking unique values for the relevant columns
gender_values = df['Gender'].unique()
safety_values = df['Emirate-Safety'].unique()

# Map Emirate-Safety to numeric scale
safety_mapping = {
    '1 - very safe, no concerns': 1,
    '2': 2,
    '3 - moderately safe, some concerns': 3,
    '4': 4,
    '5 - not safe at all, many concerns': 5
}
df['SafetyScore'] = df['Emirate-Safety'].map(safety_mapping)


# Count 'Prefer not to say' and NaN values in the Gender column
prefer_not_to_say_count = (df['Gender'] == 'Prefer not to say').sum()
na_count = df['Gender'].isna().sum()

print()
print("prefer_not_to_say_count:", prefer_not_to_say_count)
print("na_count:", na_count)
print()


# Drop rows with missing values in Gender or SafetyScore
df_clean = df.dropna(subset=['Gender', 'SafetyScore'])
df_clean = df_clean[df_clean['Gender'] != 'Prefer not to say']


# Basic stats: mean safety score by gender
mean_safety_by_gender = df_clean.groupby('Gender')['SafetyScore'].mean()

# Boxplot of Safety Score by Gender
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df_clean,
    x='Gender',
    y='SafetyScore',
    palette={
        'Male': 'skyblue',
        'Female': 'lightpink'
    }
)
plt.title("Perceived Safety Score by Gender")
plt.ylabel("Safety Score (1 = Safe, 5 = Not Safe)")
plt.xlabel("Gender")
plt.tight_layout()
plt.show()

# Create percentage table by dividing counts by the total number in each gender group
percentage_table = (
    df_clean.groupby(['SafetyScore', 'Gender'])
    .size()
    .unstack()
    .apply(lambda col: col / col.sum() * 100, axis=0)
)

# Plotting the percentage table
percentage_table.plot(kind='bar', stacked=False, color=['lightpink', 'skyblue'], edgecolor='black')
plt.title("Percentage Distribution of Safety Scores by Gender")
plt.xlabel("Safety Score (1 = Safe, 5 = Not Safe)")
plt.ylabel("Percentage (%)")
plt.legend(title="Gender")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
print()

# One-way ANOVA to test for significant differences in safety score across genders
anova_result = stats.f_oneway(
    df_clean[df_clean['Gender'] == 'Male']['SafetyScore'],
    df_clean[df_clean['Gender'] == 'Female']['SafetyScore']
)

print("mean_safety_by_gender:\n", mean_safety_by_gender)
print()
print("gender_anova_result:", anova_result.pvalue)
print()

####################################################################

### Emirate Selection vs Safety ###

# Filter rows where both Emirate and SafetyScore are present
df_emirate = df[df['Emirate'].notna() & df['Emirate-Safety'].notna()].copy()
df_emirate['SafetyScore'] = df_emirate['Emirate-Safety'].map(safety_mapping)

# Sort emirates by average safety score for nicer plot order
ordered_emirates = df_emirate.groupby('Emirate')['SafetyScore'].mean().sort_values().index

# Boxplot
plt.figure(figsize=(12, 6))
df_emirate_sorted = df_emirate[df_emirate['Emirate'].isin(ordered_emirates)]
sns.boxplot(data=df_emirate_sorted, x='Emirate', y='SafetyScore', order=ordered_emirates)
plt.title("Boxplot of Safety Scores by Emirate")
plt.ylabel("Safety Score (1 = Safe, 5 = Not Safe)")
plt.xlabel("Emirate")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Percentage distribution
percent_table = (
    df_emirate.groupby(['SafetyScore', 'Emirate'])
    .size()
    .unstack()
    .apply(lambda col: col / col.sum() * 100, axis=0)
    .T.loc[ordered_emirates]
)

# Plot percentage distribution
percent_table.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis', edgecolor='black')
plt.title("Percentage Distribution of Safety Scores by Emirate")
plt.ylabel("Percentage (%)")
plt.xlabel("Emirate")
plt.xticks(rotation=45, ha='right')
plt.legend(title='Safety Score', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Basic stats: mean safety score by gender
mean_safety_by_emirate = df_clean.groupby('Emirate')['SafetyScore'].mean()

print("mean_safety_by_emirate:\n", mean_safety_by_emirate)
print()

# Clean up any extra whitespace
df_emirate_sorted['Emirate'] = df_emirate_sorted['Emirate'].str.strip()

emirate_counts = df_emirate_sorted['Emirate'].value_counts()

print("emirate counts:\n", emirate_counts)

# Get list of emirates and their valid counts
emirate_groups = [
    df_emirate_sorted[df_emirate_sorted['Emirate'] == emirate]['SafetyScore'].dropna()
    for emirate in df_emirate_sorted['Emirate'].unique()
]

# Filter out any empty groups
emirate_groups = [group for group in emirate_groups if len(group) > 1]

# Run ANOVA
anova_result = stats.f_oneway(*emirate_groups)

print("emirate_anova_result:", anova_result.pvalue)
print()

####################################################################

# Filter relevant rows
df_anova = df[
    df['Gender'].isin(['Male', 'Female']) &
    df['Age'].notna() &
    df['Emirate-Safety'].notna()
].copy()

df_anova['SafetyScore'] = df_anova['Emirate-Safety'].map(safety_mapping)

# Convert to categorical
df_anova['Gender'] = df_anova['Gender'].astype('category')
df_anova['Age'] = df_anova['Age'].astype('category')

# Fit the two-way ANOVA model
model = ols('SafetyScore ~ C(Gender) + C(Age) + C(Gender):C(Age)', data=df_anova).fit()
anova_table = anova_lm(model, typ=2)

print(anova_table)