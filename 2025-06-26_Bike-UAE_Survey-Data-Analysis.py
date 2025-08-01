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

####################################################################

### Basic Data Visualization ###

### Gender Distribution ###
plt.figure(figsize=(6, 6))
gender_counts = df['Gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightpink', 'lightgray'])
plt.title('Gender Distribution')
plt.axis('equal')
plt.show()



### Salary Distribution ###
salary_order = [
    'Less than 5k',
    '5-10k',
    '10-15k',
    '15-20k',
    '20-25k',
    '25-30k',
    'More than 30k',
    'Prefer not to share'
]

# Get counts and reindex to match custom order
salary_counts = df['Salary'].value_counts().reindex(salary_order)

plt.figure(figsize=(5, 12))
salary_counts.plot(kind='bar', color='mediumpurple', edgecolor='black')
plt.title('Salary Distribution')
plt.xticks(rotation=45)
plt.show()



### Age Distribution ###
age_order = [
    'Under 18',
    '18-24 years old',
    '25-34 years old',
    '35-44 years old',
    '45-54 years old',
    '55-64 years old',
    '65+ years old'
]

# Get counts and reindex to custom order
age_counts = df['Age'].value_counts().reindex(age_order)

plt.figure(figsize=(5, 11))
age_counts.plot(kind='bar', color='teal', edgecolor='black')
plt.title('Age Distribution')
plt.xticks(rotation=45)
plt.show()



'''
### Duration Distribution ###
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
'''



### Active Rider Months ###

month_cols = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

# Sum the True values for each month
month_counts = df[month_cols].sum().sort_index(key=lambda x: pd.to_datetime(x, format='%B'))

plt.figure(figsize=(10, 5))
month_counts.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Number of Riders Who Ride in Each Month')
plt.xlabel('Month')
plt.ylabel('Number of Riders')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

####################################################################

### Gender vs Safety ###

print("\n--- Gender vs Safety ---\n")

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



### Emirate vs Safety ###

print("\n--- Emirate vs Safety ---\n")

# Filter rows where both Emirate and SafetyScore are present
df_emirate = df[df['Emirate'].notna() & df['Emirate-Safety'].notna()].copy()
df_emirate['SafetyScore'] = df_emirate['Emirate-Safety'].map(safety_mapping)

# Sort emirates by average safety score for nicer plot order
ordered_emirates = df_emirate.groupby('Emirate')['SafetyScore'].mean().sort_values().index
df_emirate_sorted = df_emirate[df_emirate['Emirate'].isin(ordered_emirates)]

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

# Basic stats: mean safety score by emirate
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


### ANOVA TESTS ###
from patsy import dmatrix
print("\n--- ANOVA Tests ---\n")

df_anova = df.copy()
df_anova['SafetyScore'] = df_anova['Emirate-Safety'].map(safety_mapping)


# Define the features to test (now includes TranspoType-HowOften)
features = ['Gender', 'Age', 'Salary', 'Emirate', 'TranspoType-HowOften', 'YearsRiding']
anova_results = {}

# Loop through features and run one-way ANOVA for each
for feature in features:
    df_test = df_anova[df_anova['SafetyScore'].notna() & df_anova[feature].notna()].copy()
    df_test[feature] = df_test[feature].astype('category')

    # Use Q() for safe column referencing
    formula = f'SafetyScore ~ C(Q("{feature}"))'
    
    model = ols(formula, data=df_test).fit()
    anova_table = anova_lm(model, typ=2)

    anova_results[feature] = anova_table

# Display results
for feature, table in anova_results.items():
    print(f"\n### One-Way ANOVA: {feature} ###")
    print(table)

################################################################

### Plot Years Riding vs Safety Score ###

# Filter for valid YearsRiding and SafetyScore entries
df_years = df[
    (df['YearsRiding'].notna()) &
    (df['SafetyScore'].notna())
].copy()

# Optional: define a custom order for years riding if needed (e.g., sorted by category labels)
# If YearsRiding is numeric, sort automatically
years_order = sorted(df_years['YearsRiding'].unique())

# Define custom order for salary
years_order = [
    'I am a new Rider',
    '1-5 years',
    '5-10 years',
    '10-15 years',
    '15-20 years',
    'More than 20 years',
    'Since childhood'
]

# Calculate standard deviation and count per group
group_stats = df_years.groupby('YearsRiding')['SafetyScore'].agg(['mean', 'std', 'count']).reindex(years_order)

# Calculate standard error
group_stats['se'] = group_stats['std'] / group_stats['count']**0.5

# Plot with error bars (standard error)
plt.figure(figsize=(10, 6))
plt.errorbar(
    x=group_stats.index,
    y=group_stats['mean'],
    yerr=group_stats['se'],
    fmt='o',
    color='darkorange',
    ecolor='gray',
    capsize=5
)
plt.title('Mean Safety Score by Years Riding with Standard Error')
plt.xlabel('Years Riding')
plt.ylabel('Mean Safety Score (1 = Safe, 5 = Not Safe)')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

####################################################################

### Check influence of Gender on Owning a Bike ###

print("\n--- Influence of Gender on Owning a Bike ---\n")

table = pd.crosstab(df['Gender'], df['OwnBike'])
print(table)
chi2, p, dof, expected = chi2_contingency(table)

print()
print("Degrees of Freedom:", dof)
print()
print("Chi-Square Statistic:", chi2)
print("p-vlue:", p)

####################################################################

### Plot Salary vs Safety Score ###

# Filter for valid salary and safety score entries
df_salary = df[
    (df['Salary'].notna()) &
    (df['Salary'] != 'Prefer not to share') &
    (df['SafetyScore'].notna())
].copy()

# Define custom order for salary
salary_order = [
    'Less than 5k',
    '5-10k',
    '10-15k',
    '15-20k',
    '20-25k',
    '25-30k',
    'More than 30k'
]

# Calculate standard deviation and count per salary group
group_stats = df_salary.groupby('Salary')['SafetyScore'].agg(['mean', 'std', 'count']).reindex(salary_order)

# Calculate standard error
group_stats['se'] = group_stats['std'] / group_stats['count']**0.5

# Plot with error bars (standard error)
plt.figure(figsize=(10, 6))
plt.errorbar(
    x=group_stats.index,
    y=group_stats['mean'],
    yerr=group_stats['se'],
    fmt='o',
    color='steelblue',
    ecolor='gray',
    capsize=5
)
plt.title('Mean Safety Score by Salary with Standard Error')
plt.xlabel('Monthly Salary Range (AED)')
plt.ylabel('Mean Safety Score (1 = Safe, 5 = Not Safe)')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

####################################################################

### Random Forest ###

print("\n--- Random Forest ---\n")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from sklearn.utils import resample

# Select bike type columns
bike_type_cols = [col for col in df.columns if col.startswith('Type-')]

# Drop rows with missing Gender or bike type values
df_rf = df[['Gender'] + bike_type_cols].dropna()

# Encode Gender labels (Male = 1, Female = 0)
le = LabelEncoder()
df_rf['Gender_encoded'] = le.fit_transform(df_rf['Gender'])

# Define features and target
X = df_rf[bike_type_cols]
y = df_rf['Gender_encoded']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)


# Filter out "Prefer not to say"
df_rf = df_rf[df_rf['Gender'].isin(['Male', 'Female'])]

# Resample to balance the classes
df_majority = df_rf[df_rf['Gender'] == 'Male']
df_minority = df_rf[df_rf['Gender'] == 'Female']
df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Redefine X and y
X_balanced = df_balanced[bike_type_cols]
y_balanced = df_balanced['Gender_encoded']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, stratify=y_balanced, random_state=42, test_size=0.2)

# Retrain model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluate
report = classification_report(y_test, y_pred, target_names=le.classes_[:2])
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot feature importances
importances = pd.Series(rf.feature_importances_, index=bike_type_cols).sort_values(ascending=True)

plt.figure(figsize=(8, 6))
importances.plot(kind='barh', color='teal')
plt.title("Feature Importance: Predicting Gender from Bike Types")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

print("Random Forest Report:\n")
print(report,"\n")

from scipy.stats import pointbiserialr

# Ensure 'Gender' is binary: 1 = Male, 0 = Female
df_gender_corr = df[df['Gender'].isin(['Male', 'Female'])].copy()
df_gender_corr['Gender_encoded'] = LabelEncoder().fit_transform(df_gender_corr['Gender'])  # Male=1, Female=0

# Drop rows with missing RoadBike values
df_gender_corr = df_gender_corr[df_gender_corr['Type-RoadBike'].notna()]

# Compute point-biserial correlation (binary vs continuous/binary)
corr_coef, p_value = pointbiserialr(df_gender_corr['Gender_encoded'], df_gender_corr['Type-RoadBike'])

print("Correlation between riding a roadbike and being a man:")
print(corr_coef, "with a p-value of:",p_value,"\n")

####################################################################

print("\n--- Challenging Norms vs Gender ---\n")

# Filter and prepare data
df_chal = df[['Chal-CulturalNorms', 'Gender']].dropna()

# Convert variables to categorical
df_chal['Gender'] = df_chal['Gender'].astype('category')
df_chal['Chal-CulturalNorms'] = df_chal['Chal-CulturalNorms'].astype('float')

# Fit ANOVA model: test if mean CulturalNorms challenge scores differ by gender
model = ols('Q("Chal-CulturalNorms") ~ C(Gender)', data=df_chal).fit()
anova_result = anova_lm(model, typ=2)

print("Challenging Norms Anova:")
print(anova_result)

