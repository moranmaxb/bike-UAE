### 2025-06-16_Bike-UAE_Survey-Data
### Max Moran

# Import libraries
import os

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Set Working Directory to folder with data
path = r"C:\Users\might\Documents\UTSA\Summer_2025\STA-6943_Internship\bike-UAE" #r in front is for the back slashes
os.chdir(path)

# Import data
df = pd.read_csv('LABELS-UAE Bicycle & Scooter Survey - CITIES_June 16, 2025_16.12.csv')
print(df.head(),"\n")

# Drop unnecessary rows
df = df.drop([0,1])
print("Dataframe after dropping unecessary context rows:")
print(df.head(),"\n")

# List of new column names
new_column_names = [# Automatically generated survey data
                    'StartDate', 'EndDate', 'Status', 'Progress',
                    'Duration', 'Finished', 'RecordedDate', 'ResponseID',
                    'DistributionChannel',
                    # Starting attributes
                    'UserLang', 'Introduction', 'Emirate',
                    # Transporation by type
                    'TranspoType',
                    'Bike-Why', 'Bike-Why-Other', 'Bike-HowOften', 'Bike-Months',
                    'Ebike-Why', 'Ebike-Why-Other', 'Ebike-HowOften', 'Ebike-Months',
                    'Scooter-Why', 'Scooter-Why-Other', 'Scooter-HowOften', 'Scooter-Months',
                    'Skateboard-Why', 'Skateboard-Why-Other', 'Skateboard-HowOften', 'Skateboard-Months',
                    # Riding attributes
                    'WhoRideWith', 'RiderType', 'RiderType-Other',
                    'Bikeshare', 'TrackRides',
                    'Helmet', 'Earbuds', 'Phones',
                    # Lifestyle
                    'OwnBike', 'BikeType', 'BikeType-Other',
                    'LifestyleInfluence', 'LifestyleInfluence-Other',
                    'CycleShopsInfluence', 'CycleShopsInfluence-Other',
                    'CycleShopsBenefit', 'CycleShopsBenefit-Other',
                    # Safety by Location
                    'AbuDhabi-Safety', 'AbuDhabi-Location', 'AbuDhabi-Location-Other',
                    'AlAin-Safety', 'AlAin-Location', 'AlAin-Location-Other',
                    'AlDhafra-Safety', 'AlDhafra-Location',
                    'Dubai-Safety', 'Dubai-Location', 'Dubai-Location-Other',
                    'Sharjah-Safety', 'Sharjah-Location', 'Sharjah-Location-Other',
                    'Ajman-Safety', 'Ajman-Location',
                    'UmmAlQuwain-Safety', 'UmmAlQuwain-Location',
                    'Fujairah-Safety', 'Fujairah-Location',
                    'RasAlKhaimah-Safety', 'RasAlKhaimah-Location', 'RasAlKhaimah-Location-Other',
                    # Other transportation questions
                    'ComfortLocaiton', 'Challenges', 'Challenges-Other',
                    'ExperienceImprovement', 'ExperienceImprovement-Other', 'AltMicromobility',
                    'KidsCycle', 'KidsCycle-No',
                    'RegularCommute', 'NoRideCommute', 'NoRideCommute-Other',
                    # Individual details
                    'Gender', 'Age', 'Country',
                    'Education', 'Education-Other',
                    'EmploymentStatus', 'Salary',
                    'DurationInUAE', 'YearsRiding',
                    # Enjoyment/Initiatives by Location
                    'AbuDhabi-WhyLikeRiding', 'AbuDhabi-Initiatives',
                    'AlAin-WhyLikeRiding', 'AlAin-Initiatives',
                    'AlDhafra-WhyLikeRiding', 'AlDhafra-Initiatives',
                    'Dubai-WhyLikeRiding', 'Dubai-Initiatives',
                    'Sharjah-WhyLikeRiding', 'Sharjah-Initiatives',
                    'Ajman-WhyLikeRiding', 'Ajman-Initiatives',
                    'UmmAlQuwain-WhyLikeRiding', 'UmmAlQuwain-Initiatives',
                    'Fujairah-WhyLikeRiding', 'Fujairah-Initiatives',
                    'RasAlKhaimah-WhyLikeRiding', 'RasAlKhaimah-Initiatives',
                    # Survey completion
                    'Giveaway']

# Assign the list to the DataFrame
df.columns = new_column_names
print("Dataframe with better column names:")
print(df,"\n")

# Convert quant variables to int or float
df['Duration'] = df['Duration'].astype(int)
df['Progress'] = df['Progress'].astype(int)
df['DurationInUAE'] = df['DurationInUAE'].replace({
    'Less than 1': 0.5,
    '40+': 40
}).astype(float)



### Handle spam responses
# 38 questions (minQuestions) for 100% completion, assume ~5 seconds a question (minSecQ)
# So, any entries with a completion % that was submitted to fast will be dropped (likely spammed through questions)
preSpamFilterLen = len(df)
minQuestions = 38 # this is the minimum number of questions to complete 100% of the survey
minSecQ = 5 # set this variable to how many seconds it takes on average per question
minTotTime = minQuestions * minSecQ

df = df[~(df['Duration'] < (minTotTime*df['Progress'])/100)]
print("Dataframe without spam responses:")
print(df,"\n")
print("Removed",preSpamFilterLen-len(df),"spam responses.\n")

# Convert duration to minutes
df['Duration'] = df['Duration'] / 60



### Clean Data
empty_cols = df.columns[df.isna().all()]
print("Empty columns:\n")
print(empty_cols,"\n")
df = df.dropna(axis=1, how='all') # drops empty columns
df = df.drop(columns=['StartDate', 'EndDate', 'ResponseID', 'Status', 'RecordedDate', 'DistributionChannel', 'Introduction', 'Giveaway']) # drops useless columns
print("Dataframe without empty or unhelpful columns:")
print(df,"\n")




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
'''

#df.to_excel('Bike-UAE_Basic-Cleaned.xlsx', index=False)

from scipy.stats import chi2_contingency

# Step 1: Create a contingency table
table = pd.crosstab(df['Gender'], df['OwnBike'])
print(table)

# Step 2: Run the Chi-Square test
chi2, p, dof, expected = chi2_contingency(table)

# Step 3: View results
print("Chi-Square Statistic:", chi2)
print("Degrees of Freedom:", dof)
print("p-value:", p)