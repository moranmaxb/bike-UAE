### 2025-06-16_Bike-UAE_Survey-Data
### Max Moran

# Import libraries
import os

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
from sklearn.preprocessing import MultiLabelBinarizer

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
                    'Bike-Why', 'Bike-Why-Other-Text', 'Bike-HowOften', 'Bike-Months',
                    'Ebike-Why', 'Ebike-Why-Other-Text', 'Ebike-HowOften', 'Ebike-Months',
                    'Scooter-Why', 'Scooter-Why-Other-Text', 'Scooter-HowOften', 'Scooter-Months',
                    'Skateboard-Why', 'Skateboard-Why-Other-Text', 'Skateboard-HowOften', 'Skateboard-Months',
                    # Riding attributes
                    'WhoRideWith', 'RiderType', 'RiderType-Other-Text',
                    'Bikeshare', 'TrackRides',
                    'Helmet', 'Earbuds', 'Phones',
                    # Lifestyle
                    'OwnBike', 'BikeType', 'BikeType-Other-Text',
                    'LifestyleInfluence', 'LifestyleInfluence-Other-Text',
                    'CycleShopsInfluence', 'CycleShopsInfluence-Other-Text',
                    'CycleShopsBenefit', 'CycleShopsBenefit-Other-Text',
                    # Safety by Location
                    'AbuDhabi-Safety', 'AbuDhabi-Location', 'AbuDhabi-Location-Other-Text',
                    'AlAin-Safety', 'AlAin-Location', 'AlAin-Location-Other-Text',
                    'AlDhafra-Safety', 'AlDhafra-Location',
                    'Dubai-Safety', 'Dubai-Location', 'Dubai-Location-Other-Text',
                    'Sharjah-Safety', 'Sharjah-Location', 'Sharjah-Location-Other-Text',
                    'Ajman-Safety', 'Ajman-Location',
                    'UmmAlQuwain-Safety', 'UmmAlQuwain-Location',
                    'Fujairah-Safety', 'Fujairah-Location',
                    'RasAlKhaimah-Safety', 'RasAlKhaimah-Location', 'RasAlKhaimah-Location-Other-Text',
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


### Tranform Multiselect Columns
# function to handle each needed column
def expand_multiselect_column(df, column, delimiter=',', drop_original=False):
    """
    Expands a multi-select column into individual boolean columns.

    Parameters:
        df (pd.DataFrame): Original DataFrame
        column (str): Column name with multi-select strings
        delimiter (str): Separator used between values (default is ',')
        drop_original (bool): If True, drops the original multi-select column

    Returns:
        pd.DataFrame: DataFrame with new boolean columns added
    """
    # Step 1: Convert strings to lists, handle NaN
    cleaned = df[column].apply(lambda x: [i.strip() for i in str(x).split(delimiter) if i.strip()] if pd.notna(x) else np.nan)

    # Step 2: Encode non-null rows
    non_null = cleaned.dropna()
    mlb = MultiLabelBinarizer()
    dummies = pd.DataFrame(mlb.fit_transform(non_null), columns=mlb.classes_, index=non_null.index)

    # Step 3: Create full output with NaNs preserved
    full_dummies = pd.DataFrame(index=df.index, columns=mlb.classes_)
    full_dummies.update(dummies)

    # Step 4: Convert 1.0/0.0 to True/False, but keep NaN where applicable
    full_dummies = full_dummies.where(full_dummies.isna(), full_dummies.astype(bool))

    # Step 5: Join to original DataFrame
    df_expanded = pd.concat([df, full_dummies], axis=1)

    print("Column:",column,"\n")
    print("full_dummies:")
    #print(full_dummies.loc[1:25])
    print(full_dummies.columns)
    print()

    if drop_original:
        df_expanded = df_expanded.drop(columns=[column])

    return df_expanded

# Use function on each multiselect column
df = expand_multiselect_column(df, column='Bike-Months', delimiter=',', drop_original=True)
df = expand_multiselect_column(df, column='Ebike-Months', delimiter=',', drop_original=True)
df = expand_multiselect_column(df, column='Scooter-Months', delimiter=',', drop_original=True)
df = expand_multiselect_column(df, column='Bike-Why', delimiter=',', drop_original=True)
df = expand_multiselect_column(df, column='Ebike-Why', delimiter=',', drop_original=True)
df = expand_multiselect_column(df, column='Scooter-Why', delimiter=',', drop_original=True)

df = df.rename(columns={ # avoids crossover with alternate columns using 'other'
    'Other': 'TranspoType-Why-Other'
})

# Emirates with multiselect locations
df = expand_multiselect_column(df, column='AbuDhabi-Location', delimiter=',', drop_original=True)
df = expand_multiselect_column(df, column='AlAin-Location', delimiter=',', drop_original=True)
df = expand_multiselect_column(df, column='Dubai-Location', delimiter=',', drop_original=True)
df = expand_multiselect_column(df, column='Sharjah-Location', delimiter=',', drop_original=True)
df = expand_multiselect_column(df, column='RasAlKhaimah-Location', delimiter=',', drop_original=True)

df = df.rename(columns={
    'Other': 'Emirate-Location-Other'
})

# Person attributes with multiselect
df = expand_multiselect_column(df, column='WhoRideWith', delimiter=',', drop_original=True)
df = expand_multiselect_column(df, column='BikeType', delimiter=',', drop_original=True)

df = df.rename(columns={
    'Other': 'BikeType-Other'
})





### Merge exisiting columns
from collections import Counter

# Get all duplicated column names
duplicates = [col for col, count in Counter(df.columns).items() if count > 1]

for col in duplicates:
    # Get all columns with the same name
    same_cols = df.loc[:, df.columns == col]
    
    # Combine them row-wise using logical OR, preserving NaNs where all are missing
    merged = same_cols.any(axis=1)  # True if any are True

    # If all values in the row are NaN, set result to NaN (instead of False)
    all_nan_mask = same_cols.isna().all(axis=1)
    merged[all_nan_mask] = np.nan

    # Drop original duplicate columns and insert merged one
    df = df.drop(columns=same_cols.columns)
    df[col] = merged



### Reorder months
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

# Keep only those month columns that exist in the DataFrame
existing_months = [month for month in month_order if month in df.columns]

# Get remaining columns
other_cols = [col for col in df.columns if col not in existing_months]

# Reorder columns
df = df[other_cols + existing_months]



### Rename added columns from multiselect function
df = df.rename(columns={
    # TranspoType-Why
    'I want to be the next Tour de France winner': 'TourdeFrance',
    'Commuting/Mode of Transportation': 'Commuting',
    'Competitive/Racing': 'Racing',
    'Recreational/Leisure': 'Leisure',
    # AbuDhabi-Location
    'Al Bateen': 'AlBateen',
    'Al Hudayriyat Island': 'AlHudayriyatIsland',
    'Al Maryah Island': 'AlMaryahIsland',
    'Al Raha': 'AlRaha',
    'Al Reem Island': 'AlReemIsland',
    'Al Saadiyat Island': 'AlSaadiyatIsland',
    'Al Shahama': 'AlShahama',
    'Al Wathba Cycle Track': 'AlWathbaCycleTrack',
    'Al Zahiyah/Tourist Club area': 'TouristClub',
    'Khalifa City': 'KhalifaCity',
    'Yas Island/Yas Marina Circuit': 'YasIsland',
    # AlAin-Location
    'Al Ain Cycle Track': 'AlAinCycleTrack',
    'Al Jimi': 'AlJimi',
    'Al Mutarid': 'AlMutarid',
    'Al Mutawah': 'AlMutawah',
    'Central District': 'CentralDistrict',
    'Jebel Hafeet': 'JebelHafeet',
    # Dubai-Location
    'Al Qudra Cycle Track' : 'AlQudraCycleTrack',
    'Bur Dubai': 'BurDubai',
    'Business Bay': 'BusinessBay',
    'Downtown Dubai': 'DowntownDubai',
    'Dubai Marina': 'DubaiMarina',
    'JLT (Jumeirah Lake Towers)': 'JLT',
    'Jumeirah Beach': 'JumeirahBeach',
    'Meydan DXBike': 'MeydanDXBike',
    'Mushrif Park': 'MushrifPark',
    # Sharjah-Location 
    'Al Batayeh Bicycle Track': 'AlBatayehBicycleTrack',
    'Al Qasimia': 'AlQasimia',
    'Al Riqa': 'AlRiqa',
    'Industrial Area': 'IndustrialArea',
    'Khor Fakkan': 'KhorFakkan',
    'Masaar Cycling Track': 'MasaarCyclingTrack',
    'Sharjah Corniche/Al Majaz': 'SharjahCorniche',
    'University City': 'UniversityCity',
    # RasAlKhaimah-Location
    'Al Jazeera al Hamra': 'AlJazeeraAlHamra',
    'Al Marjan Island': 'AlMarjanIsland',
    'Downtown Ras al Khaimah': 'DowntownRasAlKhaimah',
    'Jebel Jais': 'JebelJais',
    'Mina al Arab': 'MinaAlArab',
    # WhoRideWith
    'By myself': 'WithMyself',
    'With a cycling club': 'WithCyclingClub',
    'with family': 'WithFamily',
    'with friends/other riders': 'WithFriends',
    # BikeType
    'BMX Bike': 'BMXBike',
    'Cyclocross Bike': 'CyclocrossBike',
    'Electric Bike': 'ElectricBike',
    'Folding Bike': 'FoldingBike',
    'Hybrid Bike': 'HybridBike',
    'Mountain Bike': 'MountainBike',
    'NotSure': 'NotSure',
    'Road Bike': 'RoadBike'
})


################################################################################################################
print('################### TESTING ###################')
print(df.columns[-35:])
print(df.loc[1:50,['Emirate' ,'WithMyself','RoadBike']])
print()
print('###############################################')
################################################################################################################


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

# Used for exporting an excel file of cleaned dataframe
#df.to_excel('Bike-UAE_Basic-Cleaned.xlsx', index=False)

'''
# Check influence of Gender on Owning a Bike
table = pd.crosstab(df['Gender'], df['OwnBike'])
print(table)
chi2, p, dof, expected = chi2_contingency(table)

print("Chi-Square Statistic:", chi2)
print("Degrees of Freedom:", dof)
print("p-value:", p)
'''