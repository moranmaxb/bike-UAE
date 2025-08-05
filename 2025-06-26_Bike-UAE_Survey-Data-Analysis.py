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

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

import re

# Set Working Directory to folder with data
path = r"C:\Users\might\Documents\UTSA\Summer_2025\STA-6943_Internship\bike-UAE" #r in front is for the back slashes
os.chdir(path)

# Import data
df = pd.read_excel('Bike-UAE_Basic-Cleaned.xlsx')
#print(df.head(),"\n")

############################################################################################
#                                         VISUALIZATION       
############################################################################################

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

# Map TranspoType-HowOften to numeric scale
howOften_mapping = {
    'Daily': 365,
    '4-6 times a week': 260,
    '2-3 times a week': 130,
    'Once a week': 52,
    'A few times a month': 24,
    'Once a month': 12,
    'A few times a year': 4
}
df['HowOftenScore'] = df['TranspoType-HowOften'].map(howOften_mapping)


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

####################################################################

### Wordcloud ###

from wordcloud import WordCloud, STOPWORDS

def plot_wordclouds(df, columns, extra_stopwords=None, colormap='viridis'):
    """
    Generate and display word clouds for each column in a list of text columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame
    - columns (list of str): List of column names (strings) to generate word clouds for
    - extra_stopwords (set of str): Optional set of additional stopwords
    - colormap (str): Optional matplotlib colormap for the word cloud
    """
    base_stopwords = set(STOPWORDS)
    domain_stopwords = {'bike', 'biking', 'cycling', 'riding'}
    all_stopwords = base_stopwords.union(domain_stopwords)
    if extra_stopwords:
        all_stopwords.update(extra_stopwords)

    for col in columns:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not found in DataFrame. Skipping.")
            continue
        
        # Combine all text entries in column
        text = df[col].dropna().astype(str).str.cat(sep=' ').lower()
        if not text.strip():
            print(f"⚠️ Column '{col}' contains no valid text. Skipping.")
            continue
        
        # Generate word cloud
        wordcloud = WordCloud(width=1000, height=600, background_color='white',
                              stopwords=all_stopwords, colormap=colormap).generate(text)

        # Plot
        plt.figure(figsize=(12, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for '{col}'", fontsize=16)
        plt.tight_layout()
        plt.show()

columns_to_plot = ['TranspoType-Why-Other-Text', 'Education-Other-Text',
                   'KidsCycle-No-Text', 'NoRideCommute-Other-Text', 'LifestyleInfluence-Other-Text',
                   'Challenges-Other-Text', 'ExperienceImprovement-Other-Text', 'CycleShopsInfluence-Other-Text',
                   'CycleShopsBenefit-Other-Text', 'AlDhafra-Location', 'UmmAlQuwain-Location', 'Emirate-Location-Other-Text']
plot_wordclouds(df, columns_to_plot)





######################################################################

### Frequency Plots ###



def plot_true_frequencies(df, columns, as_percentage=True, title="Frequency of Reasons for Riding", color='steelblue'):
    """
    Plots the frequency or percentage of True values for given Boolean columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame
    - columns (list of str): Boolean column names to include
    - as_percentage (bool): If True, plot percentages instead of raw counts
    - title (str): Title for the plot
    - color (str): Bar color
    """
    # Validate columns and count True values
    valid_cols = [col for col in columns if col in df.columns]
    true_counts = df[valid_cols].apply(lambda col: col.fillna(False).sum())
    total_counts = df[valid_cols].notna().sum()

    if as_percentage:
        frequencies = (true_counts / total_counts * 100).round(2)
        ylabel = "Percentage of Users (%)"
    else:
        frequencies = true_counts
        ylabel = "Number of Users"

    # Plot
    plt.figure(figsize=(10, 6))
    frequencies.sort_values(ascending=False).plot(kind='bar', color=color)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



## Why Ride Frequencies
reason_columns = [
    'Why-TourdeFrance',
    'Why-Commuting',
    'Why-Racing',
    'Why-Fitness',
    'Why-Leisure',
    'TranspoType-Why-Other'
]

plot_true_frequencies(df, reason_columns, as_percentage=True, title="Why Ride")

## Bike Type Frequencies
reason_columns = [
    'Type-BMXBike',
    'Type-CyclocrossBike',
    'Type-ElectricBike',
    'Type-FoldingBike',
    'Type-HybridBike',
    'Type-MountainBike',
    'Type-RoadBike',
    'Not sure',
    'BikeType-Other'
]

plot_true_frequencies(df, reason_columns, as_percentage=True, title="Bike Types")

## No Ride Reason Frequencies
reason_columns = [
    'NoRideReason-Cultural',
    'NoRideReason-Distance',
    'NoRideReason-Judgment',
    'NoRideReason-NoPaths',
    'NoRideReason-NoFacilities',
    'NoRideReason-TooLong',
    'NoRideCommute-Other'
]

plot_true_frequencies(df, reason_columns, as_percentage=True, title="Reasons not to Ride")

## AltMicro Frequencies
reason_columns = [
    'AltMicro-Bike',
    'AltMicro-EBike',
    'AltMicro-EScooter',
    'AltMicro-ESkateboard',
    'AltMicro-Camel',
    'AltMicro-No'
]

plot_true_frequencies(df, reason_columns, as_percentage=True, title="Alternate Mirco-mobility")

## Life Style Influence Frequencies
reason_columns = [
    'LSI-ParticpationInEvents',
    'LSI-ReduceCarbonFootprint',
    'LSI-ImprovedPhysicalFitness',
    'LSI-EasierDailyCommute',
    'LSI-LifestyleNoneOfAbove',
    'LSI-NewWayToExplore',
    'LifestyleInfluence-Other - Boolean'
]

plot_true_frequencies(df, reason_columns, as_percentage=True, title="Riding Life Style Influences")

## Comfort Frequencies
reason_columns = [
    'Comfort-CityStreets',
    'Comfort-CyclePaths',
    'Comfort-CycleTracks',
    'Comfort-Highways',
    'Comfort-NeighborhoodRoads',
    'Comfort-Parks',
    'Comfort-Sidewalks'
]

plot_true_frequencies(df, reason_columns, as_percentage=True, title="Comfortable Riding Locations")

## Challenges Frequencies
reason_columns = [
    'Chal-ChargingAccess',
    'Chal-CulturalNorms',
    'Chal-ExtremeWeather',
    'Chal-EquipmentCost',
    'Chal-Judgement',
    'Chal-NoCycleLanes',
    'Chal-NoCycleParking',
    'Chal-PersonalSafetyConcerns',
    'Chal-PetrolPrices',
    'Chal-RoadSafetyConcerns',
    'Challenges-Other'
]

plot_true_frequencies(df, reason_columns, as_percentage=True, title="Riding Challenges")

## Experience Improvement Frequencies
reason_columns = [
    'ExpImp-IndoorCyclingAccess',
    'ExpImp-Awareness',
    'ExpImp-BetterRoads',
    'ExpImp-BetterSignage',
    'ExpImp-MoreBikeRentals',
    'ExpImp-MoreCyclingLanes',
    'ExpImp-MoreDedicatedTracks',
    'ExpImp-MoreShadedAreas',
    'ExperienceImprovement-Other'
]

plot_true_frequencies(df, reason_columns, as_percentage=True, title="Experience Improvements")

## Cycle Shop Influence Frequencies
reason_columns = [
    'CSI-SpendTimeWithFamily',
    'CSI-ConnectedGrowingCulture',
    'CSI-CharityCyclingEvents',
    'CSI-CyclingGroupsAndFriends',
    'CSI-WorkshopEventAccess',
    'CycleShopsInfluence-Other'
]

plot_true_frequencies(df, reason_columns, as_percentage=True, title="Cycle Shop Influence")

## Cycle Shop Benefit Frequencies
reason_columns = [
    'CSB-ReliableMaintenance',
    'CSB-WelcomingCommunity',
    'CSB-PremiumGearAccess',
    'CSB-HighQualityExpertService',
    'CSB-HealthyMotivation',
    'CycleShopsBenefit-Other'
]

plot_true_frequencies(df, reason_columns, as_percentage=True, title="Cycle Shop Benefits")



### Location Frequencies ###

## Abu Dhabi Frequencies
reason_columns = [
    'AD-AlBateen',
    'AD-AlHudayriyatIsland',
    'AD-AlMaryahIsland',
    'AD-AlRaha',
    'AD-AlReemIsland',
    'AD-AlSaadiyatIsland',
    'AD-AlShahama',
    'AD-AlWathbaCycleTrack',
    'AD-TouristClub',
    'AD-Corniche',
    'AD-KhalifaCity',
    'AD-Musaffah',
    'AD-YasIsland'
]

plot_true_frequencies(df, reason_columns, as_percentage=True, title="Abu Dhabi Riding Locations")

## Al Ain Frequencies
reason_columns = [
    'AA-AlAinCycleTrack',
    'AA-AlJimi',
    'Al Mutarid',
    'AA-AlMutawah',
    'AA-CentralDistrict',
    'AA-Hili - Boolean',
    'AA-JebelHafeet'
]

plot_true_frequencies(df, reason_columns, as_percentage=True, title="Al Ain Riding Locations")

## Dubai Frequencies
reason_columns = [
    'D-AlQudraCycleTrack',
    'D-BurDubai',
    'D-BusinessBay',
    'D-Deira',
    'D-DowntownDubai',
    'D-DubaiMarina',
    'D-JLT',
    'D-JumeirahBeach',
    'D-MeydanDXBike',
    'D-MushrifPark'
]

plot_true_frequencies(df, reason_columns, as_percentage=True, title="Dubai Riding Locations")

## Sharjah Frequencies
reason_columns = [
    'S-AlBatayehBicycleTrack',
    'S-AlQasimia',
    'S-AlRiqa',
    'S-IndustrialArea',
    'S-KhorFakkan',
    'S-MasaarCyclingTrack',
    'S-SharjahCorniche',
    'S-UniversityCity'
]

plot_true_frequencies(df, reason_columns, as_percentage=True, title="Sharjah Riding Locations")

## Ras al Khaimah Frequencies
reason_columns = [
    'RAK-AlJazeeraAlHamra',
    'RAK-AlMarjanIsland',
    'RAK-DowntownRasAlKhaimah',
    'RAK-JebelJais',
    'RAK-MinaAlArab'
]

plot_true_frequencies(df, reason_columns, as_percentage=True, title="Ras al Khaimah Riding Locations")

## Country Frequencies
country_counts = df['Country'].dropna().value_counts()

plt.figure(figsize=(12, 6))
country_counts.plot(kind='bar', color='mediumseagreen')
plt.title('Frequency of Respondents by Country')
plt.xlabel('Country')
plt.ylabel('Number of Respondents')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

## Emirate Frequencies
emirate_counts = df['Emirate'].dropna().value_counts()

plt.figure(figsize=(12, 6))
emirate_counts.plot(kind='bar', color='mediumseagreen')
plt.title('Frequency of Respondents by Emirate')
plt.xlabel('Emirate')
plt.ylabel('Number of Respondents')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



############################################################################################
#                                       ANALYSIS
############################################################################################

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

### ANOVA TEST Frequency ###
print("\n--- ANOVA AGAIN Tests ---\n")

# Copy the original dataframe
df_anova = df.copy()

# Define features to test
features = [
    'DurationInUAE','Country','Age','Salary','YearsRiding',
    'TrackRides','Emirate','OwnBike','EmploymentStatus','Gender'
]

# Dictionary to store p-values
pval_dict = {}

# Loop and calculate ANOVA
for feature in features:
    df_test = df_anova[df_anova['SafetyScore'].notna() & df_anova[feature].notna()].copy()
    df_test[feature] = df_test[feature].astype('category')
    
    formula = f'SafetyScore ~ C(Q("{feature}"))'
    model = ols(formula, data=df_test).fit()
    anova_table = anova_lm(model, typ=2)

    # Store p-value from the feature's row (not the residual)
    pval_dict[feature] = anova_table["PR(>F)"].iloc[0]

# Create and display DataFrame of p-values
pval_df = pd.DataFrame.from_dict(pval_dict, orient='index', columns=['p-value'])
pval_df = pval_df.sort_values(by='p-value')

print("\n### SafetyScore ANOVA P-Values for Each Feature ###")
print(pval_df)






### ANOVA TEST Frequency ###
print("\n--- ANOVA AGAIN Tests ---\n")

# Copy the original dataframe
df_anova = df.copy()

# Define features to test
features = [
    'DurationInUAE','Country','Age','Salary','YearsRiding',
    'TrackRides','Emirate','OwnBike','EmploymentStatus','Gender'
]

# Dictionary to store p-values
pval_dict = {}

# Loop and calculate ANOVA
for feature in features:
    df_test = df_anova[df_anova['HowOftenScore'].notna() & df_anova[feature].notna()].copy()
    df_test[feature] = df_test[feature].astype('category')
    
    formula = f'HowOftenScore ~ C(Q("{feature}"))'
    model = ols(formula, data=df_test).fit()
    anova_table = anova_lm(model, typ=2)

    # Store p-value from the feature's row (not the residual)
    pval_dict[feature] = anova_table["PR(>F)"].iloc[0]

# Create and display DataFrame of p-values
pval_df = pd.DataFrame.from_dict(pval_dict, orient='index', columns=['p-value'])
pval_df = pval_df.sort_values(by='p-value')

print("\n### HowOftenScore ANOVA P-Values for Each Feature ###")
print(pval_df)

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

# Run ANOVA
df_anova_howoften = df[['SafetyScore', 'HowOftenScore']].dropna()
df_anova_howoften['HowOftenScore'] = df_anova_howoften['HowOftenScore'].astype('category')

model = ols('SafetyScore ~ C(HowOftenScore)', data=df_anova_howoften).fit()
anova_result = anova_lm(model, typ=2)

print("Safety vs How Often Anova:")
print(anova_result)



# Drop NA values
df_model = df[['SafetyScore', 'HowOftenScore']].dropna()

# Prepare X and y
X = df_model[['HowOftenScore']].values
y = df_model['SafetyScore'].values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# Plot
plt.figure(figsize=(8, 6))
sns.regplot(x='HowOftenScore', y='SafetyScore', data=df_model, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title(f'Safety Score vs How Often They Ride\n$R^2$ = {r2:.3f}')
plt.xlabel('Riding Frequency (days/yr)')
plt.ylabel('Safety Score (1 = Safe, 5 = Not Safe)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



# Ensure required columns are present and drop NA
df_anova_own = df[['OwnBike', 'HowOftenScore']].dropna()

# Convert OwnBike to categorical
df_anova_own['OwnBike'] = df_anova_own['OwnBike'].astype('category')
df_anova_own['HowOftenScore'] = df_anova_own['HowOftenScore'].astype('float')

# Fit ANOVA model
model = ols('HowOftenScore ~ C(OwnBike)', data=df_anova_own).fit()
anova_result = anova_lm(model, typ=2)

print("Own Bike vs How Often Anova:")
print(anova_result)



#########

# Filter rows with valid HowOftenScore
df_rf = df[df['HowOftenScore'].notna()].copy()

# Expanded list of features
predictor_columns = [
    'OwnBike', 'Gender', 'Age', 'Salary', 'Emirate', 'Country', 'Nationality',
    'Why-Commuting', 'Why-Racing', 'Why-Fitness', 'Why-Leisure',
    'TrackRides', 'EmploymentStatus', 'DurationInUAE', 'YearsRiding',
    'RideWith-Myself', 'RideWith-CyclingClub', 'RideWith-Family', 'RideWith-Friends'
]

# Filter only existing columns
predictor_columns = [col for col in predictor_columns if col in df_rf.columns]

# Encode categorical and boolean columns
for col in predictor_columns:
    df_rf[col] = df_rf[col].astype(str)
    le = LabelEncoder()
    df_rf[col] = le.fit_transform(df_rf[col])

# Prepare X and y
X = df_rf[predictor_columns]
y = df_rf['HowOftenScore']

# Fit Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Feature importances
importances = pd.Series(rf_model.feature_importances_, index=predictor_columns).sort_values(ascending=False)

#########

# Updated: Remove group-based or motivational variables from main model
reduced_predictors = [
    'OwnBike', 'Gender', 'Age', 'Salary', 'Emirate', 'Country', 'Nationality',
    'TrackRides', 'EmploymentStatus', 'DurationInUAE', 'YearsRiding'
]

# Filter only existing columns
reduced_predictors = [col for col in reduced_predictors if col in df_rf.columns]

# Encode categorical
df_main = df_rf[reduced_predictors + ['HowOftenScore']].copy()
for col in reduced_predictors:
    df_main[col] = df_main[col].astype(str)
    le = LabelEncoder()
    df_main[col] = le.fit_transform(df_main[col])

# Fit random forest on reduced features
X_main = df_main[reduced_predictors]
y_main = df_main['HowOftenScore']

rf_main = RandomForestRegressor(n_estimators=100, random_state=42)
rf_main.fit(X_main, y_main)

importances_main = pd.Series(rf_main.feature_importances_, index=reduced_predictors).sort_values(ascending=False)

# Plot main feature importance
plt.figure(figsize=(8, 6))
importances_main.plot(kind='barh', color='mediumorchid')
plt.title('Feature Importance for Riding Frequency')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ===============================
# Separate model: Motivation-based features only
motivation_vars = [
    'Why-Racing', 'Why-Leisure', 'Why-Commuting', 'Why-TourdeFrance', 'Why-Fitness'
]
motivation_vars = [col for col in motivation_vars if col in df_rf.columns]

# Encode categorical
df_motive = df_rf[motivation_vars + ['HowOftenScore']].copy()
for col in motivation_vars:
    df_motive[col] = df_motive[col].astype(str)
    le = LabelEncoder()
    df_motive[col] = le.fit_transform(df_motive[col])

# Fit model on motivation vars
X_motive = df_motive[motivation_vars]
y_motive = df_motive['HowOftenScore']

rf_motive = RandomForestRegressor(n_estimators=100, random_state=42)
rf_motive.fit(X_motive, y_motive)

importances_motive = pd.Series(rf_motive.feature_importances_, index=motivation_vars).sort_values(ascending=False)

# Plot motivation-only feature importances
plt.figure(figsize=(6, 4))
importances_motive.plot(kind='barh', color='salmon')
plt.title('Motivations Feature Importance for Riding Frequency')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

