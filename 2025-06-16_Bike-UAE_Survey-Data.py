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
print(df.head())

# Drop unnecessary rows
df = df.drop([0,1])
print("Dataframe after dropping unecessary context rows:")
print(df.head())

# List of new column names
new_column_names = [# Automatically generated survey data
                    'StartDate', 'EndDate', 'Status', 'Progress',
                    'Duration', 'Finished', 'RecordedDate', 'ResponseID',
                    'DistributionChannel', 'UserLang', 'Introduction', 'Emirate',
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
                    'AlDhafra-', 'AlDhafra-Location',
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
print(df.head())

# Convert duration to int then minutes
df['Duration'] = df['Duration'].astype(int)
df['Duration'] = df['Duration'] / 60

# Convert progress to int
df['Progress'] = df['Progress'].astype(int)



### Handle spam
# 38 questions for 100% completion, assume ~3 seconds a question; that's 114 seconds (~2 minutes)
# So, any entries with 100% completion in under 2 minutes we'll be removed (likely spam through questions)
df = df[~((df['Duration'] < 2) & (df['Progress'] == 100))]
print("Dataframe without 100% entries under 2 minutes:")
print(df)



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
print("Removed",len(df['Duration'])-len(dfDurationIQR['Duration']),"outliers")

sns.boxplot(x=dfDurationIQR['Duration'])
plt.title('Narrowed Survey Duration Box Plot')
plt.xlabel('Duration (minutes)')
plt.show()

dfDurationIQR["Duration"].hist(bins=100)
#plt.ylim(0, 20)
#plt.xlim(0, 60)
plt.xlabel('Duration (minutes)')
plt.ylabel('Number of Respondents')
plt.title('Survey Completion Time Distribution')
plt.show()