### 2025-06-16_Bike-UAE_Survey-Data
### Max Moran

# Set Working Directory to folder with data
import os
path = r"C:\Users\might\Documents\UTSA\Summer_2025\STA-6943_Internship\bike-UAE" #r in front is for the back slashes
os.chdir(path)

import pandas as pd

# Import data
df = pd.read_csv('LABELS-UAE Bicycle & Scooter Survey - CITIES_June 16, 2025_16.12.csv')

# Drop unnecessary rows
print(df.head())
df = df.drop([0,1])
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
print(df.head())