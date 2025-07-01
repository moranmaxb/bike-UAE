### 2025-06-16_Bike-UAE_Survey-Data-Cleaning.py
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
df = pd.read_csv('LABELS-UAE Bicycle & Scooter Survey - CITIES_June 16, 2025_16.12.csv')
print(df.head(),"\n")

# Drop unnecessary rows
df = df.drop([0,1])
print("Dataframe after dropping unnecessary context rows:")
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
                    'ComfortLocation', 'Challenges', 'Challenges-Other-Text',
                    'ExperienceImprovement', 'ExperienceImprovement-Other-Text', 'AltMicromobility',
                    'KidsCycle', 'KidsCycle-No-Text',
                    'RegularCommute', 'NoRideCommute', 'NoRideCommute-Other-Text',
                    # Individual details
                    'Gender', 'Age', 'Country',
                    'Education', 'Education-Other-Text',
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

################# Transform Multi Select Columns #################
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
    # Convert strings to lists, handle NaN
    cleaned = df[column].apply(lambda x: [i.strip() for i in str(x).split(delimiter) if i.strip()] if pd.notna(x) else np.nan)

    # Encode non-null rows
    non_null = cleaned.dropna()
    mlb = MultiLabelBinarizer()
    dummies = pd.DataFrame(mlb.fit_transform(non_null), columns=mlb.classes_, index=non_null.index)

    # Create full output with NaNs preserved
    full_dummies = pd.DataFrame(index=df.index, columns=mlb.classes_)
    full_dummies.update(dummies)

    # Convert 1.0/0.0 to True/False, but keep NaN where applicable
    full_dummies = full_dummies.where(full_dummies.isna(), full_dummies.astype(bool))

    # Join to original DataFrame
    df_expanded = pd.concat([df, full_dummies], axis=1)

    # Debugging Prints
    #print("Column:",column,"\n")
    #print("full_dummies:")
    ##print(full_dummies.loc[1:25])
    #print(full_dummies.columns)
    #print()

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

# Multiselect for WhoRideWith & BikeType
df = expand_multiselect_column(df, column='WhoRideWith', delimiter=',', drop_original=True)
df = expand_multiselect_column(df, column='BikeType', delimiter=',', drop_original=True)
df = df.rename(columns={
    'Other': 'BikeType-Other'
})

# Multiselect for LifestyleInfluence
df = expand_multiselect_column(df, column='LifestyleInfluence', delimiter=',', drop_original=True)
df = df.rename(columns={
    'Other': 'LifestyleInfluence-Other'
})

# Multiselect for CycleShopsInfluence
# Fix commas being present in answers for CycleShopInflucne
df['CycleShopsInfluence'] = df['CycleShopsInfluence'].str.replace(
    'Provided access to workshops, events, or training sessions that enriched my cycling experience',
    'Provided access to workshops-events-or training sessions that enriched my cycling experience'
)
df = expand_multiselect_column(df, column='CycleShopsInfluence', delimiter=',', drop_original=True)
df = df.rename(columns={
    'Other': 'CycleShopsInfluence-Other'
})

# Multiselect for CycleShopsBenefit
# Fix commas being present in answers for CycleShopInflucne
df['CycleShopsBenefit'] = df['CycleShopsBenefit'].str.replace(
    'A reliable and trustworthy place for maintenance, repairs, and upgrades',
    'A reliable and trustworthy place for maintenance-repairs-and upgrades'
)
df = expand_multiselect_column(df, column='CycleShopsBenefit', delimiter=',', drop_original=True)
df = df.rename(columns={
    'Other': 'CycleShopsBenefit-Other'
})


# Multiselect for ComfortLocation & Challenges
df = expand_multiselect_column(df, column='ComfortLocation', delimiter=',', drop_original=True)
df = expand_multiselect_column(df, column='Challenges', delimiter=',', drop_original=True)
df = df.rename(columns={
    'Other': 'Challenges-Other'
})

# Multiselect for ExperienceImprovement
df = expand_multiselect_column(df, column='ExperienceImprovement', delimiter=',', drop_original=True)
df = df.rename(columns={
    'Other': 'ExperienceImprovement-Other'
})

# Multiselect for AltMicromobility & NoRideCommute
df = expand_multiselect_column(df, column='AltMicromobility', delimiter=',', drop_original=True)
# Fix commas being present in answers for NoRideCommute
df['NoRideCommute'] = df['NoRideCommute'].str.replace(
    'Lack of facilities at work (showers, cycle parking, etc.)',
    'Lack of facilities at work (showers-cycle parking-etc.)'
)
df = expand_multiselect_column(df, column='NoRideCommute', delimiter=',', drop_original=True)
df = df.rename(columns={
    'Other': 'NoRideCommute-Other'
})




### Merge exisiting columns (transport and months)
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
    'I want to be the next Tour de France winner': 'Why-TourdeFrance',
    'Commuting/Mode of Transportation': 'Why-Commuting',
    'Fitness': 'Why-Fitness',
    'Competitive/Racing': 'Why-Racing',
    'Recreational/Leisure': 'Why-Leisure',
    # AbuDhabi-Location
    'Al Bateen': 'AD-AlBateen',
    'Al Hudayriyat Island': 'AD-AlHudayriyatIsland',
    'Al Maryah Island': 'AD-AlMaryahIsland',
    'Al Raha': 'AD-AlRaha',
    'Al Reem Island': 'AD-AlReemIsland',
    'Al Saadiyat Island': 'AD-AlSaadiyatIsland',
    'Al Shahama': 'AD-AlShahama',
    'Al Wathba Cycle Track': 'AD-AlWathbaCycleTrack',
    'Al Zahiyah/Tourist Club area': 'AD-TouristClub',
    'Corniche': 'AD-Corniche',
    'Khalifa City': 'AD-KhalifaCity',
    'Musaffah': 'AD-Musaffah',
    'Yas Island/Yas Marina Circuit': 'AD-YasIsland',
    # AlAin-Location
    'Al Ain Cycle Track': 'AA-AlAinCycleTrack',
    'Al Jimi': 'AA-AlJimi',
    'Al Mutarid': 'AA-AlMutarid',
    'Al Mutawah': 'AA-AlMutawah',
    'Central District': 'AA-CentralDistrict',
    'Hili': 'AA-Hili',
    'Jebel Hafeet': 'AA-JebelHafeet',
    # Dubai-Location
    'Al Qudra Cycle Track' : 'D-AlQudraCycleTrack',
    'Bur Dubai': 'D-BurDubai',
    'Business Bay': 'D-BusinessBay',
    'Deira': 'D-Deira',
    'Downtown Dubai': 'D-DowntownDubai',
    'Dubai Marina': 'D-DubaiMarina',
    'JLT (Jumeirah Lake Towers)': 'D-JLT',
    'Jumeirah Beach': 'D-JumeirahBeach',
    'Meydan DXBike': 'D-MeydanDXBike',
    'Mushrif Park': 'D-MushrifPark',
    # Sharjah-Location 
    'Al Batayeh Bicycle Track': 'S-AlBatayehBicycleTrack',
    'Al Qasimia': 'S-AlQasimia',
    'Al Riqa': 'S-AlRiqa',
    'Industrial Area': 'S-IndustrialArea',
    'Khor Fakkan': 'S-KhorFakkan',
    'Masaar Cycling Track': 'S-MasaarCyclingTrack',
    'Sharjah Corniche/Al Majaz': 'S-SharjahCorniche',
    'University City': 'S-UniversityCity',
    # RasAlKhaimah-Location
    'Al Jazeera al Hamra': 'RAK-AlJazeeraAlHamra',
    'Al Marjan Island': 'RAK-AlMarjanIsland',
    'Downtown Ras al Khaimah': 'RAK-DowntownRasAlKhaimah',
    'Jebel Jais': 'RAK-JebelJais',
    'Mina al Arab': 'RAK-MinaAlArab',
    # WhoRideWith
    'By myself': 'RideWith-Myself',
    'With a cycling club': 'RideWith-CyclingClub',
    'with family': 'RideWith-Family',
    'with friends/other riders': 'RideWith-Friends',
    # BikeType
    'BMX Bike': 'Type-BMXBike',
    'Cyclocross Bike': 'Type-CyclocrossBike',
    'Electric Bike': 'Type-ElectricBike',
    'Folding Bike': 'Type-FoldingBike',
    'Hybrid Bike': 'Type-HybridBike',
    'Mountain Bike': 'Type-MountainBike',
    'NotSure': 'Type-NotSure',
    'Road Bike': 'Type-RoadBike',
    # LifestyleInfluence
    'Allowed me to participate in cycling events and meet like-minded people': 'LSI-ParticpationInEvents',
    'Helped me reduce my carbon footprint and be more eco-friendly': 'LSI-ReduceCarbonFootprint',
    'Improved my physical fitness and overall health': 'LSI-ImprovedPhysicalFitness',
    'Made my daily commute easier and more efficient': 'LSI-EasierDailyCommute',
    'None of the above': 'LSI-LifestyleNoneOfAbove',
    'Provided me with a new way to explore and enjoy the UAE': 'LSI-NewWayToExplore',
    # CycleShopsInfluence
    'Encouraged me to spend more time with family and friends through cycling': 'CSI-SpendTimeWithFamily',
    'Helped me feel more connected to the UAE’s growing cycling culture': 'CSI-ConnectedGrowingCulture',
    'Inspired me to take part in charity or community cycling events': 'CSI-CharityCyclingEvents',
    'Introduced me to cycling groups and new friends': 'CSI-CyclingGroupsAndFriends',
    'Provided access to workshops-events-or training sessions that enriched my cycling experience': 'CSI-WorkshopEventAccess',
    # CycleShopsBenefit
    'A reliable and trustworthy place for maintenance-repairs-and upgrades': 'CSB-ReliableMaintenance',
    'A welcoming community and knowledgeable staff who support my cycling journey': 'CSB-WelcomingCommunity',
    'Access to premium cycling gear and accessories that improved my performance': 'CSB-PremiumGearAccess',
    'High-quality products and expert service that enhanced my cycling experience': 'CSB-HighQualityExpertService',
    'Motivation to lead a healthier and more active lifestyle': 'CSB-HealthyMotivation',
    # ComfortLocation
    'City streets': 'Comfort-CityStreets',
    'Cycle paths': 'Comfort-CyclePaths',
    'Cycle tracks': 'Comfort-CycleTracks',
    'Highways': 'Comfort-Highways',
    'Parks': 'Comfort-Parks',
    'Sidewalks': 'Comfort-Sidewalks',
    'Neighborhood roads': 'Comfort-NeighborhoodRoads',
    # Challenges
    'Charging access': 'Chal-ChargingAccess',
    'Cultural norms/expectations': 'Chal-CulturalNorms',
    'Extreme weather conditions': 'Chal-ExtremeWeather',
    'High costs of equipment or gear': 'Chal-EquipmentCost',
    'Judgement from others': 'Chal-Judgement',
    'Lack of cycle and scooter lanes': 'Chal-NoCycleLanes',
    'Limited cycle/scooter parking or storage': 'Chal-NoCycleParking',
    'Personal safety concerns': 'Chal-PersonalSafetyConcerns',
    'Rising petrol prices': 'Chal-PetrolPrices',
    'Road safety concerns': 'Chal-RoadSafetyConcerns',
    # ExperienceImprovement
    'Access to indoor cycling options': 'ExpImp-IndoorCyclingAccess',
    'Awareness for drivers and cyclists': 'ExpImp-Awareness',
    'Better roads free from obstacles': 'ExpImp-BetterRoads',
    'Better signage and road markings': 'ExpImp-BetterSignage',
    'More bike rental services/stations': 'ExpImp-MoreBikeRentals',
    'More cycling lanes': 'ExpImp-MoreCyclingLanes',
    'More dedicated cycling tracks': 'ExpImp-MoreDedicatedTracks',
    'More shaded or weather-protected riding areas': 'ExpImp-MoreShadedAreas',
    # AltMicromobility
    'Bicycle': 'AltMicro-Bike',
    'Electric Bicycle': 'AltMicro-EBike',
    'Electric Scooter': 'AltMicro-EScooter',
    'Electric Skateboard': 'AltMicro-ESkateboard',
    'I ride a camel': 'AltMicro-Camel',
    'No': 'AltMicro-No',
    # NoRideCommute
    'Cultural reasons/Image': 'NoRideReason-Cultural',
    'Distance from work': 'NoRideReason-Distance',
    'Fear judgment from colleagues': 'NoRideReason-Judgment',
    'Lack of cycle paths from home to work': 'NoRideReason-NoPaths',
    'Lack of facilities at work (showers-cycle parking-etc.)': 'NoRideReason-NoFacilities',
    'Takes too much time': 'NoRideReason-TooLong'
})
################################################################

### Merge compatiable columns
def merge_string_columns(df, column_groups, separator=', '):
    """
    Concatenates multiple string columns row-wise into a single column.
    Preserves NaN if all values in a row are missing.

    Parameters:
        df (pd.DataFrame): Original DataFrame
        column_groups (dict): {new_col_name: [col1, col2, ...]}
        separator (str): Separator between strings

    Returns:
        pd.DataFrame: DataFrame with new merged string columns
    """
    for new_col, cols_to_merge in column_groups.items():
        # Combine strings row-wise, drop NAs per row
        merged = df[cols_to_merge].apply(
            lambda row: separator.join(row.dropna().astype(str)) if row.notna().any() else np.nan,
            axis=1
        )

        # Drop old columns and insert new one
        df = df.drop(columns=cols_to_merge)
        df[new_col] = merged

    return df

column_groups = {
    'Emirate-Safety': ['AbuDhabi-Safety', 'AlAin-Safety', 'AlDhafra-Safety',
                       'Dubai-Safety', 'Sharjah-Safety', # Ajman-Safety was empty
                       'UmmAlQuwain-Safety', 'Fujairah-Safety', 'RasAlKhaimah-Safety'],
    'Emirate-WhyLikeRiding': ['AbuDhabi-WhyLikeRiding', 'AlAin-WhyLikeRiding', 'AlDhafra-WhyLikeRiding',
                    'Dubai-WhyLikeRiding', 'Sharjah-WhyLikeRiding', # Ajman-WhyLikeRiding, UmmAlQuwain-WhyLikeRiding, & Fujairah-WhyLikeRiding were empty
                    'RasAlKhaimah-WhyLikeRiding'],
    'Emirate-Initiatives': ['AbuDhabi-Initiatives', 'AlAin-Initiatives', 'AlDhafra-Initiatives',
                    'Dubai-Initiatives', 'Sharjah-Initiatives',  # Ajman-WhyLikeRiding, UmmAlQuwain-WhyLikeRiding, & Fujairah-WhyLikeRiding were empty
                    'RasAlKhaimah-Initiatives'],
    'TranspoType-HowOften': ['Bike-HowOften', 'Ebike-HowOften', 'Scooter-HowOften'],
    'TranspoType-Why-Other-Text': ['Bike-Why-Other-Text', 'Ebike-Why-Other-Text', 'Scooter-Why-Other-Text'],
    'Emirate-Location-Other-Text': ['AbuDhabi-Location-Other-Text', 'AlAin-Location-Other-Text',
                    'Dubai-Location-Other-Text', 'Sharjah-Location-Other-Text', 'RasAlKhaimah-Location-Other-Text']
    # AlDhafra-Location-Other-Text, Ajman-Location-Other-Text, UmmAlQuwain-Location-Other-Text, & Fujairah-Location-Other-Text were empty
}

df = merge_string_columns(df, column_groups)





### Reorder Columns to logical layout
new_order = [# Starting User Info
             'Finished', 'UserLang', 'Progress', 'Duration',
             # Transport Info
             'TranspoType', 'Type-BMXBike', 'Type-CyclocrossBike', 'Type-ElectricBike', 'Type-FoldingBike', 'Type-HybridBike', 'Type-MountainBike', 'Type-RoadBike', 'Not sure', 'BikeType-Other', 'BikeType-Other-Text',
             'TranspoType-HowOften', 'RiderType', 'RiderType-Other-Text', 'Bikeshare',
             'Why-TourdeFrance', 'Why-Commuting', 'Why-Racing', 'Why-Fitness', 'Why-Leisure', 'TranspoType-Why-Other', 'TranspoType-Why-Other-Text',
             'AltMicro-Bike', 'AltMicro-EBike', 'AltMicro-EScooter', 'AltMicro-ESkateboard', 'AltMicro-Camel', 'AltMicro-No',   
             # Rider Info
             'Gender', 'Age', 'Country', 'Education', 'Education-Other-Text',
             'OwnBike', 'Helmet', 'Earbuds', 'Phones', 'TrackRides',
             'EmploymentStatus', 'Salary', 'DurationInUAE', 'YearsRiding',
             'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December',
             'KidsCycle', 'KidsCycle-No-Text', 'RegularCommute',
             'NoRideReason-Cultural', 'NoRideReason-Distance', 'NoRideReason-Judgment', 'NoRideReason-NoPaths', 'NoRideReason-NoFacilities', 'NoRideReason-TooLong', 'NoRideCommute-Other', 'NoRideCommute-Other-Text',
             'RideWith-Myself', 'RideWith-CyclingClub', 'RideWith-Family', 'RideWith-Friends',
             'LSI-ParticpationInEvents', 'LSI-ReduceCarbonFootprint', 'LSI-ImprovedPhysicalFitness', 'LSI-EasierDailyCommute', 'LSI-LifestyleNoneOfAbove', 'LSI-NewWayToExplore', 'LifestyleInfluence-Other', 'LifestyleInfluence-Other-Text',
             # Riding Area Info
             'Comfort-CityStreets', 'Comfort-CyclePaths', 'Comfort-CycleTracks', 'Comfort-Highways', 'Comfort-NeighborhoodRoads', 'Comfort-Parks', 'Comfort-Sidewalks',
             'Chal-ChargingAccess', 'Chal-CulturalNorms', 'Chal-ExtremeWeather', 'Chal-EquipmentCost', 'Chal-Judgement', 'Chal-NoCycleLanes', 'Chal-NoCycleParking',
             'Chal-PersonalSafetyConcerns', 'Chal-PetrolPrices', 'Chal-RoadSafetyConcerns',  'Challenges-Other', 'Challenges-Other-Text',
             'ExpImp-IndoorCyclingAccess', 'ExpImp-Awareness', 'ExpImp-BetterRoads', 'ExpImp-BetterSignage', 'ExpImp-MoreBikeRentals', 'ExpImp-MoreCyclingLanes', 'ExpImp-MoreDedicatedTracks',
             'ExpImp-MoreShadedAreas', 'ExperienceImprovement-Other', 'ExperienceImprovement-Other-Text',
             # Shop Info
             'CSI-SpendTimeWithFamily', 'CSI-ConnectedGrowingCulture', 'CSI-CharityCyclingEvents', 'CSI-CyclingGroupsAndFriends', 'CSI-WorkshopEventAccess', 'CycleShopsInfluence-Other',  'CycleShopsInfluence-Other-Text',
             'CSB-ReliableMaintenance', 'CSB-WelcomingCommunity', 'CSB-PremiumGearAccess', 'CSB-HighQualityExpertService', 'CSB-HealthyMotivation', 'CycleShopsBenefit-Other', 'CycleShopsBenefit-Other-Text',
             # Location Info
             'Emirate', 'Emirate-Safety', 'Emirate-WhyLikeRiding', 'Emirate-Initiatives',
             'AD-AlBateen', 'AD-AlHudayriyatIsland', 'AD-AlMaryahIsland',
             'AD-AlRaha', 'AD-AlReemIsland', 'AD-AlSaadiyatIsland', 'AD-AlShahama', 'AD-AlWathbaCycleTrack', 'AD-TouristClub', 'AD-Corniche', 'AD-KhalifaCity',
             'AD-Musaffah', 'AD-YasIsland', 'AA-AlAinCycleTrack', 'AA-AlJimi', "Al Mu'tarid", 'AA-AlMutawah', 'AA-CentralDistrict', 'AA-Hili', 'AA-JebelHafeet',
             'D-AlQudraCycleTrack', 'D-BurDubai', 'D-BusinessBay', 'D-Deira', 'D-DowntownDubai', 'D-DubaiMarina', 'D-JLT', 'D-JumeirahBeach', 'D-MeydanDXBike',
             'D-MushrifPark', 'S-AlBatayehBicycleTrack', 'S-AlQasimia', 'S-AlRiqa', 'S-IndustrialArea', 'S-KhorFakkan', 'S-MasaarCyclingTrack', 'S-SharjahCorniche',
             'S-UniversityCity', 'RAK-AlJazeeraAlHamra', 'RAK-AlMarjanIsland', 'RAK-DowntownRasAlKhaimah', 'RAK-JebelJais', 'RAK-MinaAlArab',
             'AlDhafra-Location', 'UmmAlQuwain-Location',
             'Emirate-Location-Other', 'Emirate-Location-Other-Text']

df = df.reindex(columns=new_order)

print("Dataframe with split multiselect and merged questions:")
print(df.head(50))
print()

# Used for exporting an excel file of cleaned dataframe
#df.to_excel('Bike-UAE_Basic-Cleaned.xlsx', index=False)