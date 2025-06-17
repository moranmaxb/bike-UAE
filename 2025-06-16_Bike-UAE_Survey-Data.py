### 2025-06-16_Bike-UAE_Survey-Data
### Max Moran

# Set Working Directory to folder with data
import os
path = r"C:\Users\might\Documents\UTSA\Summer_2025\STA-6943_Internship\bike-UAE" #r in front is for the back slashes
os.chdir(path)

import pandas as pd

df = pd.read_csv('LABELS-UAE Bicycle & Scooter Survey - CITIES_June 16, 2025_16.12.csv')

# Preview the first 5 rows
print(df.head())
print(df["StartDate"])