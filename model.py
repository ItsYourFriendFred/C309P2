import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 50)
df = pd.read_csv(r'C:\Users\Fred\Year Directory\Centennial\Year 4\Semester 2\COMP309 Data Warehousing\Assignments\Group Project\CYCLIST_KSI_1124.csv')

print(df.columns.values)
print (df.head())
print (df.info())

# Data Cleaning

# Dropping OBJECTID, INDEX, ACCNUM, HOOD_158, HOOD_140, DATE, STREET1, STREET2, OFFSET, FATAL_NO as they are unique identifiers that do not help the model generalize
df = df.drop(columns=['OBJECTID', 'INDEX', 'ACCNUM', 'HOOD_158', 'HOOD_140', 'DATE', 'STREET1', 'STREET2', 'OFFSET', 'FATAL_NO'])
# Dropping LATITUDE, LONGITUDE, INITDIR, NEIGHBOURHOOD_140, NEIGHBOURHOOD_158, DIVISION as these geographical features are too narrow to the point of being nearly unique. Also, frankly too intensive for the purpose of this assignment.
df = df.drop(columns=['LATITUDE', 'LONGITUDE', 'INITDIR', 'NEIGHBOURHOOD_140', 'NEIGHBOURHOOD_158', 'DIVISION'])
# Dropping x, y as it is a mapping of the latitude and longitude into coordinates based on the WGS 1984 Web Mercator (Auxiliary Sphere) projection
df = df.drop(columns=['x', 'y'])
# Dropping DRIVACT, DRIVCOND, PEDTYPE, PEDACT, PEDCOND as we only will examine the circumstances of the cyclist - they're blank once filtered by 'INVTYPE = Cyclist' anyways
df = df.drop(columns=['DRIVACT', 'DRIVCOND', 'PEDTYPE', 'PEDACT', 'PEDCOND'])
# Dropping VEHTYPE as it is made redundant by INVTYPE being Cyclist meaning that the only value is Bicycle
df = df.drop(columns=['VEHTYPE'])
# Dropping INJURY as it is made redundant by ACCLASS (Classification of Accident)
df = df.drop(columns=['INJURY'])
# Dropping CYCLISTYPE as it is too detailed (high cardinality) to generalize without significant extraction and over 50% of its data are NaN values which would be hard to handle if not impossible with imputation for text
df = df.drop(columns=['CYCLISTYPE'])
# Dropping IMPACTYPE due to low variability; over 93% is Cyclist Collisions as opposed to more descriptive values describing the collision like "Rear End" or "Sideswipe" - it would not have much predictive power
df = df.drop(columns=['IMPACTYPE'])
# Dropping CYCLIST as it is made redundant by INVTYPE being Cyclist so it necessarily involves a cyclist
df = df.drop(columns=['CYCLIST'])
# Dropping DISABILITY as all records contain the same value (blank, i.e., "No") and so does not provide any useful variation for modeling
df = df.drop(columns=['DISABILITY'])

# Simplifying TIME to be just the hour in 24H format
df['TIME'] = df['TIME'].astype(str).str.zfill(4)
df['TIME'] = pd.to_datetime(df['TIME'], format='%H%M').dt.hour

# Simplifying VISIBILITY into Clear or Not Clear
df['VISIBILITY'] = df['VISIBILITY'].replace({'Drifting Snow': 'Not Clear',
                                   'Other': 'Not Clear',
                                   'Rain': 'Not Clear',
                                   'Snow': 'Not Clear'})
# Convert to binary for modeling
df['VISIBILITY'] = df['VISIBILITY'].replace({'Not Clear': 1, 'Clear': 0})

# Simplifying the TRAFFCTL conditions
df['TRAFFCTL'] = df['TRAFFCTL'].replace({'No Control': 'No Control',
                                         'Pedestrian Crossover': 'Stop Control',
                                         'Stop Sign': 'Stop Control',
                                         'Streetcar (Stop for)': 'Stop Control',
                                         'Traffic Controller': 'Active Traffic Management',
                                         'Traffic Signal': 'Active Traffic Management'})

# Simplifying the LIGHT conditions
df['LIGHT'] = df['LIGHT'].replace({'Dark, artificial': 'Low Light',
                                   'Dawn, artificial': 'Low Light',
                                   'Daylight, artificial': 'Daylight',
                                   'Dusk, artificial': 'Low Light',
                                   'Dusk': 'Low Light',
                                   'Dawn': 'Low Light'})

# Simplifying the CYCCOND feature
df['CYCCOND'] = df['CYCCOND'].replace({'Ability Impaired, Alcohol': 'Impaired',
                                       'Ability Impaired, Alcohol Over .80': 'Impaired',
                                       'Ability Impaired, Drugs': 'Impaired',
                                       'Had Been Drinking': 'Impaired',
                                       'Fatigue': 'Poor Physical Condition',
                                       'Medical or Physical Disability': 'Poor Physical Condition',
                                       'Inattentive': 'Inattentive',
                                       'Normal': 'Normal',
                                       'Other': 'Other',
                                       'Unknown': 'Other',
                                       np.nan: 'Other'})

# Simplifying the CYCACT feature
df['CYCACT'] = df['CYCACT'].replace({'Disobeyed Traffic Control': 'Traffic Violations',
                                     'Failed to Yield Right of Way': 'Traffic Violations',
                                     'Wrong Way on One Way Road': 'Traffic Violations',
                                     'Following too Close': 'Unsafe Driving Behavior',
                                     'Improper Lane Change': 'Unsafe Driving Behavior',
                                     'Improper Passing': 'Unsafe Driving Behavior',
                                     'Improper Turn': 'Unsafe Driving Behavior',
                                     'Speed too Fast For Condition': 'Unsafe Driving Behavior',
                                     'Lost control': 'Loss of Control',
                                     'Other': 'Other',
                                     np.nan: 'Other'})

# Simplifying the ACCLOC feature
df['INTERSECTION'] = df['ACCLOC'].replace({'At Intersection': 1,
                                     'Intersection Related': 1,
                                     'At/Near Private Drive': 0,
                                     'Non Intersection': 0,
                                     'Overpass or Bridge': 0,
                                     np.nan: 0})
df = df.drop(columns=['ACCLOC'])  # Drop after since now represented by INTERSECTION

# Simplifying the MANOEUVER feature
df['MANOEUVER'] = df['MANOEUVER'].replace({'Changing Lanes': 'Lane Change / Merging',
                                           'Merging': 'Lane Change / Merging',
                                           'Overtaking': 'Lane Change / Merging',
                                           'Pulling Away from Shoulder or Curb': 'Lane Change / Merging',
                                           'Pulling Onto Shoulder or towardCurb': 'Lane Change / Merging',
                                           'Turning Left': 'Turning',
                                           'Turning Right': 'Turning',
                                           'Making U Turn': 'Turning',
                                           'Slowing or Stopping': 'Slowing or Stopped',
                                           'Stopped': 'Slowing or Stopped',
                                           'Going Ahead': 'Forwards',
                                           'Parked': 'Slowing or Stopped',
                                           'Other': 'Other',
                                           'Unknown': 'Other',
                                           np.nan: 'Other'
                                           })

# INVAGE is represented as ranges, so we'll take the midpoint instead for simplicity
def get_midpoint(age_range):
    try:
        low, high = map(int, age_range.split(' to '))
        return (low + high) / 2
    except ValueError:
        return None

# Replace the age range values with the midpoint
df['INVAGE'] = df['INVAGE'].apply(get_midpoint)

# Impute "unknown" INVAGE values with the median; the data doesn't have any other explicit missing/NaN values so there's no other NaN handling needed.
median_age = df['INVAGE'].median()
df['INVAGE'].fillna(median_age, inplace=True)

# Convert PEDESTRIAN, AUTOMOBILE, MOTORCYCLE, TRUCK, TRSN_CITY_VEH, EMERG_VEH, PASSENGER, SPEEDING, AG_DRIV, REDLIGHT, ALCOHOL, DISABILITY
# to binary columns as they only have "Yes" values where blank values are implicitly "No" by TPS' documentation
columns_to_binary = ['PEDESTRIAN', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL']
df[columns_to_binary] = df[columns_to_binary].applymap(lambda x: 1 if x == 'Yes' else 0)

# ACCLASS, our class label, has values of Fatal and Non-Fatal Injury; convert to binary for modeling
df['ACCLASS'] = df['ACCLASS'].replace({'Fatal': 1, 'Non-Fatal Injury': 0})

# In RDSFCOND (road surface condition), there are only 4 records with Loose Snow, 4 with Other, and 2 with Slush,
# hence filter out since they represent less than 0.5% of data
df = df[df['RDSFCOND'].isin(['Dry', 'Wet'])]
# Convert to binary for modeling
df['RDSFCOND'] = df['RDSFCOND'].replace({'Wet': 1, 'Dry': 0})

# Only 1 record containing Major Shoreline and 8 NaN values, so filtering them out
df = df[(df['ROAD_CLASS'] != 'Major Shoreline') & pd.notna(df['ROAD_CLASS'])]

# Only 12 records with NaN values in DISTRICT, so filtering them out
df = df[pd.notna(df['DISTRICT'])]

# Only 6 records with NaN values in TRAFFCTL, so filtering them out
df = df.dropna(subset=['TRAFFCTL'])

# We want to only look at cyclists' outcomes so filter by the role of the person in the collision, INVTYPE, to cyclists only
df_cyclist = df[df['INVTYPE'] == 'Cyclist']
df_cyclist = df_cyclist.drop(columns=['INVTYPE'])  # Drop after since it's now all just Cyclist value

categories = []
for col, col_type in df_cyclist.dtypes.items():
     if col_type == 'O':
          categories.append(col)
     else:
          df_cyclist[col].fillna(0, inplace=True)
print(categories)
print(df_cyclist.columns.values)
print(df_cyclist.head())
df_cyclist.describe()
df_cyclist.info()

print(len(df_cyclist) - df_cyclist.count())

