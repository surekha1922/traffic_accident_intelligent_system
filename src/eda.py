import pandas as pd
df = pd.read_csv("/home/rgukt/traffic_accident_intelligent_system/data/US_Accidents_March23.csv",low_memory = False,
                 on_bad_lines = "skip",
                 nrows = 100000)
drop_cols = [
    'ID', 'Source', 'Description', 'Street', 'City', 'County',
    'Zipcode', 'Country', 'Timezone', 'Airport_Code',
    'End_Time', 'End_Lat', 'End_Lng'
]

df = df.drop(columns=drop_cols)
drop_cols_2 = [
    'Weather_Timestamp', 'Wind_Chill(F)', 'Wind_Direction',
    'State', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'
]

df = df.drop(columns=drop_cols_2)
#more missing - so 0's
df['Precipitation(in)'].fillna(0, inplace=True)
#Moderate missing-median
df['Wind_Speed(mph)'].fillna(df['Wind_Speed(mph)'].median(), inplace=True)
#Small missing - fill with median
df['Humidity(%)'].fillna(df['Humidity(%)'].median(), inplace=True)
df['Visibility(mi)'].fillna(df['Visibility(mi)'].median(), inplace=True)
df['Temperature(F)'].fillna(df['Temperature(F)'].median(), inplace=True)
df['Pressure(in)'].fillna(df['Pressure(in)'].median(), inplace=True)
#Categorical columns-fill with mode
df['Weather_Condition'].fillna(df['Weather_Condition'].mode()[0], inplace=True)
df['Sunrise_Sunset'].fillna(df['Sunrise_Sunset'].mode()[0], inplace=True)
def simplify_weather(condition):
    condition = str(condition).lower()
    
    if 'rain' in condition:
        return 'Rain'
    elif 'snow' in condition:
        return 'Snow'
    elif 'fog' in condition or 'mist' in condition:
        return 'Fog'
    elif 'clear' in condition:
        return 'Clear'
    elif 'cloud' in condition:
        return 'Cloud'
    else:
        return 'Other'

df['Weather_Condition'] = df['Weather_Condition'].apply(simplify_weather)

df['Start_Time'] = pd.to_datetime(df['Start_Time'])

df['hour'] = df['Start_Time'].dt.hour
df['day'] = df['Start_Time'].dt.day
df['month'] = df['Start_Time'].dt.month

df['is_night'] = df['Sunrise_Sunset'].apply(lambda x: 1 if x == 'Night' else 0)
df = df.drop(columns=['Start_Time', 'Sunrise_Sunset'])

df = pd.get_dummies(df, columns=['Weather_Condition'], drop_first=True)
bool_cols = df.select_dtypes(include='bool').columns

df[bool_cols] = df[bool_cols].astype(int)
print(df.dtypes)
