# ================================
# Airbnb NYC 2019 Data Cleaning
# ================================

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np

# 2️⃣ Load Dataset
df = pd.read_csv("C:\OASIS\PROJECT3\AB_NYC_2019.csv")

print("Initial Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())


# 1️⃣ Data Integrity Check
print("===== Data Integrity Check =====")
print("Initial Dataset Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nStatistical Summary (Price & Minimum Nights):")
print(df[['price','minimum_nights']].describe())

# Check invalid values
invalid_price = df[df['price'] <= 0].shape[0]
invalid_nights = df[df['minimum_nights'] <= 0].shape[0]
print(f"\nInvalid price values removed: {invalid_price}")
print(f"Invalid minimum_nights removed: {invalid_nights}")

# 2️⃣ Missing Data Handling
print("\n===== Missing Data Handling =====")
missing_before = df.isnull().sum()
print("Missing values before handling:\n", missing_before)

# Fill missing reviews_per_month
if 'reviews_per_month' in df.columns:
    df['reviews_per_month'].fillna(df['reviews_per_month'].mean(), inplace=True)

# Fill missing last_review
if 'last_review' in df.columns:
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    df['last_review'].fillna(method='ffill', inplace=True)

missing_after = df.isnull().sum()
print("\nMissing values after handling:\n", missing_after)

# 3️⃣ Duplicate Removal
print("\n===== Duplicate Removal =====")
duplicates = df.duplicated().sum()
print("Number of duplicate rows found:", duplicates)

# Remove duplicates
df.drop_duplicates(inplace=True)
print("Shape after removing duplicates:", df.shape)

# 6️⃣ STANDARDIZATION

print("===== Standardization =====")

# 1. Text Columns: lowercase and strip spaces
text_columns = df.select_dtypes(include=['object']).columns
for col in text_columns:
    df[col] = df[col].str.lower().str.strip()
print("Text columns standardized (lowercase & stripped).")

# 2. Date Columns: ensure datetime type
if 'last_review' in df.columns:
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    print("Date columns standardized to datetime format.")

# 3. Numeric Columns (Optional): ensure correct numeric type
numeric_columns = df.select_dtypes(include=['int64','float64']).columns
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
print("Numeric columns standardized.")

# 4. Units Consistency (Optional Example)
# For Airbnb, if needed, e.g., convert price to float (already numeric)
df['price'] = df['price'].astype(float)
print("Price column standardized as float.")

# 7️⃣ OUTLIER DETECTION & REMOVAL

print("===== Outlier Detection & Removal =====")

def remove_outliers(data, column):
    """
    Removes outliers from a column using the IQR method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Log number of outliers
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    print(f"Column '{column}': {outliers.shape[0]} outliers removed.")
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Remove outliers from 'price'
df = remove_outliers(df, 'price')

# (Optional) Remove outliers from 'minimum_nights' if needed
df = remove_outliers(df, 'minimum_nights')

print("Shape after outlier removal:", df.shape)

# =========================================
# 9️⃣ SAVE CLEANED DATA
# =========================================
output_path = "AB_NYC_2019_Cleaned.csv"  # You can change the path as needed

df.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved successfully at: {output_path}")
