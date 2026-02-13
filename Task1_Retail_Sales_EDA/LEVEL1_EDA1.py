import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.getcwd())

# Load dataset
df = pd.read_csv(r"C:\OASIS\PROJECT1\menu.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# ============================================
# 3️⃣ DATA CLEANING
# ============================================

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert Date column (if exists)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

# Fill numeric missing values with mean
for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].mean(), inplace=True)

print("\nData Info After Cleaning:")
print(df.info())

# ============================================
# 4️⃣ DESCRIPTIVE STATISTICS
# ============================================

print("\nSummary Statistics:")
print(df.describe())

if 'Sales' in df.columns:
    print("\nMean Sales:", df['Sales'].mean())
    print("Median Sales:", df['Sales'].median())
    print("Mode Sales:", df['Sales'].mode()[0])
    print("Standard Deviation:", df['Sales'].std())

# ============================================
# 5️⃣ TIME SERIES ANALYSIS
# ============================================

if 'Date' in df.columns and 'Sales' in df.columns:
    monthly_sales = df.groupby('Month')['Sales'].sum()

    print("\nMonthly Sales:\n", monthly_sales)

# ============================================
# 6️⃣ CUSTOMER & PRODUCT ANALYSIS
# ============================================

# Sales by Category
if 'Category' in df.columns and 'Sales' in df.columns:
    category_sales = df.groupby('Category')['Sales'].sum()
    print("\nSales by Category:\n", category_sales)

# Top 5 Products
if 'Product' in df.columns and 'Sales' in df.columns:
    top_products = df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(5)
    print("\nTop 5 Products:\n", top_products)

# Customer Analysis (if Customer column exists)
if 'Customer_ID' in df.columns:
    customer_purchase = df.groupby('Customer_ID')['Sales'].sum().sort_values(ascending=False).head(5)
    print("\nTop 5 Customers:\n", customer_purchase)

# ============================================
# 7️⃣ VISUALIZATION (ALL IN ONE PAGE)
# ============================================

plt.figure(figsize=(18,5))

# =========================
# 1️⃣ BAR CHART - Top 5 Highest Calorie Items
# =========================
plt.subplot(1,3,1)

top_calories = df.sort_values(by='Calories', ascending=False).head(5)

plt.bar(top_calories['Item'], top_calories['Calories'])
plt.xticks(rotation=45)
plt.title("Top 5 High Calorie Items")
plt.ylabel("Calories")

# =========================
# 2️⃣ LINE PLOT - Calories vs Protein
# =========================
plt.subplot(1,3,2)

plt.plot(df['Calories'], label='Calories')
plt.plot(df['Protein'], label='Protein')
plt.title("Calories vs Protein Trend")
plt.legend()

# =========================
# 3️⃣ HEATMAP - Correlation
# =========================
plt.subplot(1,3,3)

sns.heatmap(df.corr(numeric_only=True), annot=False)
plt.title("Correlation Heatmap")

plt.tight_layout()

print("\n========== BUSINESS RECOMMENDATIONS ==========")

print("✔ Promote high-rated menu items more aggressively.")
print("✔ Remove or redesign low-selling dishes.")
print("✔ Adjust pricing for items with low demand.")
print("✔ Introduce combo offers for popular categories.")
print("✔ Improve menu design to highlight best dishes.")

plt.show()



