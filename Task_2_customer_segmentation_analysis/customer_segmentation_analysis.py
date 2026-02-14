# ============================================
# CUSTOMER SEGMENTATION ANALYSIS PROJECT
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ============================================
# 1. DATA COLLECTION
# ============================================

# Load Dataset
df = pd.read_csv("C:\OASIS\PROJECT2/ifood_df.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nMissing Values:\n", df.isnull().sum())

# ============================================
# 2. DATA CLEANING
# ============================================

# Drop duplicates
df = df.drop_duplicates()

# Fill missing numeric values with median
num_cols = df.select_dtypes(include=['int64','float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Convert date column if exists
if 'Dt_Customer' in df.columns:
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

print("\nFinal Dataset Shape After Cleaning:", df.shape)

# ============================================
# 3. FEATURE ENGINEERING
# ============================================

# Total Spending
spending_cols = ['MntWines','MntFruits','MntMeatProducts',
                 'MntFishProducts','MntSweetProducts','MntGoldProds']

df['Total_Spending'] = df[spending_cols].sum(axis=1)

# Total Purchases
purchase_cols = ['NumWebPurchases','NumCatalogPurchases','NumStorePurchases']
df['Total_Purchases'] = df[purchase_cols].sum(axis=1)

# ============================================
# 4. DESCRIPTIVE STATISTICS
# ============================================

print("\nAverage Purchase Value:", df['Total_Spending'].mean())
print("Average Purchase Frequency:", df['Total_Purchases'].mean())
print("\nRecency Summary:\n", df['Recency'].describe())

# ============================================
# 5. CUSTOMER SEGMENTATION (K-MEANS)
# ============================================

# Select Features
features = df[['Income','Total_Spending','Total_Purchases','Recency']]

# Scale Features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-Means (4 Clusters)
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

print("\nCluster Distribution:\n", df['Cluster'].value_counts())

# ============================================
# 6. VISUALIZATION (ALL GRAPHS IN ONE PAGE)
# ============================================

plt.figure(figsize=(12,10))

# 1. Income vs Spending
plt.subplot(2,2,1)
plt.scatter(df['Income'], df['Total_Spending'], c=df['Cluster'])
plt.xlabel("Income")
plt.ylabel("Total Spending")
plt.title("Income vs Spending")

# 2. Purchases vs Spending
plt.subplot(2,2,2)
plt.scatter(df['Total_Purchases'], df['Total_Spending'], c=df['Cluster'])
plt.xlabel("Total Purchases")
plt.ylabel("Total Spending")
plt.title("Purchases vs Spending")

# 3. Average Spending per Cluster
cluster_spending = df.groupby('Cluster')['Total_Spending'].mean()
plt.subplot(2,2,3)
cluster_spending.plot(kind='bar')
plt.title("Avg Spending per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Avg Spending")

# 4. Average Income per Cluster
cluster_income = df.groupby('Cluster')['Income'].mean()
plt.subplot(2,2,4)
cluster_income.plot(kind='bar')
plt.title("Avg Income per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Avg Income")

plt.tight_layout()
plt.show()

# ============================================
# END OF PROJECT
# ============================================
