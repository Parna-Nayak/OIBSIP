import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(r"C:\OASIS\PROJECT1\sales.csv")

print(df.head())

# ============================================================
# 2. DATA CLEANING
# ============================================================

# Remove missing values
df.dropna(inplace=True)

# Clean column names
df.columns = df.columns.str.strip().str.lower()

print("\nColumns After Cleaning:")
print(df.columns)

# ============================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================

print("\n========== DESCRIPTIVE STATISTICS ==========")
print("Mean:\n", df.mean(numeric_only=True))
print("Median:\n", df.median(numeric_only=True))
print("Mode:\n", df.mode().iloc[0])
print("Standard Deviation:\n", df.std(numeric_only=True))

# ============================================================
# 4. DEFINE IMPORTANT COLUMNS
# ============================================================

date_column = 'date'
sales_column = 'total amount'
category_column = 'product category'

# Convert date column
df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
df.dropna(subset=[date_column], inplace=True)

# Create month column
df['month'] = df[date_column].dt.month

# ============================================================
# 5. PREPARE DATA FOR VISUALIZATION
# ============================================================

sales_trend = df.groupby(date_column)[sales_column].sum()
category_sales = df.groupby(category_column)[sales_column].sum()

pivot_table = df.pivot_table(
    values=sales_column,
    index=category_column,
    columns='month',
    aggfunc='sum'
)

# ============================================================
# 6. CREATE SINGLE DASHBOARD (3 GRAPHS IN ONE PAGE)
# ============================================================

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24,6))

# ---- 1️⃣ LINE PLOT ----
axes[0].plot(sales_trend)
axes[0].set_title("Sales Trend Over Time")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Total Sales")
axes[0].tick_params(axis='x', rotation=45)

# ---- 2️⃣ BAR CHART ----
axes[1].bar(category_sales.index, category_sales.values)
axes[1].set_title("Sales by Product Category")
axes[1].set_xlabel("Category")
axes[1].set_ylabel("Total Sales")
axes[1].tick_params(axis='x', rotation=45)

# ---- 3️⃣ HEATMAP ----
sns.heatmap(pivot_table, annot=True, ax=axes[2])
axes[2].set_title("Sales Heatmap (Category vs Month)")

# Dashboard Title
fig.suptitle("Retail Sales Analysis Dashboard", fontsize=16)

plt.tight_layout()
plt.show()

# ============================================================
# 7. BUSINESS RECOMMENDATIONS
# ============================================================

print("\n========== BUSINESS RECOMMENDATIONS ==========")
print("✔ Focus marketing on high-performing product categories.")
print("✔ Increase stock during peak sales months.")
print("✔ Provide targeted promotions based on customer trends.")
print("✔ Improve strategy for low-performing categories.")
print("✔ Use monthly trends for inventory planning.")
