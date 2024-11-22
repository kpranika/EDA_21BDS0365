# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the given path
data = pd.read_csv(r'C:\Users\Pranika Kumar\Downloads\Suicide data.csv')

# Check for missing values
print("\n=== Missing Values ===")
print(data.isnull().sum())

# Drop columns with more than 15% missing values
threshold = 0.15 * len(data)
data_cleaned = data.loc[:, data.isnull().sum() < threshold]

# Separate numeric and non-numeric columns
numeric_cols = data_cleaned.select_dtypes(include=['number']).columns
non_numeric_cols = data_cleaned.select_dtypes(exclude=['number']).columns

# Fill missing values for numeric columns with the mean
data_cleaned[numeric_cols] = data_cleaned[numeric_cols].fillna(data_cleaned[numeric_cols].mean())

# Display the cleaned data (first 5 rows)
print("\n=== Cleaned Data (Preview) ===")
data_cleaned.head()

# ==========================
# Visualization Techniques
# ==========================

# Plot histogram for 'suicides_no' (numeric column)
if 'suicides_no' in numeric_cols:
    plt.figure(figsize=(8, 6))
    sns.histplot(data_cleaned['suicides_no'], kde=True)
    plt.title('Histogram of Suicides')
    plt.xlabel('Number of Suicides')
    plt.ylabel('Frequency')
    plt.show()

# Scatter plot for 'suicides_no' vs 'population' (both numeric columns)
if 'suicides_no' in numeric_cols and 'population' in numeric_cols:
    plt.figure(figsize=(8, 6))
    plt.scatter(data_cleaned['suicides_no'], data_cleaned['population'])
    plt.title('Suicides vs Population')
    plt.xlabel('Number of Suicides')
    plt.ylabel('Population')
    plt.show()

# Box plot for suicide rates by 'generation' (categorical column)
if 'generation' in non_numeric_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data_cleaned, x='generation', y='suicides/100k pop')
    plt.title('Suicide Rate by Generation')
    plt.xlabel('Generation')
    plt.ylabel('Suicides per 100k Population')
    plt.xticks(rotation=45)
    plt.show()

# Heatmap for correlation matrix (if there are enough numeric columns)
corr_matrix = data_cleaned[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


