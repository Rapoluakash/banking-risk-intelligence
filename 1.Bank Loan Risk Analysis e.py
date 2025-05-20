# Bank Loan Risk Analysis - Cleaned Code



# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Settings
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

# Load the dataset
app = pd.read_csv('application_data.csv')  # <-- Update path if needed

# Initial Data Check
print(app.shape)
print(app.dtypes)
app.head()

# Missing Values Analysis
missing_val_app = app.isnull().sum()
missing_perc_app = pd.DataFrame({
    'Columns': missing_val_app.index,
    'Percentage': (missing_val_app.values / app.shape[0]) * 100
})

# Plot missing values
plt.figure(figsize=(20, 6))
sns.pointplot(data=missing_perc_app, x='Columns', y='Percentage')
plt.xticks(rotation=90)
plt.axhline(50, color='red', linestyle='--')
plt.title('Missing Value % per Column')
plt.show()

# Drop columns with >50% missing
missing_more_50 = missing_perc_app[missing_perc_app['Percentage'] >= 50]
app1 = app.drop(columns=missing_more_50['Columns'].to_list())

# Optional: Drop less useful or redundant columns
drop_cols = [
    'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
    'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',
    'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
    'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
    'DAYS_LAST_PHONE_CHANGE'
]
app1 = app1.drop(columns=[col for col in drop_cols if col in app1.columns])

# Impute missing values (example)
app1['AMT_GOODS_PRICE'].fillna(app1['AMT_GOODS_PRICE'].median(), inplace=True)
app1['NAME_TYPE_SUITE'].fillna('Unknown', inplace=True)

# Create Age Range Feature
app1['DAYS_BIRTH'] = app1['DAYS_BIRTH'].abs()
app1['Age'] = (app1['DAYS_BIRTH'] / 365).round(1)
bins = [0, 30, 40, 50, 60, 100]
labels = ['<30', '30-40', '40-50', '50-60', '60+']
app1['AGE_RANGE'] = pd.cut(app1['Age'], bins=bins, labels=labels)

# Check target distribution
print(app1['TARGET'].value_counts(normalize=True) * 100)
sns.countplot(x='TARGET', data=app1)
plt.title('Target Class Distribution')
plt.show()

# Correlation heatmap (optional)
plt.figure(figsize=(15, 10))
sns.heatmap(app1.corr(numeric_only=True), cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap')
plt.show()

# Save cleaned data (optional)
app1.to_csv('application_data_cleaned.csv', index=False)

print("\nData cleaning complete. Ready for modeling!")
