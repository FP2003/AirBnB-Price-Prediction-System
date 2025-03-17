# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# %%
# Load the dataset
df = pd.read_csv('London_Listings.csv')

# Original shape
print('Original Shape:', df.shape)

# Display the first few rows
df.head()

# %%
# Check types before categorizing
print(df.dtypes)

# %%
# Separate categorical and numerical columns
categorical_var = df.select_dtypes(include=['object']).columns.tolist()
numerical_var = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Categorical Variables:", categorical_var)
print("Numerical Variables:", numerical_var)

# %%
# Clean the price column
df['price'] = df['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Describe price
df['price'].describe()

# %%
# Replace empty strings or lists in categorical variables with NaN
for col in categorical_var:
    df[col] = df[col].replace(['', '[]'], np.nan)
    
# Summary of missing values
missing_col_val = df.isnull().sum()
print('Missing column values:\n', missing_col_val[missing_col_val > 0])

# %%
# Drop unnecessary columns
drop_columns = ['calendar_last_scraped', 'bathrooms_text', 'latitude', 'longitude']
df.drop(columns=drop_columns, inplace=True, errors='ignore')

# Updated shape
print("Shape after dropping unnecessary columns:", df.shape)

# %%
# Drop duplicates
df.drop_duplicates(inplace=True)

# Drop rows with NaN values
df.dropna(inplace=True)

# Updated shape
print('Shape after dropping duplicates and NaN rows:', df.shape)

# %%
# Remove outliers using IQR method
def remove_outliers(df, columns):
    for col in columns:
        if col not in df.columns:
            print(f"Column '{col}' does not exist. Skipping.")
            continue
        
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

# Apply outlier removal
df = remove_outliers(df, numerical_var)

# Updated shape
print('Shape after removing outliers:', df.shape)

# %%
# Log transformation of prices
df = df[df['price'] > 0]  # Ensure no zero or negative prices
df['price_log'] = np.log1p(df['price'])

# %%
# Update numerical variable list (remove 'price' and use 'price_log')
numerical_var = [
    'accommodates', 'bedrooms', 'beds',
    'price_log', 'number_of_reviews', 'review_scores_rating'
]

# Create a copy before normalization
df_original = df.copy()

# Normalize numerical features
scaler = StandardScaler()
df[numerical_var] = scaler.fit_transform(df[numerical_var])

# Display updated dataframe
print(df.head())

# %%
# Visualization: Log Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df_original['price'], bins=50, kde=True, color='blue', label='Original Price')
sns.histplot(df_original['price_log'], bins=50, kde=True, color='red', label='Log Price')
plt.legend()
plt.title('Distribution of Original vs Log-Transformed Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# %%
# Visualization: Price Across Neighborhoods (Filtered)
neighbourhood_counts = df_original['neighbourhood'].value_counts()
valid_neighbourhoods = neighbourhood_counts[neighbourhood_counts >= 10].index
df_filtered = df_original[df_original['neighbourhood'].isin(valid_neighbourhoods)]

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_filtered, x='neighbourhood', y='price_log')
plt.xticks(rotation=90)
plt.title('Price Distribution Across Neighborhoods (Filtered)')
plt.xlabel('Neighborhood')
plt.ylabel('Log Price')
plt.tight_layout()
plt.show()

# %%
# Visualization: Price by Number of Tenants (Accommodates)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_original, x='accommodates', y='price_log')
plt.title('Price Distribution by Number of Tenants')
plt.xlabel('Number of Tenants (Accommodates)')
plt.ylabel('Log Price')
plt.tight_layout()
plt.show()

# %%
# Visualization: Relationship Between Review Scores and Prices
plt.figure(figsize=(12, 6))
sns.regplot(
    data=df_original,
    x='review_scores_rating',
    y='price_log',
    scatter_kws={'alpha': 0.3},
    line_kws={'color': 'red'}
)
plt.title('Relationship Between Review Scores and Prices')
plt.xlabel('Review Scores (Rating)')
plt.ylabel('Log Price')
plt.tight_layout()
plt.show()
