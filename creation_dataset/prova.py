import pandas as pd

# Assuming your data is stored in a CSV file with headers
# Reading the data into a DataFrame
df = pd.read_csv('training_set.csv', header=None, names=['filename', 'upper', 'lower', 'gender', 'bag', 'hat'])
print("values null", df.isnull())
# Convert the 'color' column to categorical
df['upper'] = df['upper'].astype('category')
df['lower'] = df['lower'].astype('category')

# Perform one-hot encoding on the 'color' column
one_hot_encoded_color = pd.get_dummies(df, columns=['upper', 'lower'])
#print(one_hot_encoded_color.head())
df = pd.DataFrame(one_hot_encoded_color)
print(df.isnull().sum())
# Concatenate the one-hot encoded color columns with the original DataFrame
#df_encoded = pd.concat(one_hot_encoded_color, axis=1)

# Drop the original 'color' column if needed
#one_hot_encoded_color.drop(['upper', 'lower'], axis=1, inplace=True)

#df_encoded.drop('lower', axis=1, inplace=True)
# Display the DataFrame with one-hot encoded color
print(one_hot_encoded_color.head())
