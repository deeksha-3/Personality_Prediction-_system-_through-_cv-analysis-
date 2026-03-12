import pandas as pd

df = pd.read_csv('personality_dataset.csv')
print(df.head())
print(df.columns)
print("Unique values per column:")
for col in df.columns[:-1]:
    print(col, df[col].unique())
