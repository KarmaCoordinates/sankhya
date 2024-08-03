import pandas as pd

df = pd.read_csv('.static/sankhya-karika-sanskrit-word-definitions.csv').astype(str)

# stack all columns
cols = [df[col].squeeze() for col in df]
df_col = pd.concat(cols, ignore_index=True)

# remove NaN
list_wo_na = [x for x in df_col if str(x) != 'nan']

# list to DF
df_wo_na = pd.DataFrame(list_wo_na, columns=['Word-Meaning'])

# split column 1 on '-' to series
ser_wo_na = df_wo_na['Word-Meaning'].str.split('-', n=1)

# series to multi-column DF
df_wo_na = pd.DataFrame(ser_wo_na.to_list(), columns=['sanskrit-word', 'english-synonyms'])

df_wo_na = df_wo_na.drop_duplicates()

df_wo_na.to_csv('.static/sankhya-karika-sanskrit-word-definitions-vector.csv')
