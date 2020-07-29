import pandas as pd

data = pd.read_csv('final_dataset.csv')

for index, row in data.iterrows():
    if(row['FTHG']<row['FTAG']):
        print(str(row['FTHG'])+"    "+str(row['FTAG'])+"    "+row['FTR'])
        update_row = pd.DataFrame({'FTR':['A']}, index=[index])
        data.update(update_row)

for index, row in data.iterrows():
    if(row['FTHG'] == row['FTAG']):
        print(str(row['FTHG'])+"    "+str(row['FTAG'])+"    "+row['FTR'])
        update_row = pd.DataFrame({'FTR':['D']}, index=[index])
        data.update(update_row)

print(data.head())
data.to_csv('final_final_dataset.csv')