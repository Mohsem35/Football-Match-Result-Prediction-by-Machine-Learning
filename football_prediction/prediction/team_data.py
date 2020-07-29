import pandas as pd


hometeamlist = {}
for index, row in data.iterrows():
    #print(str(index)+"   "+row['HomeTeam']+"   "+row['AwayTeam'])
    hometeamlist[index] = row['HomeTeam']