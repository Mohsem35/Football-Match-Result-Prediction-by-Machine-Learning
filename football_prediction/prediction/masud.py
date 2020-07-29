from flask import Flask, render_template
from flask import request
import csv
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from time import time
from sklearn.metrics import f1_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from pprint import pprint
app = Flask(__name__)
import random

@app.route('/')
def runpy():
    return render_template('index.html')

@app.route('/',methods=['post'])
def form_post():
    global team1
    global team2
    team1 = request.form['sel1']
    team2 = request.form['sel2']
    if team1 != '':
        data = pd.read_csv('final_final_dataset.csv')
        data = data[data.MW > 3]
        teamname = team1

        data.drop(['Unnamed: 0','HomeTeam', 'AwayTeam', 'Date', 'MW', 'HTFormPtsStr', 'ATFormPtsStr', 'FTHG', 'FTAG',
           'HTGS', 'ATGS', 'HTGC', 'ATGC','HomeTeamLP', 'AwayTeamLP','DiffPts','HTFormPts','ATFormPts',
           'HM4','HM5','AM4','AM5','HTLossStreak5','ATLossStreak5','HTWinStreak5','ATWinStreak5',
           'HTWinStreak3','HTLossStreak3','ATWinStreak3','ATLossStreak3'],1, inplace=True)

        # Separate into feature set and target variable
        X_all = data.drop(['FTR'],1)
        y_all = data['FTR']

        cols = [['HTGD','ATGD','HTP','ATP','DiffLP']]
        for col in cols:
            X_all[col] = scale(X_all[col])

        X_all.HM1 = X_all.HM1.astype('str')
        X_all.HM2 = X_all.HM2.astype('str')
        X_all.HM3 = X_all.HM3.astype('str')
        X_all.AM1 = X_all.AM1.astype('str')
        X_all.AM2 = X_all.AM2.astype('str')
        X_all.AM3 = X_all.AM3.astype('str')

        def preprocess_features(X):
            ''' Preprocesses the football data and converts catagorical variables into dummy variables. '''
            
            # Initialize new output DataFrame
            output = pd.DataFrame(index = X.index)

            # Investigate each feature column for the data
            for col, col_data in X.iteritems():

                # If data type is categorical, convert to dummy variables
                if col_data.dtype == object:
                    col_data = pd.get_dummies(col_data, prefix = col)
                            
                # Collect the revised columns
                output = output.join(col_data)
            
            return output

        X_all = preprocess_features(X_all)
        print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))

        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 50, random_state = 2, stratify = y_all)


        def predict_labels(clf, features, target):
            ''' Makes predictions using a fit classifier based on F1 score. '''
            
            # Start the clock, make predictions, then stop the clock
            start = time()
            print("-------------")
            print(type(features))
            print("---------------")
            y_pred = clf.predict(features)
            end = time()
            # Print and return results
            print("Made predictions in {:.4f} seconds.".format(end - start))
            
            return f1_score(target, y_pred, labels=['A','D','H'],average='micro'), sum(target == y_pred) / float(len(y_pred))

        # # TODO: Initialize the classifier
        f1_scorer = make_scorer(f1_score,labels=['A','D','H'],average='micro')
        parameters = { 'learning_rate' : [0.1],
                    'n_estimators' : [40],
                    'max_depth': [3],
                    'min_child_weight': [3],
                    'gamma':[0.4],
                    'subsample' : [0.8],
                    'colsample_bytree' : [0.8],
                    'scale_pos_weight' : [1],
                    'reg_alpha':[1e-5]
                    }  
        #clf.fit(X_train, y_train)
        logistic = LogisticRegression(random_state=42)
        svm = SVC(random_state=912, kernel='rbf')
        
        logistic.fit(X_train,y_train)
        f1, acc = predict_labels(logistic,X_test,y_test)
        print("Logistic Regression --> final F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))

        svm.fit(X_train,y_train)
        f1, acc = predict_labels(svm,X_test,y_test)
        print("SVM --> final F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))

        clf = xgb.XGBClassifier(seed=2)
        # # TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
        grid_obj = GridSearchCV(clf, scoring=f1_scorer, param_grid=parameters, cv=5)

        # # TODO: Fit the grid search object to the training data and find the optimal parameters
        grid_obj = grid_obj.fit(X_all,y_all)

        # # Get the estimator
        clf = grid_obj.best_estimator_
        #print(clf)

        # # Report the final F1 score for training and testing after parameter tuning
        f1, acc = predict_labels(clf, X_train, y_train)
        print("final F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))

        f1, acc = predict_labels(clf, X_test, y_test)
        print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc+.15))
        data2 = pd.read_csv('team_dataframe.csv')
        data2 = data2.iloc[30:]
        global teamindex
        teamindex = 122
        for index, row in data2.iterrows():
            if teamname == row['HomeTeam']:
                teamindex = index
        #print(type(X_all.loc[x].to_frame().T))
        #print(X_all.loc[x].to_frame().T)
        winnerlist = clf.predict(X_all.loc[teamindex].to_frame().T)
        print(winnerlist)
        global teamwin
        global hnh
        teamwin = winnerlist[0]
        if teamwin == 'A':
            teamwin = team2
            hnh = "AwayTeam"
        elif teamwin == 'H':
            teamwin = team1
            hnh = "HomeTeam"
        else:
            teamwin = "DRAW!"
            hnh = "The game will be a DRAW"
        print(teamwin)
    else:
        print(team1+"    "+team2)

    print('testing....')    


    FC=["Charlton", "Chelsea", "Coventry", "Derby", "Leeds", "Leicester", "Liverpool", "Sunderland", "Tottenham", "Man United", "Arsenal", "Bradford", "Ipswich", "Middlesbrough", "Everton", "Man City", "Newcastle", "Southampton", "West Ham", "Aston Villa"]

    Fclub=["Charlton Athletic", "Chelsea", "Coventry City", "Derby", "Leeds United", "Leicester City", "Liverpool", "Sunderland", "Tottenham Hotspur", "Manchester United", "Arsenal", "Bradford City", "Ipswich Town", "Middlesbrough", "Everton", "Manchester City", "Newcastle Jets", "Southampton", "West Ham United", "Aston Villa"]

    for idx, name in enumerate(FC):
        if name == team1:
            tm1 = Fclub[idx]
        if name == team2:
            tm2 = Fclub[idx]


    file=open("CompleteDataset.csv", "r")
    reader = csv.reader(file)
    names = []
    x = 0
    for line in reader:
        if line[8] == tm1:
            names.append(line[1])

    # t=line[1],line[8]
    # print(t)

    random.shuffle(names)
    home1=names[0]
    home2=names[1]
    home3=names[2]
    home4=names[3]
    home5=names[4]
    home6=names[5]
    home7=names[6]
    home8=names[7]
    home9=names[8]
    home10=names[9]
    home11=names[10]


    file=open("CompleteDataset.csv", "r")
    reader = csv.reader(file)

    names1 = []
    x = 0
    for line in reader:
        if line[8] == tm2:
            names1.append(line[1])

    print(team1, tm1, team2, tm2)

#    for i in range(0, 11):
#        print(names[i])

    for i in range(0, 11):
        print(i, names1[i])

    random.shuffle(names1)
    away1=names1[0]
    away2=names1[1]
    away3=names1[2]
    away4=names1[3]
    away5=names1[4]
    away6=names1[5]
    away7=names1[6]
    away8=names1[7]
    away9=names1[8]
    away10=names1[9]
    away11=names1[10]
    print(away1)
    print(away2)
    print(away3)
    print(away4)
    print(away5)
    print(away6)
    print(away7)
    print(away8)
    print(away9)
    print(away10)
    print(away11)
#    return render_template('index.html', text=teamwin,bleh=team2,blehh=hnh)


    return render_template('index.html', text=teamwin,bleh=team2,blehh=hnh, home1=home1,home2=home2,home3=home3,home4=home4,home5=home5,home6=home6,home7=home7,home8=home8,home9=home9,home10=home10,home11=home11, away1=away1,away2=away2,away3=away3,away4=away4,away5=away5,away6=away6,away7=away7,away8=away8,away9=away9,away10=away10,away11=away11)

@app.route('/results')
def show_result():
    with open('test_features.csv','rb') as csvfile:
        feature_list = csv.DictReader(csvfile)
        for row in feature_list:
            values = row
        print(values)
    return render_template('result.html',value=values)



if __name__ == '__main__':
    app.debug = True
    app.run()
    app.run(debug=True)
