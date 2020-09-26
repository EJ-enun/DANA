# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 04:44:44 2020

@author: EJ Bassey Enun
"""

#Derived Artificial Neural_Networks Algorithm(DANA) 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.compose import ColumnTransformer

#Load UFC Datasets for Derived Artificial Neural_Network Algorithm(DANA)
df_ufc_fights_X = pd.read_csv("C:/Users/USER/Desktop/Workspace/ML/UFC_Data/ALL_UFC_FIGHTS_2_23_2016.csv")
df_ufc_fighters = pd.read_csv("C:/Users/USER/Desktop/Workspace/ML/UFC_Data/ALL_UFC_FIGHTERS_2_23_2016.csv")
df_fights_ufc = pd.read_csv("C:/Users/USER/Desktop/Workspace/ML/UFC_Data/data.csv")
pre_proc_ufc = pd.read_csv("C:/Users/USER/Desktop/Workspace/ML/UFC_Data/preprocessed_data.csv")
ufc_fighters = pd.read_csv("C:/Users/USER/Desktop/Workspace/ML/UFC_Data/datasets_raw_fighter_details.csv")
ufc_combined = pd.read_csv("C:/Users/USER/Desktop/Workspace/ML/UFC_Data/DeepUFC2-master/ufc_combined.csv")
ufc_bouts = pd.read_csv("C:/Users/USER/Desktop/Workspace/ML/UFC_Data/DeepUFC2-master/ufc_bouts.csv")
ufc_fighters_combine = pd.read_csv("C:/Users/USER/Desktop/Workspace/ML/UFC_Data/DeepUFC2-master/ufc_fighters.csv")

def findFeatures(data_frame, array_val, search_term):
    df = pd.DataFrame(data_frame)
    for result in df[array_val]:
        if re.search(search_term, result):
            return df[df[array_val].isin([search_term])]

def divFunction(obj1, obj2):
    obj1 = np.array(obj1)
    obj2 = np.array(obj2)
    return obj1/obj2
#Abstract Methods
def find(data_frame, array_val, search_term):
    df = pd.DataFrame(data_frame)
    for result in df[array_val]:
        if re.search(result, search_term):
            return df[df[array_val].isin(search_term)]

def findFighter(data_frame, array_val, search_term):
    df = pd.DataFrame(data_frame)
    for result in df[array_val]:
        if re.search(search_term, result):
            return df[df[array_val].isin([search_term])]

        
def findFighterFights(data_frame, array_val, search_term):
    df = pd.DataFrame(data_frame) 
    for result in df[array_val]:
        if re.search(search_term, result):
            return df[df[array_val].isin([search_term])]

#Method to measure how much mpact size and atheticism is being utilized by a fighter and their opponent     
def checkAthleticism_Utilization(reach_diff):
 for r in reach_diff:
  if r >= 4 or r <= 5.5: 
      re = (1.125 * df_fights_ufc["B_avg_SIG_STR_landed"])- df_fights_ufc["B_avg_opp_SIG_STR_landed"]
      return re   
  elif r >= 5.6 or r <= 8:
      re =  (1.25 * df_fights_ufc["B_avg_SIG_STR_landed"]) - df_fights_ufc["B_opp_avg_SIG_STR_landed"]
      return re   
  elif r >= 8.1:
      re = (1.5 * df_fights_ufc["B_avg_SIG_STR_landed"]) - df_fights_ufc["B_opp_avg_SIG_STR_landed"]
      return re   



#Collate all fighters fights
fight_total = (df_fights_ufc["B_wins"] + df_fights_ufc["B_losses"])

#Build a train set with fighters with above 5 wins and losses combined
train_cv_test_set = df_fights_ufc.loc[fight_total >= 5]

fighter_fights_collated = (train_cv_test_set["B_wins"] + train_cv_test_set["B_losses"])



#Reach factors
reach_blue = df_fights_ufc["B_Reach_cms"]
reach_red = df_fights_ufc["R_Reach_cms"]
reach_diff = df_fights_ufc["B_Reach_cms"] - df_fights_ufc["R_Reach_cms"]
athleticism_utilization = checkAthleticism_Utilization(reach_diff)

#features = 18
#strikes landed per minute
slpm = divFunction(train_cv_test_set["R_avg_SIG_STR_landed"], train_cv_test_set["R_total_rounds_fought"]*5)
#strikes absorbed per minute
sapm = divFunction(train_cv_test_set["R_avg_opp_SIG_STR_landed"], train_cv_test_set["R_total_rounds_fought"]*5)
#percentage of overall significant strikes landed
stk_acc = (train_cv_test_set["R_avg_SIG_STR_landed"]/train_cv_test_set["R_avg_SIG_STR_att"])
#avg number of significant strikes missed by opponents
opp_stk_missed = train_cv_test_set["R_avg_opp_SIG_STR_att"] - train_cv_test_set["R_avg_opp_SIG_STR_landed"]
#pecentage of opponents overall strikes missed
stk_def = divFunction(opp_stk_missed, train_cv_test_set["R_avg_opp_SIG_STR_att"])
#average of strikes landed - average of opponents strikes landed
stk_diff = train_cv_test_set["R_avg_SIG_STR_landed"] - train_cv_test_set["R_avg_opp_SIG_STR_landed"]
#stk_raw_percentage = stk_diff/(df_fights_ufc["B_avg_SIG_STR_landed"] + df_fights_ufc["B_avg_opp_SIG_STR_landed"])
#difference between strkes landed and opponent strikes landed divided by total number of strikes landed and multiplied by 100 to get its %
stk_diff_percentage_rel_tot_landed = (stk_diff/(train_cv_test_set["R_avg_SIG_STR_landed"] + train_cv_test_set["R_avg_opp_SIG_STR_landed"]))
#strike differntal per minute of fight time
stk_diff_per_min = stk_diff/(train_cv_test_set["R_total_rounds_fought"]*5)
#percentage of strikes landed relative to opponents overall trikes landed
#stk_ratio = divFunction(df_fights_ufc["B_avg_SIG_STR_landed"], df_fights_ufc["B_avg_opp_SIG_STR_landed"])
#Average takedowns landed per 15 minutes
td_avg  = divFunction(train_cv_test_set["R_avg_TD_landed"], train_cv_test_set["R_total_rounds_fought"]*5)
#Percentage of average of takedowns landed from all takedown attempts
td_acc  =  divFunction(train_cv_test_set["R_avg_TD_landed"],train_cv_test_set["R_avg_TD_att"])
#average opponents takedowns defended
opp_td_missed = (train_cv_test_set["R_avg_opp_TD_att"] - train_cv_test_set["R_avg_opp_TD_landed"])
#percentage of average of oppinents takedown defended
td_def  = divFunction(opp_td_missed, train_cv_test_set["R_avg_opp_TD_att"])
#average submission attempts per 15 minutes
sub_avg = divFunction(train_cv_test_set["R_avg_opp_SUB_ATT"], train_cv_test_set["R_total_rounds_fought"]*5)
#Percentage of submissions completed
sub_acc =  divFunction(train_cv_test_set["R_win_by_Submission"], train_cv_test_set["R_avg_opp_SUB_ATT"])
#Will is same as stk_diff_per_min just not a percentage
will  = divFunction(stk_diff, train_cv_test_set["R_total_rounds_fought"]*5)
#Percentage of finishes from average significant strikes landed
fin_rate = ((train_cv_test_set["R_win_by_KO/TKO"] + train_cv_test_set["R_win_by_TKO_Doctor_Stoppage"])/train_cv_test_set["R_avg_SIG_STR_landed"])
#Percentage of Signficant Strikes relative to strikes landed
kickbox_acc = stk_acc
#Percentage of wins from average of submission attempts
grap_acc = divFunction(train_cv_test_set["R_win_by_Submission"], train_cv_test_set["R_avg_SUB_ATT"])
#Power 
power = divFunction((((train_cv_test_set["R_win_by_KO/TKO"] + train_cv_test_set["R_win_by_TKO_Doctor_Stoppage"])/train_cv_test_set["R_wins"])/train_cv_test_set["R_avg_KD"]), train_cv_test_set["R_avg_SIG_STR_landed"])
#Alternative power measurements
alt_power = divFunction((train_cv_test_set["R_win_by_KO/TKO"] + train_cv_test_set["R_win_by_TKO_Doctor_Stoppage"])/train_cv_test_set["R_wins"], train_cv_test_set["R_avg_KD"])
#number of wins
wins = train_cv_test_set["R_wins"]
#B_avg_KD = Knock Downs 
#Wins relative to opponents wins 
win_rel_opp = train_cv_test_set["R_wins"] - train_cv_test_set["B_wins"]



stk_features =  np.stack([stk_acc, stk_def, stk_diff, stk_diff_per_min, slpm, sapm, opp_stk_missed, kickbox_acc],  axis = 1)
td_features = np.stack([td_avg, td_acc, opp_td_missed, td_def], axis = 1)
sub_features = np.stack([sub_acc, sub_avg, grap_acc],  axis = 1)
other_features = np.stack([will, fin_rate, power, alt_power, wins],  axis = 1)
#train_test_features = np.stack([ fin_rate, grap_acc, opp_td_missed, power,  sapm, slpm, stk_acc, stk_def, stk_diff, sub_acc, sub_avg, td_avg, td_acc, td_def, will], axis=1)

y_feature = train_cv_test_set["Winner"].value_counts()

rep = {"Red":1, "Blue":0, "Draw":0}
inf = {"inf":0}
y_tester = train_cv_test_set["Winner"].replace(rep)
train_test_features = np.stack([ fin_rate, grap_acc, opp_td_missed, power,  sapm, slpm, stk_acc, stk_def, stk_diff, sub_acc, sub_avg, td_avg, td_acc, td_def, will, y_tester], axis=1)

train_cv_test_set.dtypes

obj_df = pd.DataFrame(train_test_features).fillna(0)
t_feature_set = obj_df.replace(np.inf, 0)
x_tester = t_feature_set

exp = x_tester.to_csv("C:/Users/USER/Desktop/ufc_dana.csv")

from sklearn.model_selection import train_test_split
from sklearn import linear_model

from sklearn import datasets
diabetes = datasets.load_diabetes()
x_train = diabetes.data[:-20]
y_train = diabetes.target[:-20]
x_test = diabetes.data[-20:]
y_test = diabetes.target[-20:]
plt.figure(figsize=(8,12))
for f in range(0,10):
    xi_test = x_test[:,f]
    xi_train = x_train[:,f]
    xi_test = xi_test[:,np.newaxis]
    xi_train = xi_train[:,np.newaxis]
linreg = linear_model.LinearRegression()
linreg.fit(xi_train,y_train)
y = linreg.predict(xi_test)
plt.subplot(5,2,f+1)
plt.scatter(xi_test,y_test,color='k')
plt.plot(xi_test,y,color='b',linewidth=3)

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix 
from sklearn.model_selection import KFold, cross_val_score,RandomizedSearchCV,GridSearchCV
from scipy.stats import uniform
model = linear_model.LogisticRegression()


x, x_test, y, y_test = train_test_split(x_tester, y_tester, test_size = 0.3)
model.fit(x,y)
y_pred = model.predict(x_test)




#cv = KFold(n_splits=5)
#for train_index, test_index in cv.split(x):
#    model.fit(x.iloc[train_index], y.iloc[train_index])
#    print(model.score(x.iloc[train_index], y.iloc[train_index]))
#    cross_val_score(model, x_tester,y_tester, cv=5)

#c = uniform(loc=0, scale=4)
#parameters ={"c": [0.001,0.01,0.1,0.5,1,5,10,100]}
#params_2 ={"c": c}
#cv_model = GridSearchCV(model, parameters)
#cv_model.fit(x, y)
#best_par = cv_model.best_params_
#cv_rand = RandomizedSearchCV(model, params_2)
#cv_rand.fit(x_tester, y_tester)
#best_est = cv_rand.best_estimator_
#best_score = cv_rand.best_score_

#y_pred = model.predict(x_test)
#results = np.stack([y_test, y_pred], axis = 1)
#res = pd.DataFrame(results)
#log_axis = plt.figure().add_subplot(1,1,1)
#log_axis.plot(y_pred, y_test, "x", color = "red")
#log_axis.set_title("Ypred and Ytest")
#log_axis.set_xlabel("Ypred")
#log_axis.set_ylabel("Ytest")

#obj_df[obj_df.isnull().any(axis=1)]
fights_tot = df_fights_ufc["B_total_rounds_fought"]/3
losses =  df_fights_ufc["B_losses"]
l=[]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y_feature) 
lister = list(le.classes_)
le.transform(y_feature)

#Other factors
#out Length 



        
aj = findFighter(df_fights_ufc, "B_fighter", "Marlon Moraes")
aj_data = findFighter(ufc_bouts, "fighter1", "Cejudo")

#cejudo =  findFighter(ufc_fighters, "fighter_name", "Henry Cejudo")
#R_fighter = "R_fighter"
#B_fighter = "B_fighter"
#fights_red_cejudo = findFighterFights(df_fights_ufc, R_fighter, "Henry Cejudo")
#fights_blue_cejudo = findFighterFights(df_fights_ufc, B_fighter, "Henry Cejudo")

#group fights by win/draw methods
win_method_tko = findFighter(df_ufc_fights_X, "method", "TKO")
win_method_sub = findFighter(df_ufc_fights_X, "method", "Submission")
win_method_ko = findFighter(df_ufc_fights_X, "method", "KO")
win_method_decision = findFighter(df_ufc_fights_X, "method", "Decision")
drawn_fight = findFighter(df_ufc_fights_X, "method", "Draw")
fighter_dq = findFighter(df_ufc_fights_X, "method", "DQ")
fighter_nc = findFighter(df_ufc_fights_X, "method", "No Contest")
figher_tech_sub = findFighter(df_ufc_fights_X, "method", "Technical")
fighter_nc2 = findFighter(df_ufc_fights_X, "method", "No")

group_dec = df_ufc_fights_X.groupby(['method','method_d']).count()
dec = df_ufc_fights_X.groupby('method')['method_d'].nunique()
group_fname = df_ufc_fights_X.groupby(['f1name', 'f2name', 'f1result','method','method_d','round','time']).count()
#red_judo = np.array(fights_red_cejudo)
#blue_judo = np.array(fights_blue_cejudo)

#merged = pd.merge(fights_blue_cejudo,fights_red_cejudo, how="inner", on=["new_guy"])
#joined = red_judo.join(blue_judo, how ="inner")



#fig = plt.figure()
#axis_1 = fig.add_subplot(1,1,1)
#axis_1.plot(df_fights_ufc["B_wins"], df_fights_ufc["R_wins"], "x", color="pink")
#axis_1.set_xlabel("Blue")
#axis_1.set_ylabel("Red")
#axis_1.set_title("The relationshiip between Fights and wins")

axis_2 = plt.figure().add_subplot(1,1,1)
axis_2.plot(df_fights_ufc["B_total_rounds_fought"]/3, df_fights_ufc["R_total_rounds_fought"]/3, "o", color="green")
axis_2.set_xlabel("Blue")
axis_2.set_ylabel("Red")
axis_2.set_title("Rel btw Stk Acc and Total Fights")



#plot3 = plt.figure().add_subplot(1,1,1)
#plot3.plot(fight_a , a["B_wins"],  "x", color="black" )
#plot3.set_ylabel("Wins")
#plot3.set_xlabel("No. of Fights")
#plot3.set_title("Rel btw Wins and Fight")
















from sklearn.preprocessing import StandardScaler



#a = df_fights_ufc.loc[df_fights_ufc["B_wins"] + df_fights_ufc["B_losses"] >= 20]


#wins_r_ten = np.array(df_fights_ufc["B_wins"])
#a = np.where(np.logical_and(wins_r_ten>=10, wins_r_ten<=10))
#arr = np.stack(a, axis = 1)
#wins_b_ten = find(df_fights_ufc,"B_wins", "5")

#create scaler
scaler = StandardScaler()


#transform

