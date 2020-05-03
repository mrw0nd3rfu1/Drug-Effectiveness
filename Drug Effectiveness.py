import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score

X = pd.read_csv('new_train.csv')
X_test = pd.read_csv('new_test.csv')
temp_test = pd.read_csv('new_test.csv')

Y = X.base_score
X.drop(['base_score'],axis=1,inplace=True)
X_test.drop(['patient_id'],axis=1,inplace=True)

X['name_of_drug'] = LabelEncoder().fit_transform(X['name_of_drug'])
X['use_case_for_drug'] = LabelEncoder().fit_transform(X['use_case_for_drug'])

X_test['name_of_drug'] = LabelEncoder().fit_transform(X_test['name_of_drug'])
X_test['use_case_for_drug'] = LabelEncoder().fit_transform(X_test['use_case_for_drug'])

#train_X,val_X,train_Y,val_Y = train_test_split(X,Y,train_size=0.8,test_size=0.2,random_state=0)
model = RandomForestRegressor(n_estimators=100, random_state=1)
model.fit(X,Y)
preds = model.predict(X_test)

#checking with validation data for accuracy scores
# me=mean_absolute_error(val_Y, preds)
# r_score = r2_score(val_Y,preds)
# print(me,r_score)
output = pd.DataFrame({'patient_id':temp_test.patient_id,'base_score':preds})
output.to_csv('submission.csv',index=False)