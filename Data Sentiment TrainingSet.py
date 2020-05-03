import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

X = pd.read_csv('train.csv')
temp = pd.read_csv('train.csv')

X.dropna(axis=0 , subset=['base_score'],inplace=True)
#Y=X.effectiveness_rating
X.drop(['patient_id','name_of_drug','use_case_for_drug','drug_approved_by_UIC','number_of_times_prescribed','base_score','effectiveness_rating'], axis=1, inplace=True)

temp.drop(['patient_id','drug_approved_by_UIC'], axis=1, inplace=True)

cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 
X.drop(cols_with_missing, axis=1, inplace=True)

sid = SentimentIntensityAnalyzer()
list_neg=[]
list_pos=[]
list_neu=[]
list_compound=[]
for index,row in X.iterrows():
    ss = sid.polarity_scores(row['review_by_patient'])
    list_neg.append(ss['neg'])
    list_pos.append(ss['pos'])
    list_neu.append(ss['neu'])
    list_compound.append(ss['compound'])
    #print(row,ss['neg'],ss['pos'],ss['neu'],ss['compound'])
se_neg = pd.Series(list_neg)
se_pos = pd.Series(list_pos)
se_neu = pd.Series(list_neu)
se_compound = pd.Series(list_compound)
#print(len(se),len(X))
X['negative'] = se_neg.values
X['positive'] = se_pos.values
X['neutral'] = se_neu.values
X['compound'] = se_compound.values

# print(X)

output = pd.DataFrame({'name_of_drug': temp.name_of_drug,'use_case_for_drug':temp.use_case_for_drug,'effectiveness_rating':temp.effectiveness_rating,'number_of_times_prescribed':temp.number_of_times_prescribed,'positive':X.positive,'negative':X.negative,'neutral':X.neutral,'compound':X.compound,'base_score':temp.base_score})
output.to_csv('new_train.csv', index=False)