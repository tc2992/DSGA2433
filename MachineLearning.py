import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
warnings.filterwarnings("ignore")

class ml_model(object):
    def __init__(self):
        self.model=pickle.load(open('trained_model.pkl','rb'))

    def data_preprocessing(self, demo, diet, exam, labs, ques):
        '''
        demo, diet, exam, labs, ques are the original datasets
        '''
        df = demo.join(diet.set_index('SEQN'), on='SEQN', how='inner')
        df = df.join(exam.set_index('SEQN'), on='SEQN', how='inner')
        df = df.join(labs.set_index('SEQN'), on='SEQN', how='inner')
        df = df.join(ques.set_index('SEQN'), on='SEQN', how='inner')
        df=df.dropna(axis=1,how='all')
        df=df.dropna(axis=0,how='all')
        df=df.loc[:,['SEQN', 'RIAGENDR', 'INDFMPIR', 'LBXGH', 'DBD100', 'DMDEDUC2', 'RIDAGEYR', 'BMXBMI', 'BMDAVSAD', 'MGDCGSZ']]
        df.columns = ['SEQN','Gender','Family_income','LBXGH','Salt_level','Education_level','Age','BMI','Abdominal_size','Grip_strength']
        df=df.dropna(axis=1,how='all')
        df=df.dropna(axis=0,how='all')
        df = df[df['Family_income'].notnull()]
        df['LBXGH']=df['LBXGH'].fillna(df['LBXGH'].mean())
        df['Salt_level']=df['Salt_level'].fillna(df['Salt_level'].median())
        df['Education_level']=df['Education_level'].fillna(df['Education_level'].median())
        df['BMI']=df['BMI'].fillna(df['BMI'].mean())
        df['Abdominal_size']=df['Abdominal_size'].fillna(df['Abdominal_size'].mean())
        df['Grip_strength']=df['Grip_strength'].fillna(df['Grip_strength'].mean())
        df.loc[df['LBXGH']<6.0, 'Diabetes']=0
        df.loc[df['LBXGH']>=6.0, 'Diabetes']=1
        df=df.drop('LBXGH',axis=1)
        df=df.astype({'Salt_level':'int64','Education_level':'int64','Diabetes':'int64'})
        X=df.drop('Diabetes',axis=1)
        X=X.drop('SEQN',axis=1)
        y=df['Diabetes']
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)
        return X_train,X_test,y_train,y_test

    def retrain(self, demo, diet, exam, labs, ques):
        '''
        demo, diet, exam, labs, ques are the original datasets
        '''
        X_train,X_test,y_train,y_test = self.data_preprocessing(demo, diet, exam, labs, ques)
        rf=RandomForestClassifier()
        params={'class_weight':[{1:3},{1:3.5},{1:4},{1:4.5},{1:10},'balanced','balanced_subsample'],'n_estimators':[25,50,100],'max_depth':[2,5,10,20,50,100]}
        gs=GridSearchCV(rf, params,'f1')
        gs.fit(X_train,y_train)
        clf=gs.best_estimator_
        y_pred = clf.predict(X_test)
        mean_accu=clf.score(X_test,y_test)
        f1=f1_score(y_test,y_pred)
        self.model=clf
        print('mean accuracy is {}, f1 score is {}'.format(mean_accu,f1))
        
    def predict(self, data):
        '''
        data is of shape (n_samples, n_features)
        '''
        predict_class=self.model.predict(data)[0]
        diabetes_prob=self.model.predict_proba(data)[0][1]
        return predict_class, diabetes_prob
    
    
