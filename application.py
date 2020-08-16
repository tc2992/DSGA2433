from tkinter import *
import pandas as pd
from MachineLearning import ml_model

class Root(Tk):
    def __init__(self, re_train):
        super(Root,self).__init__()
        self.data=[]
        self.title("{COMPANY_NAME}")
        self.minsize(640,400)
        self.scrollbar = Scrollbar(self)
        self.scrollbar.pack( side = RIGHT,fill=Y)
        self.mylist=Listbox(self, yscrollcommand = self.scrollbar.set )
        #self.wm_iconbitmap('icon.ico')
        self.getanswers()
        #print (self.getanswers())
        self.getresult(retrain=re_train)
        self.mylist.pack()
        self.scrollbar.config( command = self.mylist.yview )

    
    def getanswers(self):
        self.getinfo('Please enter your name:')
        self.getinfo('Please enter your Gender {1 for Male, 2 for Female}:')
        self.getinfo('Please enter your family income level correspending to this standard {1:$11,670;2:$15,730;3:$19,790;4:$23,850;5:$27,910;6:$31,970;7:$36,030;8:$40,090},You can enter fraction.')
        self.getinfo('Please evaluate your salt level from 1 to 9 {1 for very plain, 9 for very salty}:')
        self.getinfo('Please enter the highest level of school {1:less than a high shool diploma; 2: high school gradeate; 3: College with no degree; 4: Professonal certificate; 5: Associate degree; 6: Bachelors degree; 7: Masters Degree; 8: Doctoral Degree; 9: Professonal Degree}:')
        self.getinfo('Please enter your age:')
        self.getinfo('Please enter your BMI:')
        self.getinfo('Please enter your abdominal size in cm:')
        self.getinfo('Please enter your grip_strength index:')
        #self.mylist.pack( side = LEFT, fill = BOTH )

    def getinfo(self,question):
        def click():
            val=e.get()
            if val.isdigit():
                self.data.append(float(val))
            else:
                self.data.append(val)
            out='You entered '+val
            self.mylist.insert(END,out)
            lb= Label(self,text=out)
            lb.pack()
        label= Label(self,text=question)
        e=Entry(self,width=50)
        label.pack()
        e.pack()
        bt=Button(self,text='COMFIRM',command=click)
        bt.pack()

    def getresult(self, retrain):
        def finalclick():
            #call Model here
            if retrain==False:
                model=ml_model()
            else:
                demo=pd.read_csv('data/demographic.csv')
                diet=pd.read_csv('data/diet.csv')
                exam=pd.read_csv('data/examination.csv')
                labs=pd.read_csv('data/labs.csv')
                ques=pd.read_csv('data/questionnaire.csv')
                model=ml_model()
                model.retrain(demo,diet,exam,labs,ques)
            pred_class,pred_proba=model.predict([self.data[1:]])
            if pred_proba>0.6:
                plan='Diabetes_Recommend_Product'
            else:
                plan='Regular_Product'
            out='For Customer '+ self.data[0]+', the insurance plan is '+plan
            self.mylist.insert(END,out)
            lb= Label(self,text=out)
            lb.pack()
        bt=Button(self,text='--START TO EVALUATE--',command=finalclick)
        bt.pack()

    


root = Root(re_train=False)
root.mainloop()
