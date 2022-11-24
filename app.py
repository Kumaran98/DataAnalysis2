# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:32:01 2022

@author: Eelapriya
"""

from fastapi import FastAPI
import uvicorn
import pickle

from models import Women
app=FastAPI()
model = pickle.load(open("model_1.pkl","rb"))

@app.get("/")
def greet():
   return{"Hello World"}

@app.post("/predict")
def predict(req:Women):
   
   sex                = req.sex 
   Age                = req.Age
   Married             = req.Married
   Number_children      = req.Number_children
   education_level      = req.education_level
   total_members        = req.total_members
   gained_asset         = req.gained_asset
   durable_asset        = req.durable_asset
   save_asset           = req.save_asset 
   living_expenses      = req.living_expenses 
   other_expenses       = req.other_expenses 
   incoming_salary       = req.incoming_salary
   incoming_own_farm     = req.incoming_own_farm 
   incoming_business     = req.incoming_business
   incoming_no_business  = req.incoming_no_business
   incoming_agricultural  = req.incoming_agricultural
   farm_expenses         = req.farm_expenses
   labor_primary         = req.labor_primary
   lasting_investment    = req.lasting_investment 
   no_lasting_investmen  = req.no_lasting_investmen  
   features=list([ sex  ,Age ,Married ,Number_children ,education_level ,total_members,  gained_asset ,durable_asset, save_asset, living_expenses ,
other_expenses, incoming_salary, incoming_own_farm, incoming_business, incoming_no_business, incoming_agricultural, farm_expenses,
labor_primary, lasting_investment, no_lasting_investmen])      
   predict=model.predict([features])  
   probab = model.predict_proba([features])    
   if(predict ==1):
      return{"ans":"You are likely to have Depression with {} Probability".format(probab[0][1])}
   else:
      return{"ans":"You did not have the Depression with {} Probability".format(probab[0][0])}

if __name__ == "__main__":
  uvicorn.run(app)
