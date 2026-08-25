from fastapi import FastAPI
import joblib
import pandas as pd 
app=FastAPI() 
model = joblib.load("mymodel.pkl")
# model - predication = store /db mysql 
# api -call = list 
@app.get("/")
def testing():
    return {"test":"all ok"}

@app.post("/predication")
def mypredication(hours:float):
    newdata=pd.DataFrame({
    "StudyHours":[hours]
    }) 
    mynewdata= model.predict(newdata)
    return {
        "predication":float(mynewdata[0])
    }
