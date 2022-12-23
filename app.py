import dill
from flask import Flask,request,app,jsonify,render_template,url_for
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

app=Flask(__name__)

#loading a pickle file
#model=pickle.load(open('model.pkl','rb'))

#Home page for enterring data
@app.route('/')
def Home():
    return render_template("Home.html")

#creating the api
@app.route('/predict_api',methods=['POST'])
def predict_api(): #for predicting single output from model

    #for getting json data from postman
    data=request.json['data']
    print(data)
    new_data=[list(data.values())] #converting single data into 2-d array
    output=model.predict(new_data)[0]
    return jsonify(output)

@app.route('/predict',methods=['POST','GET'])
def predict():
    #data = [float(x) for x in request.form.values()]
    #final_features = [np.array(data)]
    #print(data)

    #output = model.predict(final_features)[0]
    #print(output)
    # output = round(prediction[0], 2)
    if request.method == 'POST':
        gender = request.form['gender']
        Age_bin = request.form['Age_bin']
        hypertension = float(request.form['hypertension'])
        heart_disease = float(request.form['heart_disease'])
        ever_married = request.form['ever_married']
        work_type = request.form['work_type']
        Residence_type = request.form['Residence_type']
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = request.form['smoking_status']

        output = preprocessing(gender,Age_bin,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status)
        if(output[0]==0):
            output1="No"
        else:
            output1="Yes"

        return render_template('Home.html', prediction_text=output1)
    return render_template('Home.html')


def preprocessing(gender,Age_bin,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status):
    numerical_pipeline=Pipeline([
        ('feature_scaling',StandardScaler())
    ])

    categorical_pipeline=Pipeline([
        ('categorical_encoder', OrdinalEncoder())
    ])

    numerical_columns=['avg_glucose_level', 'bmi']
    categorical_columns=['gender', 'Age_bin', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

    column_pipeline=ColumnTransformer([
        ("numerical_pipeline",numerical_pipeline,numerical_columns),
        ("categorical_pipeline",categorical_pipeline,categorical_columns)
    ])

    print(column_pipeline)

    #t={"gender":['Female'],"Age_bin":["Adults"],"hypertension":[1],"heart_disease":[0],"ever_married":['Yes'],"work_type":["Self-employed"],"Residence_type":["Urban"],"avg_glucose_level":[196.92],"bmi":[22.2],"smoking_status":["never smoked"]}
    #l=pd.DataFrame(t)
    #X_test=column_pipeline.transform(X_test)
    with open("model.pkl",'rb') as file:
        output=dill.load(file)
    Train_set=pd.read_csv('Train.csv')
    X_train_processed=column_pipeline.fit_transform(Train_set)     
    result = get_as_dataframe(gender,Age_bin,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status)
    result_output = output.predict(column_pipeline.transform(result))
    return result_output

def get_as_dataframe(
    gender,Age_bin,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status
):
    try:
        input_data = {
            "gender" : [gender],
            "Age_bin" : [Age_bin],
            "hypertension" : [hypertension],
            "heart_disease" : [heart_disease],
            "ever_married" : [ever_married],
            "work_type" : [work_type],
            "Residence_type" : [Residence_type],
            "avg_glucose_level" : [avg_glucose_level],
            "bmi" : [bmi],
            "smoking_status" : [smoking_status]}
        input_dataframe = pd.DataFrame(input_data)
        return input_dataframe
    except Exception as e:
        raise Exception(e)


if __name__=="__main__":
    app.run(debug=True)

    