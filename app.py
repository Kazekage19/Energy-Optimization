import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

## Load the model
with open("objects.pkl", "rb") as f:
        objects = pickle.load(f)
model=objects[0]
scalar=objects[1]
@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/predict_api',methods=['POST','GET'])
# def predict_api():
#     data=request.json['data']
#     print(data)
#     print(np.array(list(data.values())).reshape(1,7,1))
#     new_data=scalar.transform(np.array(list(data.values())).reshape(1,7,1))
#     output=model.predict(new_data)
#     print(output[0])
#     return jsonify(output)


@app.route('/predict',methods=['POST'])
def predict():
    # data=[float(x) for x in request.form.values()]
    # final_input=scalar.transform(np.array(data).reshape(1,7,1))
    # print(final_input)
    # output=model.predict(final_input)[0]
    # return render_template("index.html",prediction_text="Outputs {}".format(output))
    form_data=request.form 
    json_data=json.dumps(form_data)
    data=json.loads(json_data)
    inp=np.array(data.values())
    out=model.predict(inp)
    return out



if __name__=="__main__":
    app.run(debug=True)