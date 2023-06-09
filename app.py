import pickle
from flask import Flask, request, jsonify, app, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
## Load the model
reg_model = pickle.load(open("regmodel.pkl", "rb"))
scalar = pickle.load(open("scaling.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict_api", methods=["POST"])
def predict_api():
    # taking new data data for prediction
    data = request.json["data"]
    # reshaping the data into 1D array
    reshaped_data = np.array(list(data.values())).reshape(1, -1)
    print(reshaped_data)
    # transforming data with StandardScaler
    scaled_data = scalar.transform(reshaped_data)
    # making prediction
    output = reg_model.predict(scaled_data)
    # returning final output of prediction
    print(output[0])
    return jsonify(output[0])


@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = reg_model.predict(final_input)[0]
    return render_template(
        "home.html", prediction_text="The House Price prediction is {}".format(output)
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
