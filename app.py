import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
from dotenv import load_dotenv
import os
import string
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from flask import Flask, request

load_dotenv()

def encoder(data):
    string_cols = ['animal_type', 'breed', 'color', 'name', 'outcome_type', 'outcome_subtype', 'sex_upon_outcome']
    data_csv = pd.read_csv('aac_shelter_outcomes.csv')
    for col in string_cols:
        data[col] = data[col].str.lower()
        data[col] = data[col].apply(lambda x:text_process(str(x)))
        data_csv[col] = data_csv[col].str.lower()
        data_csv[col] = data_csv[col].apply(lambda x:text_process(str(x)))
        
    for col in string_cols:
        label_encoder = LabelEncoder()
        # label_encoder.classes_ = np.load('models/classes.npy', allow_pickle=True)
        label_encoder.fit(data_csv[col])
        data[col] = label_encoder.transform(data[col])
    return data

def decode(pred):
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('models/classes.npy', allow_pickle=True,)
    label_encoder.inverse_transform(pred[0])
    return pred

def text_process(text):
    text = text.replace('/', ' ')
    return ''.join([char for char in text if char not in string.punctuation])

app = Flask(__name__)

@app.route("/", methods=['POST'])
def sms():
    msg = request.form.get('Body').lower()
    if (msg.startswith('details: ')):
        msg = msg.replace('details: ', '').split(',')
        data = pd.DataFrame({
            'animal_type': msg[0].strip(),
            'breed': msg[1].strip(),
            'color': msg[2].strip(),
            'name': msg[3].strip(),
            'outcome_type': 'transfer',
            'outcome_subtype': 'partner',
            'sex_upon_outcome': msg[4].strip(),
            'age_in_days': msg[5].strip(),
        },
        index=[1])
        data = encoder(data)
        rf = joblib.load('models/rf_animal_centre_outcome_prediction.joblib')
        pred = rf.predict(data.drop(['outcome_type', 'outcome_subtype'], axis=1))
        print(pred)
        pred = decode(pred)
        print(pred)
        res = MessagingResponse()
        res.message("Thank you again.\nPlease not that this result is not 100\% accurate.\nResult: " + pred)
        return str(res)

    res = MessagingResponse()
    res.message("Hello there!!\nThank you for contacting us.\nEnter the details of the animal in the format shown below to get the statistical chances of an animal being adopted.\n\ndetails: animal_type, breed, color, name, sex, age_in_days\n\nBe careful to not leave any extra spaces or commas.")
    return str(res)

if __name__ == "__main__":
    app.run(debug=False)
