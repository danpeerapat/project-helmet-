from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
import base64
import re
import io
import requests
from PIL import Image
import paho.mqtt.client as mqtt
import threading
import time
import os

app = Flask(__name__)
CORS(app)

vehicle_model_url = 'https://www.dropbox.com/scl/fi/zlmm6k6u96qgemddm4bzn/vehicle_classification.onnx?rlkey=pvcrm0bv2vxczmhou6bfqoa9h&st=4pplshvw&dl=1'
helmet_model_url = 'https://www.dropbox.com/scl/fi/gd8djpwcr9itx3nkxgjbr/helmet_detection_model.onnx?rlkey=f5p5ezg76wdicvcuzfw4kzuub&st=bpc3tiuu&dl=1'

def download_model(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download model from {url}")

download_model(vehicle_model_url, 'vehicle_classification.onnx')
vehicle_session = ort.InferenceSession('vehicle_classification.onnx')

download_model(helmet_model_url, 'helmet_detection.onnx')
helmet_session = ort.InferenceSession('helmet_detection.onnx')

# MQTT broker details
broker = "d8229ac5fefe43a9a7c09fabb5f30929.s1.eu.hivemq.cloud"
port = 8883
username = "python"
password = "123456789"
topic = "esp32cam/capture"

client = mqtt.Client()
client.tls_set()
client.username_pw_set(username, password)
client.connect(broker, port)

class_names = ['bike', 'car']
helmet_class_names = ['no helmet', 'helmet']
CONFIDENCE_THRESHOLD = 0.7  

def prepare_image(img, img_width=150, img_height=150):
    img = img.resize((img_width, img_height))
    img_array = np.array(img).astype('float32')
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def ping_mqtt():
    while True:
        client.publish(topic, "ping")
        print("Sent 'ping' to MQTT server")  
        time.sleep(60) 

def classify_vehicle(img):
    img_array = prepare_image(img)
    input_name = vehicle_session.get_inputs()[0].name
    prediction = vehicle_session.run(None, {input_name: img_array})
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    return predicted_class, confidence

def classify_helmet(img):
    img_array = prepare_image(img)
    input_name = helmet_session.get_inputs()[0].name
    outputs = helmet_session.run(None, {input_name: img_array})
    prediction = outputs[0]
    predicted_helmet_class = np.argmax(prediction)
    helmet_confidence = np.max(prediction)
    return predicted_helmet_class, helmet_confidence

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' in request.form:
        image_data = request.form['image']
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        img = img.convert('RGB')

        # ตรวจจับประเภทพาหนะ
        predicted_vehicle_class, vehicle_confidence = classify_vehicle(img)

        if vehicle_confidence < CONFIDENCE_THRESHOLD:
            return jsonify({'message': "No action required for low confidence prediction"}), 200

        if predicted_vehicle_class == 0:  # มอเตอร์ไซค์
            client.publish(topic, "reset_camera")
            predicted_helmet_class, helmet_confidence = classify_helmet(img)

            if predicted_helmet_class == 0:
                result = {
                    'vehicle_result': f"Prediction: Bike with confidence {vehicle_confidence * 100:.2f}%",
                    'helmet_result': f"Helmet detected with confidence {helmet_confidence * 100:.2f}%",
                    'action': "pass"
                }
                client.publish(topic, "pass")
            else:
                result = {
                    'vehicle_result': f"Prediction: Bike with confidence {vehicle_confidence * 100:.2f}%",
                    'helmet_result': f"No helmet detected with confidence {helmet_confidence * 100:.2f}%",
                    'action': "no_pass"
                }
                client.publish(topic, "no helmet")

        elif predicted_vehicle_class == 1:  # รถยนต์
            result = {
                'vehicle_result': f"Prediction: Car with confidence {vehicle_confidence * 100:.2f}%",
                'action': "pass"
            }
            client.publish(topic, "pass")
        else:
            return jsonify({'message': "No action required for non-vehicle objects"}), 200

        return jsonify(result)
    
    else:
        return "No image found", 400

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    # เริ่มต้น Thread สำหรับ ping MQTT
    ping_thread = threading.Thread(target=ping_mqtt)
    ping_thread.daemon = True
    ping_thread.start()

    # เริ่ม Flask server
    port = int(os.environ.get("PORT", 5000))  # ใช้พอร์ตจากตัวแปรสภาพแวดล้อมหรือ 5000 ถ้าไม่ได้กำหนด
    app.run(host='0.0.0.0', port=port)
