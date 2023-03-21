import cv2
import torch
import urllib.request
import numpy as np
from flask import Flask, request, jsonify, send_file
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

@app.route("/")
def indexPage():
    return send_file("frontend/index.html")

# Load a model

model_type = "DPT_Hybrid" 

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()



@app.route('/predict', methods=['POST'])
def predict():
    try:
         # Get the uploaded file
        filename = request.files['file']

        # Load transforms to resize and normalize the image for large or small model
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        # Load image and apply transforms

        img = cv2.imdecode(np.fromstring(filename.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)

        # Predict and resize to original resolution

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()

        # Show result
        fig, ax = plt.subplots()
        ax.imshow(output)
        
        # Convert plot to Base64 encoded image
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Close plot
        plt.close()

        # Return Base64 encoded image
        return f'<img src="data:image/png;base64,{image_base64}"/>'
    
    except Exception as e:
        return f'Error: {str(e)}'
    
def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    return f'<img src="data:image/png;base64,{string.decode("utf-8")}" alt="plot">'
