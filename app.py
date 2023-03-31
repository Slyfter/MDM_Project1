import io
import torch
import numpy as np
from flask import Flask, request, jsonify, send_file
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from PIL import Image
import torchvision.transforms.functional as TF

app = Flask(__name__)

#model_type = "DPT_Hybrid"
model_type = "MiDaS_small"

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

@app.route("/")
def indexPage():
    return send_file("frontend/index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded file
        file = request.files['file']
        
        # Read image using PIL
        img = Image.open(file)

        # Resize the image to a larger size
        img = TF.resize(img, (800, 800))

        input_tensor = TF.to_tensor(img)
        input_tensor = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = midas(input_tensor)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.size,
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
