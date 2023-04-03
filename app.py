import torch
from flask import Flask, request, send_file
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from PIL import Image
import torchvision.transforms.functional as TF

app = Flask(__name__)


# Laden des Modells und auf passende Hardware gesetzt
model_type = "MiDaS_small"
#model_type = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()


# Verschiedene Transformationsmethoden für die verschiedenen Modelle
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


# Hauptseite
@app.route("/")
def indexPage():
    return send_file("frontend/index.html")


# /predict nimmt eine Bilddatei entgegen, passt die Grösse des Bildes an und gibt die Depthmap als Bild zurück
@app.route('/predict', methods=['POST'])
def predict():
    #Fehlerbehandlung
    try:
        # Das hochgeladene Bild wird als File-Objekt übergeben
        file = request.files['file']
        
        # Lesen des Bildes mit Pillow
        img = Image.open(file)

        # Anpassen der Grösse des Bildes
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

        # Anzeigen des Bildes
        fig, ax = plt.subplots()
        ax.imshow(output)
        
        # Plot als Base64 encoded image zurückgeben
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        # Base64 encoded image als HTML-Tag zurückgeben
        return f'<img src="data:image/png;base64,{image_base64}"/>'
    
    # Fehlerbehandlung
    except Exception as e:
        return f'Error: {str(e)}'

# Starten des Servers auf dem Port 8000
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
