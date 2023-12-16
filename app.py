import torch
from torchvision import transforms
from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import torch.nn as nn
import torcheval
from models import *

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # Example: set max size to 1024 MB

# Load the trained PyTorch model
model_path = 'model.pth'  # Update with your model path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('model.pth')
model.eval()

# Define the image transformation (update according to your model)
transform = transforms.Compose([transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

labels = {0:'apple_pie', 1:'baby_back_ribs', 2:'baklava'}

# Define a simple neural network prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        file = request.files['file']
        print(f'Received file: {file}')
        image = Image.open(io.BytesIO(file.read()))
        image = transform(image).unsqueeze(0)

        print(type(model))
        # Make prediction
        with torch.no_grad():
            output = model(image)

        # Convert the output to a probability score
        probability = torch.argmax(output).numpy().tolist()

        return jsonify({'prediction': labels.get(probability)})
    

    except Exception as e:
        print(f'Error: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

