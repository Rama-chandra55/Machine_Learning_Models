import io
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x


# Load your high-accuracy weights file
device = torch.device("cpu")
model = CNN()
try:
    model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
except FileNotFoundError:
    print("⚠️ Working directory missing 'mnist_cnn.pth'. Ensure weights are saved properly.")
model.eval()

# Preprocessing transformation (matching validation metrics)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']

        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]

        image_bytes = base64.b64decode(image_data)
        raw_img = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert directly to Grayscale

        # Convert PIL Image to a NumPy array for OpenCV processing
        img_np = np.array(raw_img)

        # 1. Invert colors: Canvas is black-on-white, we need white-on-black
        img_np = cv2.bitwise_not(img_np)

        # 2. Thresholding: Turn gray smudges into clean, distinct white lines
        _, thresh = cv2.threshold(img_np, 50, 255, cv2.THRESH_BINARY)

        # 3. Find bounding box of the drawn digit to crop out excess empty space
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Get the largest contour (the drawn number)
            ctr = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(ctr)

            # Crop tightly around the digit
            cropped = thresh[y:y + h, x:x + w]

            # Make the crop a square by adding symmetric padding
            max_dim = max(w, h)
            pad_w = (max_dim - w) // 2
            pad_h = (max_dim - h) // 2

            # Add a protective border margin around the number (crucial for MNIST!)
            margin = int(max_dim * 0.3)
            padded = cv2.copyMakeBorder(
                cropped,
                pad_h + margin, pad_h + margin,
                pad_w + margin, pad_w + margin,
                cv2.BORDER_CONSTANT, value=0
            )

            # 4. Resize cleanly to 28x28
            final_img = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
        else:
            # FIX: Used valid Python list formatting instead of JavaScript array logic
            return jsonify({'success': True, 'distribution': [0.0] * 10})

        # Convert back to PIL Image to apply the standard PyTorch normalization tensor safely
        processed_pil = Image.fromarray(final_img.astype('uint8'))
        tensor = transform(processed_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output, _ = model(tensor)
            probabilities = F.softmax(output, dim=1).squeeze(0)
            percentage_distribution = (probabilities * 100).tolist()

        return jsonify({
            'success': True,
            'distribution': percentage_distribution
        })
    except Exception as e:
        print(f"Server Error: {str(e)}")  # This prints error details directly to your terminal
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)