import tritonclient.http as httpclient
import numpy as np
from PIL import Image

# Change this to your test image path
image_path = "/mnt/c/Users/rajpu/Desktop/Trinton/test_img.webp"

# Preprocess image
image = Image.open(image_path).convert("RGB")
image = image.resize((300, 300))
image_np = np.array(image, dtype=np.float32) / 255.0  # normalize to [0,1]
image_np = np.transpose(image_np, (2, 0, 1))  # channels first
image_np = np.expand_dims(image_np, axis=0)   # add batch dim

# Create client
client = httpclient.InferenceServerClient(url="13.232.161.142:8000")

# Create inputs and outputs
inputs = httpclient.InferInput("x.1", image_np.shape, "FP32")
inputs.set_data_from_numpy(image_np, binary_data=True)

outputs = httpclient.InferRequestedOutput("output__0")

# Run inference
response = client.infer(model_name="room_model", inputs=[inputs], outputs=[outputs])

# Get output
output_data = response.as_numpy("output__0")
print("Model output probabilities:", output_data)
print("Predicted class:", np.argmax(output_data))
