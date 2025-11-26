import tritonclient.http as httpclient
import numpy as np

TRITON_SERVER_URL = "localhost:8000"  # or your public IP:port if using external

MODEL_NAME = "mistral-7b"

# Create client
client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL, verbose=True, concurrency=1, ssl=False, connection_timeout=300, network_timeout=300)

# Check readiness
if not client.is_server_live():
    print("❌ Server is not live!")
if not client.is_model_ready(MODEL_NAME):
    print(f"❌ Model {MODEL_NAME} is not ready!")
    exit(1)

print("✅ Server and model are ready!")

# Prepare example input
prompt_text = "Tell me a fun fact about space."
input_data = np.array([[prompt_text.encode("utf-8")]], dtype=np.object_)

inputs = []
outputs = []

# INPUT_TEXT
inputs.append(httpclient.InferInput("INPUT_TEXT", input_data.shape, "BYTES"))
inputs[0].set_data_from_numpy(input_data)

# MAX_TOKENS
max_tokens = np.array([[128]], dtype=np.int32)
inputs.append(httpclient.InferInput("MAX_TOKENS", max_tokens.shape, "INT32"))
inputs[1].set_data_from_numpy(max_tokens)

# TEMPERATURE
temperature = np.array([[0.7]], dtype=np.float32)
inputs.append(httpclient.InferInput("TEMPERATURE", temperature.shape, "FP32"))
inputs[2].set_data_from_numpy(temperature)

# Output
outputs.append(httpclient.InferRequestedOutput("OUTPUT_TEXT"))

# Inference
response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

# Parse output
output_data = response.as_numpy("OUTPUT_TEXT")[0][0].decode("utf-8")

print("📝 Response from Mistral-7B:")
print(output_data)
