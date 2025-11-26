import tritonclient.http as httpclient
import numpy as np

TRITON_SERVER_URL = "65.0.55.109:8000"
MODEL_NAME = "mistral-7b"

client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

if not client.is_server_live():
    print("❌ Server is not live!")
if not client.is_model_ready(MODEL_NAME):
    print(f"❌ Model {MODEL_NAME} is not ready!")
    exit(1)

print("✅ Server and model are ready!")

prompt_text = "Tell me a fun fact about space."

# ✅ Add extra dimension: shape [1, 1]
input_data = np.array([[prompt_text.encode("utf-8")]], dtype=np.object_)

inputs = []

inputs.append(httpclient.InferInput("INPUT_TEXT", input_data.shape, "BYTES"))
inputs[0].set_data_from_numpy(input_data)

max_tokens = np.array([[128]], dtype=np.int32)      # shape [1, 1]
temperature = np.array([[0.7]], dtype=np.float32)  # shape [1, 1]

max_tokens_input = httpclient.InferInput("MAX_TOKENS", max_tokens.shape, "INT32")
max_tokens_input.set_data_from_numpy(max_tokens)
inputs.append(max_tokens_input)

temperature_input = httpclient.InferInput("TEMPERATURE", temperature.shape, "FP32")
temperature_input.set_data_from_numpy(temperature)
inputs.append(temperature_input)

outputs = []
outputs.append(httpclient.InferRequestedOutput("OUTPUT_TEXT"))

response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

output_data = response.as_numpy("OUTPUT_TEXT")[0].decode("utf-8")

print("📝 Response from Mistral-7B:")
print(output_data)

