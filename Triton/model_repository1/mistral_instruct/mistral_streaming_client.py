import numpy as np
import gradio as gr
import tritonclient.http as httpclient
import time

TRITON_URL = "localhost:8000"  # or "<your-ec2-public-ip>:8000"
MODEL_NAME = "mistral_instruct"

def infer(prompt):
    client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
    
    inputs = []
    outputs = []

    input_text = httpclient.InferInput("PROMPT", [1], "BYTES")
    input_text.set_data_from_numpy(np.array([prompt.encode("utf-8")], dtype="object"))
    inputs.append(input_text)

    output_text = httpclient.InferRequestedOutput("OUTPUT", binary_data=False)
    outputs.append(output_text)

    result = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
    output = result.as_numpy("OUTPUT")[0].decode("utf-8")

    return output

# Launch Gradio UI
demo = gr.Interface(fn=infer, inputs="text", outputs="text", title="Triton Mistral Inference")

if __name__ == "__main__":
    demo.launch(server_port=7860, server_name="0.0.0.0", debug=True)
