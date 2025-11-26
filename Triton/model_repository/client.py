import tritonclient.grpc.aio as grpcclient
import numpy as np
from tritonclient.grpc import InferInput
import asyncio
import time

async def main():
    model_name = "llama2vllm"  # Change this if your model has a different name

    async with grpcclient.InferenceServerClient("localhost:8001") as client:
        # Check server and model readiness
        if not await client.is_server_ready():
            print("Server is not ready.")
            return

        if not await client.is_model_ready(model_name):
            print(f"Model '{model_name}' is not ready.")
            return

        prompt = "What is the capital of France?"
        input_tensor = InferInput("PROMPT", [1], "BYTES")
        input_tensor.set_data_from_numpy(np.array([prompt.encode("utf-8")], dtype=np.bytes_))

        # Send inference request
        response = await client.infer(model_name, [input_tensor])
        result = response.as_numpy("OUTPUT")[0].decode("utf-8")

        # Simulate streaming (word-by-word output)
        print("Streaming response:")
        for word in result.split():
            print(word, end=" ", flush=True)
            time.sleep(0.1)  # simulate delay
        print("\n\nDone.")

if __name__ == "__main__":
    asyncio.run(main())
