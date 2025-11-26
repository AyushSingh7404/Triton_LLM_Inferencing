import tritonclient.http as httpclient
import numpy as np
import json
import time

class TritonMistralClient:
    def __init__(self, url="localhost:8000"):
        self.client = httpclient.InferenceServerClient(url=url)
        self.model_name = "mistral-7b"
        
    def is_server_ready(self):
        """Check if server is ready"""
        try:
            return self.client.is_server_ready()
        except Exception as e:
            print(f"Server not ready: {e}")
            return False
    
    def is_model_ready(self):
        """Check if model is ready"""
        try:
            return self.client.is_model_ready(self.model_name)
        except Exception as e:
            print(f"Model not ready: {e}")
            return False
    
    def generate_text(self, prompt, max_tokens=512, temperature=0.7):
        """Generate text using the model"""
        try:
            # Prepare inputs
            inputs = [
                httpclient.InferInput("INPUT_TEXT", [1], "BYTES"),
                httpclient.InferInput("MAX_TOKENS", [1], "INT32"),
                httpclient.InferInput("TEMPERATURE", [1], "FP32")
            ]
            
            # Set input data
            inputs[0].set_data_from_numpy(np.array([prompt.encode('utf-8')], dtype=object))
            inputs[1].set_data_from_numpy(np.array([max_tokens], dtype=np.int32))
            inputs[2].set_data_from_numpy(np.array([temperature], dtype=np.float32))
            
            # Prepare outputs
            outputs = [
                httpclient.InferRequestedOutput("OUTPUT_TEXT")
            ]
            
            # Make inference request
            start_time = time.time()
            response = self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs
            )
            inference_time = time.time() - start_time
            
            # Get result
            result = response.as_numpy("OUTPUT_TEXT")[0].decode('utf-8')
            
            return {
                "text": result,
                "inference_time": inference_time,
                "prompt": prompt
            }
            
        except Exception as e:
            return {"error": str(e)}

def main():
    print("Testing Triton Mistral Client...")
    
    client = TritonMistralClient()
    
    # Wait for server to be ready
    print("Waiting for server to be ready...")
    while not client.is_server_ready():
        time.sleep(1)
    
    print("Server is ready!")
    
    # Wait for model to be ready
    print("Waiting for model to be ready...")
    while not client.is_model_ready():
        time.sleep(1)
    
    print("Model is ready!")
    
    # Test prompts
    test_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot learning to paint.",
        "What are the benefits of renewable energy?"
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")
        
        result = client.generate_text(prompt, max_tokens=256, temperature=0.7)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Response: {result['text']}")
            print(f"Inference time: {result['inference_time']:.2f}s")

if __name__ == "__main__":
    main()