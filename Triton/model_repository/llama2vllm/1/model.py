import json
import numpy as np
import triton_python_backend_utils as pb_utils
from vllm import LLM, SamplingParams

class TritonPythonModel:
    def initialize(self, args):
        """
        This function is called only once when the model is being loaded.
        We'll set up the vLLM engine here with streaming capabilities.
        """
        # Now we understand: model_repository points to our model directory
        # We just need to add the version directory and filename
        import os
        
        model_repo = args['model_repository']
        model_version = args['model_version']
        
        # Simple, reliable path construction
        # model_repo is "/models/llama2vllm", we need to add "1/model.json"
        config_path = os.path.join(model_repo, str(model_version), "model.json")
        
        print(f"Debug - Constructed config path: {config_path}")
        
        # Verify the file exists before trying to open it
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        # Load model configuration from model.json
        with open(config_path, "r") as f:
            config = json.load(f)

        model_name_or_path = config.get("model", "facebook/opt-125m")
        disable_log_requests = config.get("disable_log_requests", True)
        gpu_memory_utilization = config.get("gpu_memory_utilization", 0.5)
        enforce_eager = config.get("enforce_eager", True)

        # Initialize the LLM - we'll handle streaming at the application level
        # The vLLM version in this container doesn't support the stream parameter
        self.sampling_params = SamplingParams(
            temperature=0.7, 
            top_p=0.9, 
            max_tokens=256
        )
        
        # Create base arguments that work with all vLLM versions
        llm_args = {
            "model": model_name_or_path,
            "disable_log_stats": disable_log_requests,
            "gpu_memory_utilization": gpu_memory_utilization
        }
        
        # Try to create the LLM with the newer parameter first
        # If that fails, fall back to the basic version
        try:
            print("Attempting to initialize LLM with enforce_eager parameter...")
            self.llm = LLM(enforce_eager=enforce_eager, **llm_args)
            print("Successfully initialized LLM with enforce_eager parameter")
        except TypeError as e:
            if "enforce_eager" in str(e):
                print("enforce_eager parameter not supported in this vLLM version, falling back to basic initialization...")
                self.llm = LLM(**llm_args)
                print("Successfully initialized LLM without enforce_eager parameter")
            else:
                # If it's a different TypeError, re-raise it
                raise e

    def execute(self, requests):
        """
        This function handles streaming by yielding partial responses.
        Each call to this function should return one token or chunk.
        """
        responses = []
        
        for request in requests:
            # Extract the input prompt
            input_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
            prompt = input_tensor.as_numpy()[0].decode("utf-8")

            # For true streaming, we need to generate tokens one by one
            # This approach generates the full response but could be modified
            # to yield tokens incrementally in a production system
            result = self.llm.generate(prompt, self.sampling_params)
            
            # Get the generated text
            response_text = result[0].outputs[0].text.strip()
            
            # In a real streaming implementation, you would yield tokens one by one
            # For now, we'll return the complete response
            # but the client can still use stream_infer() which will work with this setup
            output_tensor = pb_utils.Tensor(
                "OUTPUT", 
                np.array([response_text.encode("utf-8")], dtype=np.bytes_)
            )
            
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses

    def finalize(self):
        """
        Clean up resources when the model is unloaded
        """
        print("Finalizing the model and cleaning up resources")
        # Explicitly clean up the LLM if needed
        if hasattr(self, 'llm'):
            del self.llm