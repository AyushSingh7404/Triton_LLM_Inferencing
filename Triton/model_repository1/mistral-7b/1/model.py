from huggingface_hub import login

login("your_huggingface_token_here")

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import triton_python_backend_utils as pb_utils
from torch.cuda.amp import autocast
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TritonPythonModel:
    def initialize(self, args):
        """Initialize the model"""
        logger.info("Initializing Mistral-7B model...")
        
        # Parse model config
        self.model_config = model_config = json.loads(args['model_config'])
        
        # Get model parameters
        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(model_config)
        if using_decoupled:
            raise pb_utils.TritonModelException("This model doesn't support decoupled mode")
        
        # Model configuration
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations for Tesla T4
        logger.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # Use FP16 for T4
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=True
        )
        
        # Optimize model for inference
        self.model.eval()
        if hasattr(self.model, 'generation_config'):
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        
        logger.info("Model initialization complete")
    
    def execute(self, requests):
        """Execute inference on batch of requests"""
        responses = []

        for request in requests:
            # Get input text
            input_text_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            input_text_arr = input_text_tensor.as_numpy()  # shape: [batch_size, seq_len]

            # Decode first example (batch 0)
            prompt_bytes = input_text_arr[0][0]
            prompt_str = prompt_bytes.decode("utf-8")

            # Get optional parameters
            max_tokens = 512  # default
            temperature = 0.7  # default

            max_tokens_tensor = pb_utils.get_input_tensor_by_name(request, "MAX_TOKENS")
            if max_tokens_tensor is not None:
                max_tokens = int(max_tokens_tensor.as_numpy()[0][0])

            temperature_tensor = pb_utils.get_input_tensor_by_name(request, "TEMPERATURE")
            if temperature_tensor is not None:
                temperature = float(temperature_tensor.as_numpy()[0][0])

            # Format prompt for Mistral
            formatted_prompt = f"<s>[INST] {prompt_str} [/INST]"

            try:
                # Generate response
                output_text = self._generate_response(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                # Create output tensor
                output_tensor = pb_utils.Tensor(
                    "OUTPUT_TEXT",
                    np.array([[output_text.encode('utf-8')]], dtype=np.object_)
                )

                response = pb_utils.InferenceResponse(output_tensors=[output_tensor])

            except Exception as e:
                logger.error(f"Error during inference: {str(e)}")
                error_msg = f"Error: {str(e)}"
                output_tensor = pb_utils.Tensor(
                    "OUTPUT_TEXT",
                    np.array([[error_msg.encode('utf-8')]], dtype=np.object_)
                )
                response = pb_utils.InferenceResponse(output_tensors=[output_tensor])

            responses.append(response)

        return responses

    
    def _generate_response(self, prompt, max_tokens=512, temperature=0.7):
        """Generate response using the model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            ).to(self.device)
            
            # Generate with optimized settings for T4
            with torch.no_grad():
                with autocast():  # Use automatic mixed precision
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return response.strip()
            
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory error")
            return "Error: GPU memory insufficient. Try reducing max_tokens or input length."
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return f"Error during text generation: {str(e)}"
    
    def finalize(self):
        """Clean up resources"""
        logger.info("Finalizing model...")
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()
