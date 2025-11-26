import os
import numpy as np
import triton_python_backend_utils as pb_utils

# Lazy load TTS model
_tts = None
_sample_rate = 22050

def _ensure_tts_loaded():
    global _tts, _sample_rate
    if _tts is not None:
        return
    from tortoise.api import TextToSpeech
    _tts = TextToSpeech()
    _sample_rate = 22050

class TritonPythonModel:
    def initialize(self, args):
        _ensure_tts_loaded()

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get inputs as uint8 arrays
            text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            voice_tensor = pb_utils.get_input_tensor_by_name(request, "VOICE_NAME")
            preset_tensor = pb_utils.get_input_tensor_by_name(request, "PRESET")

            def _to_str(t, default=""):
                if t is None:
                    return default
                arr = t.as_numpy()
                if arr.size == 0:
                    return default
                # Convert uint8 array to bytes, then decode
                return bytes(arr).decode("utf-8") if arr.ndim == 1 else default

            text = _to_str(text_tensor, default="")
            voice_name = _to_str(voice_tensor, default="tom")
            preset = _to_str(preset_tensor, default="fast")

            if not text:
                err = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError("TEXT input is required")
                )
                responses.append(err)
                continue

            # Run TTS
            import torch
            with torch.no_grad():
                wav = _tts.tts(text=text, voice=voice_name, preset=preset)

            # Ensure mono float32 numpy array
            if hasattr(wav, "cpu"):
                wav = wav.squeeze().cpu().numpy().astype(np.float32)
            else:
                wav = np.asarray(wav).astype(np.float32).squeeze()

            audio_out = pb_utils.Tensor("AUDIO", wav)
            sr_out = pb_utils.Tensor("SAMPLE_RATE", np.array([_sample_rate], dtype=np.int32))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[audio_out, sr_out]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        pass
