import base64
import json
import os
from io import StringIO
from typing import Dict, Any

import torch
from transformers import pipeline


class EndpointHandler:

    def __init__(self, asr_model_path: str = "./whisper-german-v3-endpoint"):
        device = 0 if torch.cuda.is_available() else -1
        print("Using device:", device)
        # Create an ASR pipeline using the model located in the specified directory
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model = asr_model_path,
            device = device
        )

    def __call__(self, data: Dict[str, Any]) -> str:

        if "audio_data" not in data.keys():
            raise Exception("Request must contain a top-level key named 'audio_data'")

        # Get the audio data from the input
        audio_data = data["audio_data"]
        options = data["options"]

        # Decode the binary audio data if it's provided as a base64 string
        if isinstance(audio_data, str):
            audio_data = base64.b64decode(audio_data)

        # Process the audio data with the ASR pipeline
        transcription = self.asr_pipeline(
            audio_data,
            return_timestamps = True,
            chunk_length_s = 60,
            batch_size = 8,
            max_new_tokens = 10000,
            generate_kwargs = options
        )

        # Convert the transcription to JSON
        result = StringIO()
        json.dump(transcription, result)

        return result.getvalue()