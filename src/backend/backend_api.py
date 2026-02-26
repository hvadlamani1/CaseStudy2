import os
import time
import tempfile
import numpy as np
import soundfile as sf
import librosa
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import InferenceClient

# Initialize FastAPI App
app = FastAPI(
    title="ATC Speech Transcription API",
    description="Backend API for processing Air Traffic Control audio into plain English.",
    version="1.0.0"
)

# --- Global Variables & Device Detection ---
model = None
processor = None
atc_translator = None

def detect_device():
    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    elif torch.backends.mps.is_available():
        return "mps", torch.float32
    else:
        return "cpu", torch.float32

device, torch_dtype = detect_device()

def load_resources():
    global model, processor, atc_translator
    if model is None:
        print("Loading Whisper model...")
        model = WhisperForConditionalGeneration.from_pretrained(
            "tclin/whisper-large-v3-turbo-atcosim-finetune",
            torch_dtype=torch_dtype
        ).to(device)
        processor = WhisperProcessor.from_pretrained("tclin/whisper-large-v3-turbo-atcosim-finetune")

    if atc_translator is None:
        print("Loading Translator model...")
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )
        atc_translator = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)

def atc_english_translation(atc_prompt):
    if not atc_prompt or "Error" in atc_prompt:
        return "Waiting for transcription"

    messages = [
        {"role": "system", "content": "You are an aviation expert. Translate the following technical ATC radio transmission into simple, conversational plain English. Do not give definitions, just simply translate to conversational english! Be concise."},
        {"role": "user", "content": atc_prompt}
    ]
    prompt = atc_translator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = atc_translator(prompt, do_sample=False, max_new_tokens=256, return_full_text=False)
    return outputs[0]['generated_text'].strip()


# --- API Endpoint ---
@app.post("/process_audio")
async def process_audio(
    audio_file: UploadFile = File(...),
    use_local_model: bool = Form(False),
    hf_token: str = Form(None)
):
    """
    Accepts an audio file and returns the ATC transcription and plain English translation.
    """
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        content = await audio_file.read()
        temp_audio.write(content)
        temp_audio_path = temp_audio.name

    try:
        t0 = time.time()
        
        if use_local_model:
            load_resources()
            speech, sample_rate = sf.read(temp_audio_path)
            speech = speech.astype(np.float32)

            if len(speech.shape) > 1:
                speech = np.mean(speech, axis=1)

            if sample_rate != 16000:
                speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)

            input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(device=device, dtype=torch_dtype)

            generated_ids = model.generate(input_features, max_new_tokens=128, repetition_penalty=1.1)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            t1 = time.time()
            translation = atc_english_translation(transcription)
            t2 = time.time()

        else:
            # API Mode
            if not hf_token:
                raise HTTPException(status_code=401, detail="Hugging Face token required for API mode.")
            
            client = InferenceClient(token=hf_token)
            asr_result = client.automatic_speech_recognition(temp_audio_path, model="openai/whisper-large-v3-turbo")
            transcription = getattr(asr_result, "text", asr_result.get("text") if isinstance(asr_result, dict) else str(asr_result))
            
            t1 = time.time()
            
            messages = [
                {"role": "system", "content": "You are an aviation expert. Translate the following technical ATC radio transmission into simple, conversational plain English. Do not give definitions, just simply translate to conversational english! Be concise."},
                {"role": "user", "content": transcription}
            ]
            chat_completion = client.chat_completion(messages, model="Qwen/Qwen2.5-1.5B-Instruct", max_tokens=256)
            translation = chat_completion.choices[0].message.content
            t2 = time.time()

        # Clean up temp file
        os.remove(temp_audio_path)

        # Return standard JSON response
        return JSONResponse(content={
            "transcription": transcription,
            "translation": translation,
            "transcription_time_sec": round(t1 - t0, 2),
            "translation_time_sec": round(t2 - t1, 2)
        })

    except Exception as e:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        raise HTTPException(status_code=500, detail=str(e))