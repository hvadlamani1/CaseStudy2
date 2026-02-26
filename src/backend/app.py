import gradio as gr
import time
import torch
import torchaudio
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import InferenceClient

import os

import soundfile as sf
import librosa

model = None
processor = None
atc_translator = None

# --- Detecting Device ---
def detect_device():
    # Check for CUDA (Nvidia)
    if torch.cuda.is_available():
        print("Status: Detected Nvidia GPU. Using CUDA (float16).")
        return "cuda:0", torch.float16

    # Check for MPS (Apple)
    elif torch.backends.mps.is_available():
        print("Status: Detected Apple Silicon. Using MPS (float32) to prevent loops.")
        return "mps", torch.float32

    # Fallback to CPU incase no GPU is avail
    else:
        print("Status: No GPU detected. Using CPU (slow).")
        return "cpu", torch.float32

# Detect device immediately for Gradio interface labels
device, torch_dtype = detect_device()

# --- Loading Local Model ---
def load_resources():
    global model, processor, atc_translator, device, torch_dtype
    
    if model is None:
        print("Loading Whisper model...")
        loaded_model = WhisperForConditionalGeneration.from_pretrained(
            "tclin/whisper-large-v3-turbo-atcosim-finetune",
            torch_dtype=torch_dtype
        )
        model = loaded_model.to(device)
        processor = WhisperProcessor.from_pretrained("tclin/whisper-large-v3-turbo-atcosim-finetune")

    if atc_translator is None:
        print("Loading Translator model")
        # Using Qwen for translation
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto", 
            device_map="auto"
        )
        
        atc_translator = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)


# --- Translation Logic ---
def atc_english_translation(atc_prompt):
    load_resources()
    if not atc_prompt or "Error" in atc_prompt:
        return "Waiting for transcription"

    # Qwen uses a standard chat template
    messages = [
        {"role": "system", "content": "You are an aviation expert. Translate the following technical ATC radio transmission into simple, conversational plain English. Do not give definitions, just simply translate to conversational english! Be concise."},
        {"role": "user", "content": atc_prompt}
    ]
    
    # We use the tokenizer's chat template for the best results
    prompt = atc_translator.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    try:
        outputs = atc_translator(
            prompt,
            do_sample=False,
            max_new_tokens=256,
            return_full_text=False
        )
        return outputs[0]['generated_text'].strip()
    except Exception as e:
        return f"Translation Error: {str(e)}"

# --- Transcription Logic ---
def transcribe_audio(audio_file, use_local_model, hf_token: gr.OAuthToken = None):
    if audio_file is None:
        return "Please upload an audio file", "Please upload an audio file"

    if use_local_model:
        load_resources()
        try:
            t0 = time.time()
            
            # Use soundfile to load. It bypasses the libtorchcodec system errors.
            speech, sample_rate = sf.read(audio_file)
            
            # Ensure it's float32 for whisper
            speech = speech.astype(np.float32)

            # Convert Stereo to Mono
            if len(speech.shape) > 1:
                speech = np.mean(speech, axis=1)

            # re-sample to different hz
            if sample_rate != 16000:
                speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)

            # Prepare for Whisper
            input_features = processor(
                speech,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features

            # Move to device 
            input_features = input_features.to(device=device, dtype=torch_dtype)

            # Generate transcription
            generated_ids = model.generate(
                input_features,
                max_new_tokens=128,
                repetition_penalty=1.1
            )

            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            t1 = time.time()
            models_time = f"Metadata: Whisper took {t1 - t0:.2f}s"
            
            yield gr.update(value=transcription, label=f"Step 1: Raw ATC Transcription ({t1 - t0:.2f}s)"), \
                  gr.update(value="Waiting for translation...", label="Step 2: Plain English Interpretation")

            translated_text = atc_english_translation(transcription)

            t2 = time.time()
            
            yield gr.update(value=transcription, label=f"Step 1: Raw ATC Transcription ({t1 - t0:.2f}s)"), \
                  gr.update(value=translated_text, label=f"Step 2: Plain English Interpretation ({t2 - t1:.2f}s)")

        except Exception as e:
            yield f"Error processing audio: {str(e)}", f"Error: {str(e)}"
    
    else:
        # API MODE
        print("[MODE] api")

        if hf_token is None or not hf_token.token:
            yield "⚠️ Please log in with your Hugging Face account first.", "Please log in."
            return

        client = InferenceClient(token=hf_token.token)

        try:
            t0 = time.time()
            
            # API Whisper
            asr_result = client.automatic_speech_recognition(
                audio_file, 
                model="openai/whisper-large-v3-turbo"
            )
            transcription = getattr(asr_result, "text", asr_result.get("text") if isinstance(asr_result, dict) else str(asr_result))
            
            t1 = time.time()
            
            yield gr.update(value=transcription, label=f"Step 1: Raw ATC Transcription ({t1 - t0:.2f}s)"), \
                  gr.update(value="Waiting for translation...", label="Step 2: Plain English Interpretation")
            
            # API Qwen
            messages = [
                {"role": "system", "content": "You are an aviation expert. Translate the following technical ATC radio transmission into simple, conversational plain English. Do not give definitions, just simply translate to conversational english! Be concise."},
                {"role": "user", "content": transcription}
            ]
            
            chat_completion = client.chat_completion(
                messages,
                model="Qwen/Qwen2.5-1.5B-Instruct", 
                max_tokens=256
            )
            
            translated_text = chat_completion.choices[0].message.content
            
            t2 = time.time()
            
            yield gr.update(value=transcription, label=f"Step 1: Raw ATC Transcription ({t1 - t0:.2f}s)"), \
                  gr.update(value=translated_text, label=f"Step 2: Plain English Interpretation ({t2 - t1:.2f}s)")

        except Exception as e:
            error_msg = str(e)
            print(f"API Error: {error_msg}")
            
            if "403" in error_msg or "permissions" in error_msg.lower():
                yield (
                    f"⚠️ API Permission Error: {error_msg}\n\n"
                )
                return
            
            yield f"API Error: {error_msg}", "API Error"


# --- Gradio Interface ---
grInt = gr.Interface(
    fn=transcribe_audio,
    inputs=[
        gr.Audio(type="filepath"),
        gr.Checkbox(label="Use Local Model", value=False)
    ],
    outputs=[
        gr.Textbox(label="Step 1: Raw ATC Transcription"),
        gr.Textbox(label="Step 2: Plain English Interpretation")
    ],
    title="ATC Speech Transcription",
    description=f"Running on: {device.upper()}",
    
    examples=None, 
    
    cache_examples=False,
    allow_flagging="never"
)

with gr.Blocks() as demo:
    gr.LoginButton()
    grInt.render()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_api=False)