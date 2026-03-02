import os
import time
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from huggingface_hub import InferenceClient

# Initialize FastAPI App
app = FastAPI(
    title="ATC Speech Transcription API",
    description="Backend API for processing Air Traffic Control audio into plain English.",
    version="1.0.0"
)

# --- API Endpoint ---
@app.post("/process_audio")
async def process_audio(
    audio_file: UploadFile = File(...),
    hf_token: str = Form(None)
):
    """
    Accepts an audio file and returns the ATC transcription and plain English translation.
    """
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="wb") as temp_audio:
        content = await audio_file.read()
        temp_audio.write(content)
        temp_audio_path = temp_audio.name

    try:
        t0 = time.time()
        
        # API Mode
        if not hf_token:
            raise HTTPException(status_code=401, detail="Hugging Face token required for API mode.")
        
        client = InferenceClient(token=hf_token)
        asr_result = client.automatic_speech_recognition(temp_audio_path, model="openai/whisper-large-v3-turbo")
        transcription = getattr(asr_result, "text", asr_result.get("text") if isinstance(asr_result, dict) else str(asr_result))
        
        t1 = time.time()
        
        messages = [
            {"role": "system", "content": "You are an expert aviation translator. Your ONLY job is to take the provided ATC transmission and rewrite it into simple, conversational plain English for a non-pilot.\nRules:\n1. DO NOT add any extra dialogue or respond to the transmission.\n2. DO NOT pretend to be ATC.\n3. DO NOT expand or add any new information.\n4. ONLY provide the direct translation.\n\nExamples:\nInput: 'Delta 123 heavy, cleared ILS runway 27R approach.'\nOutput: 'Delta flight 123, you are cleared to land using the instruments on runway 27 Right.'\n\nInput: 'Mayday, mayday, mayday. I'm going down.'\nOutput: 'Emergency, emergency, emergency. The aircraft is crashing.'"},
            {"role": "user", "content": f"Input: '{transcription}'\nOutput:"}
        ]
        chat_completion = client.chat_completion(messages, model="meta-llama/Llama-3.2-1B-Instruct", max_tokens=256)
        translation = chat_completion.choices[0].message.content
        t2 = time.time()

        # Clean up temp file
        os.remove(temp_audio_path)

        # Return standard JSON response
        return JSONResponse(content={
            "transcription": transcription,
            "translation": translation,
            "transcription_time_sec": round(float(t1 - t0), 2),
            "translation_time_sec": round(float(t2 - t1), 2)
        })

    except Exception as e:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        raise HTTPException(status_code=500, detail=str(e))