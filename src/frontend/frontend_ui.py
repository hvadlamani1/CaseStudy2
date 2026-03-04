import gradio as gr
import requests
import time
import os

#Where to send the backend requests
BACKEND_API_URL = "http://paffenroth-23.dyn.wpi.edu:9001/process_audio"


def transcribe_audio_ui(audio_file):
    if audio_file is None:
        yield "Please upload an audio file", "Please upload an audio file"
        return

    # Let the user know the request is in transit
    yield gr.update(value="Sending audio to backend VM for transcription...", label="Step 1: Raw ATC Transcription"), \
        gr.update(value="Waiting...", label="Step 2: Plain English Interpretation")

    # Prepare the payload to send to your FastAPI backend
    payload_data = {}

    # Safely grab the token from the environment variable we exported
    hf_env_token = os.environ.get("HF_TOKEN")
    if hf_env_token:
        payload_data["hf_token"] = hf_env_token

    try:
        # Open the audio file and send it via HTTP POST
        with open(audio_file, "rb") as f:
            files = {"audio_file": (os.path.basename(audio_file), f, "audio/wav")}

            t0 = time.time()
            response = requests.post(BACKEND_API_URL, files=files, data=payload_data)
            t1 = time.time()

        # Handle the API Response
        if response.status_code == 200:
            result = response.json()

            transcription = result.get("transcription", "Error parsing transcription")
            translation = result.get("translation", "Error parsing translation")

            # Formatted labels with timing info
            trans_label = f"Step 1: Raw ATC Transcription ({result.get('transcription_time_sec', 0)}s compute)"
            interp_label = f"Step 2: Plain English Interpretation ({result.get('translation_time_sec', 0)}s compute)"

            yield gr.update(value=transcription, label=trans_label), \
                gr.update(value=translation, label=interp_label)

        else:
            # Handle backend errors (e.g., 500 Internal Server Error)
            error_msg = response.json().get("detail", response.text)
            yield f"Backend API Error: {error_msg}", f"Backend API Error: {error_msg}"

    except requests.exceptions.ConnectionError:
        error_msg = f"Failed to connect to backend at {BACKEND_API_URL}. Is your FastAPI server running?"
        yield gr.update(value=error_msg, label="Connection Error"), gr.update(value=error_msg, label="Connection Error")
    except Exception as e:
        yield f"Unexpected UI Error: {str(e)}", f"Error: {str(e)}"


# --- Gradio Interface ---
grInt = gr.Interface(
    fn=transcribe_audio_ui,
    inputs=[
        gr.Audio(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Step 1: Raw ATC Transcription"),
        gr.Textbox(label="Step 2: Plain English Interpretation")
    ],
    title="ATC Speech Transcription (Frontend UI)",
    description="This UI sends audio to the backend VM for processing to keep the frontend lightweight.",
    examples=None,
    cache_examples=False,
    flagging_mode="never"
)

with gr.Blocks() as demo:
    grInt.render()

if __name__ == "__main__":
    # The shared frontend runs the UI. Keep this on 0.0.0.0 so it's accessible.
    demo.launch(server_name="0.0.0.0", server_port=7001, share=True, show_api=False)
