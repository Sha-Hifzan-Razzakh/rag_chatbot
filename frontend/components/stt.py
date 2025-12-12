from json import load
import os
import requests
import streamlit as st
#---------------------------------------------------------------------------
# Speech to Text component

def stt_widget(backend_url: str):
    st.subheader("🎙️ Speech to Text")
    audio_file = st.file_uploader(
        "Upload an audio file (wav/mp3/m4a)", type=["wav", "mp3", "m4a"]
    )

    if audio_file is not None and st.button("Transcribe"):
        with st.spinner("Transcribing audio..."):
            files = {"file": (audio_file.name, audio_file, audio_file.type)}
            resp = requests.post(f"{backend_url}/v1/audio/stt", files=files)
        if resp.status_code == 200:
            data = resp.json()
            st.success("Transcription complete")
            # show and use transcription
            st.write(data["text"])
            # Optionally pre-fill chat input text – you can store it in session_state
            st.session_state["pending_input"] = data["text"]
        else:
            st.error(f"Error: {resp.text}")
