import streamlit as st
from faster_whisper import WhisperModel
import tempfile, os

@st.cache_resource
def load_model():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

model = load_model()

def ts(sec):
    ms = int(sec * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

st.title("Subtitle Generator")
file = st.file_uploader("Upload audio or video", type=["mp4","mp3","wav","mkv","mov"])

if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
        tmp.write(file.read())
        path = tmp.name

    st.info("Transcribing... please wait")
    segments, info = model.transcribe(path)

    srt = ""
    for i, seg in enumerate(segments, 1):
        srt += f"{i}\n{ts(seg.start)} --> {ts(seg.end)}\n{seg.text.strip()}\n\n"

    st.success("Done!")
    st.download_button("Download SRT", srt, file_name="subtitles.srt")
    os.remove(path)
