"""
Video Transcriber System - Streamlit Application
Extracts exact voice (transcription) from video files.
"""

import streamlit as st
import os
import sys
from pathlib import Path
import json

# Add project root to sys.path natively before any package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
import shutil
import asyncio
import atexit
import threading
import time

# ── Windows asyncio fix (prevents "Event loop is closed" on shutdown) ──────
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass


def graceful_shutdown():
    try:
        from transcriber.speech_understanding import _whisper_model as _wm
        if _wm is not None:
            del _wm
    except Exception:
        pass
    main_thread = threading.main_thread()
    for t in threading.enumerate():
        if t is not main_thread and not t.daemon:
            try:
                t.join(timeout=2.0)
            except Exception:
                pass


atexit.register(graceful_shutdown)

st.set_page_config(
    page_title="Video Transcriber",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

def download_youtube_to_file(url: str, status_placeholder) -> "str | None":
    try:
        import yt_dlp
    except ImportError:
        st.error("yt-dlp is not installed. Run:  pip install yt-dlp")
        return None

    tmp_dir  = Path(tempfile.mkdtemp())
    out_tmpl = str(tmp_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "outtmpl": out_tmpl,
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "quiet": True,
        "no_warnings": True,
    }

    status_placeholder.text("⬇️ Extracting YouTube audio stream…")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        candidates = list(tmp_dir.glob("*.wav"))
        if not candidates:
            candidates = [f for f in tmp_dir.glob("*.*") if f.suffix.lower() != ".webp"]
        if candidates:
            return str(candidates[0])

        st.error("Download completed but output WAV file not found.")
    except Exception as exc:
        st.error(f"YouTube download failed: {exc}")
    return None

def main():
    st.title("🎬 Video Transcriber")
    st.markdown("Upload a video or enter a YouTube URL to extract exact transcription.")
    st.markdown("---")

    with st.sidebar:
        st.header("📊 System Status")
        st.info("System Ready")

    st.subheader("📤 Upload & Process Video")

    input_method = st.radio(
        "Choose input method:",
        ["Upload Video File", "YouTube URL", "Use Existing Videos"],
        horizontal=True,
    )

    video_path: "str | None"  = None
    youtube_url: "str | None" = None

    if input_method == "Upload Video File":
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "mkv", "avi", "mov", "webm", "mp3", "wav", "m4a"],
        )
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                shutil.copyfileobj(uploaded_file, tmp)
                video_path = tmp.name
            st.success(f"✅ Ready to process: {uploaded_file.name}")

    elif input_method == "YouTube URL":
        youtube_url = st.text_input(
            "Enter YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=...",
        )

    elif input_method == "Use Existing Videos":
        video_dir = Path("data/videos")
        if video_dir.exists():
            videos = sorted(
                list(video_dir.glob("*.mp4"))
                + list(video_dir.glob("*.mkv"))
                + list(video_dir.glob("*.avi"))
            )
            if videos:
                selected = st.selectbox("Select video:", [str(v) for v in videos])
                video_path = selected
            else:
                st.warning("No videos found in data/videos/")
        else:
            st.warning("data/videos/ directory not found")

    ready_to_process = bool(video_path or youtube_url)
    if ready_to_process:
        if st.button("🚀 Process Video", type="primary", use_container_width=True):
            _run_process_video(video_path=video_path, youtube_url=youtube_url)

def _run_process_video(video_path: "str | None", youtube_url: "str | None" = None):
    progress_bar = st.progress(0)
    status_text  = st.empty()

    try:
        if youtube_url:
            video_path = download_youtube_to_file(youtube_url, status_text)
            if video_path is None:
                return
            progress_bar.progress(15)

        from transcriber.video_processor import convert_video_to_audio
        from transcriber.speech_understanding import transcribe_audio

        status_text.text("🎵 Extracting audio from video…")
        progress_bar.progress(30)
        audio_path = convert_video_to_audio(video_path, force=True)
        if not audio_path:
            st.error("Audio extraction failed.")
            return

        status_text.text("🎤 Transcribing speech with Whisper…")
        progress_bar.progress(60)
        transcript_path = transcribe_audio(audio_path)
        
        if not transcript_path:
            st.error("Transcription failed.")
            return

        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()

        st.success("✅ Transcription Complete!")
        
        # Display transcript
        with open(transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        full_text = data.get("full_transcript", "")
        
        st.subheader("📝 Exact Voice Transcript")
        
        # Download button for easy access to full raw text
        st.download_button(
            label="⬇️ Download Full Transcript as TXT",
            data=full_text,
            file_name=f"transcript_{Path(video_path).stem}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        # Text area for easy copy-pasting
        st.text_area("Full Transcript (Copy-Paste Ready)", full_text, height=400)
            
    except Exception as exc:
        st.error(f"Error during processing: {exc}")
        progress_bar.empty()
        status_text.empty()

if __name__ == "__main__":
    main()

