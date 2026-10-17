import os
import sys
import tempfile
import shutil
from pathlib import Path

import gradio as gr
import yt_dlp
from faster_whisper import WhisperModel

try:
    import spaces
    import torch
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False

# Cache models to avoid reloading
MODELS = {}

def get_whisper_model(model_size="base"):
    if model_size not in MODELS:
        device = "cuda" if HAS_GPU else "cpu"
        compute_type = "float16" if HAS_GPU else "int8"
        print(f"Loading WhisperModel({model_size}) on {device} ({compute_type})...")
        MODELS[model_size] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return MODELS[model_size]

def format_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def download_youtube_audio(url: str) -> str:
    tmp_dir = Path(tempfile.mkdtemp())
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
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    candidates = list(tmp_dir.glob("*.wav"))
    if not candidates:
        candidates = [f for f in tmp_dir.glob("*.*") if f.suffix.lower() != ".webp"]
    if candidates:
        return str(candidates[0])
    raise RuntimeError("Failed to extract audio from YouTube URL.")

def process_transcription(input_file, youtube_url, model_size, progress=gr.Progress(track_tqdm=True)):
    if not input_file and not youtube_url:
        return "Please upload a media file or provide a YouTube URL.", None, None

    audio_path = None
    tmp_files = []

    try:
        progress(0.1, desc="Preparing audio source...")
        if youtube_url and youtube_url.strip():
            progress(0.2, desc="Downloading YouTube audio...")
            audio_path = download_youtube_audio(youtube_url.strip())
            tmp_files.append(audio_path)
        elif input_file:
            audio_path = input_file
        
        progress(0.4, desc=f"Loading Faster-Whisper ({model_size})...")
        model = get_whisper_model(model_size)

        progress(0.6, desc="Transcribing speech to text...")
        segments, info = model.transcribe(audio_path, beam_size=2, vad_filter=True)

        full_text_list = []
        srt_lines = []
        srt_idx = 1

        for seg in segments:
            full_text_list.append(seg.text.strip())
            start_str = format_timestamp(seg.start)
            end_str = format_timestamp(seg.end)
            srt_lines.append(f"{srt_idx}\n{start_str} --> {end_str}\n{seg.text.strip()}\n")
            srt_idx += 1

        full_transcript = " ".join(full_text_list)
        srt_content = "\n".join(srt_lines)

        progress(0.9, desc="Exporting files...")
        out_dir = Path(tempfile.mkdtemp())
        txt_path = str(out_dir / "transcript.txt")
        srt_path = str(out_dir / "transcript.srt")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_transcript)
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        summary = f"Detected Language: {info.language.upper()} (Probability: {info.language_probability:.2f})\n\n{full_transcript}"
        return summary, txt_path, srt_path

    except Exception as e:
        return f"Error during transcription: {str(e)}", None, None
    finally:
        for f in tmp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass

# ZeroGPU decorator when spaces module is active
try:
    if HAS_GPU:
        process_transcription = spaces.GPU(process_transcription)
except Exception:
    pass

# Custom CSS for modern clean UI
custom_css = """
.container { max-width: 950px; margin: auto; }
.header-box { text-align: center; margin-bottom: 1.5rem; }
"""

with gr.Blocks(title="AI Video & Audio Transcriber", css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_classes=["container"]):
        gr.Markdown(
            """
            # 🎬 AI-Powered Video & Audio Transcriber
            Convert speech from **Video files, Audio recordings, or YouTube URLs** into exact text transcripts and timestamped SRT subtitles.
            *Powered by Faster-Whisper & CTranslate2*
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                input_file = gr.File(
                    label="Upload Video or Audio",
                    file_types=["video", "audio"],
                    type="filepath"
                )
                youtube_url = gr.Textbox(
                    label="Or Enter YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=..."
                )
                model_size = gr.Dropdown(
                    label="Whisper Model Size",
                    choices=["tiny", "base", "small", "medium"],
                    value="base",
                    info="Larger models improve accuracy; 'base' and 'small' provide the ideal speed/accuracy balance."
                )
                transcribe_btn = gr.Button("🚀 Transcribe Now", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="Exact Transcript Output",
                    lines=14,
                    show_copy_button=True,
                    placeholder="Transcript will appear here once processing completes..."
                )
                with gr.Row():
                    download_txt = gr.File(label="Download Transcript (TXT)")
                    download_srt = gr.File(label="Download Subtitles (SRT)")

        transcribe_btn.click(
            fn=process_transcription,
            inputs=[input_file, youtube_url, model_size],
            outputs=[output_text, download_txt, download_srt]
        )
        
        gr.Markdown(
            """
            ---
            Built by **Muhammed Hammad** | [GitHub](https://github.com/hammad986) | [LinkedIn](https://linkedin.com/in/hammad986)
            """
        )

if __name__ == "__main__":
    demo.launch()
