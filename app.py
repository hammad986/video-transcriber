import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path

import gradio as gr
import yt_dlp
from faster_whisper import WhisperModel

# Patch gradio_client schema reflection bug with boolean schemas
try:
    import gradio_client.utils
    _orig_get_type = gradio_client.utils.get_type
    def _safe_get_type(schema):
        if isinstance(schema, bool):
            return "boolean"
        return _orig_get_type(schema)
    gradio_client.utils.get_type = _safe_get_type
except Exception:
    pass

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"Loading WhisperModel({model_size}) on {device} ({compute_type})...")
        try:
            MODELS[model_size] = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            print(f"Failed to load on {device}: {e}. Falling back to CPU int8...")
            MODELS[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
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
        "format": "ba/b",
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios"]
            }
        },
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
        candidates = [f for f in tmp_dir.glob("*.*") if f.suffix.lower() not in [".webp", ".jpg", ".png", ".json"]]
    if candidates:
        return str(candidates[0])
    raise RuntimeError("Failed to extract audio from YouTube URL.")

def convert_to_wav(file_path: str) -> str:
    """Extract audio from video or convert audio to 16kHz WAV for Faster-Whisper"""
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    out_dir = Path(tempfile.mkdtemp())
    out_wav = str(out_dir / "audio_16k.wav")
    try:
        cmd = [
            "ffmpeg", "-y", "-i", file_path,
            "-vn", "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le", out_wav
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0 and os.path.exists(out_wav) and os.path.getsize(out_wav) > 0:
            return out_wav
    except Exception as err:
        print(f"ffmpeg conversion warning: {err}")
    
    return file_path

def transcribe_core(audio_path: str, model_size: str):
    model = get_whisper_model(model_size)
    segments, info = model.transcribe(audio_path, beam_size=2, vad_filter=True)
    seg_list = []
    for s in segments:
        seg_list.append({
            "start": s.start,
            "end": s.end,
            "text": s.text.strip()
        })
    return seg_list, info.language.upper(), info.language_probability

# ZeroGPU decorator
try:
    if HAS_GPU:
        transcribe_core = spaces.GPU(duration=120)(transcribe_core)
except Exception:
    pass

def process_transcription(source_mode, input_file, youtube_url, model_size, progress=gr.Progress(track_tqdm=True)):
    tmp_files = []
    try:
        audio_path = None
        if source_mode == "Upload File":
            if not input_file:
                return "⚠️ Please upload a video or audio file first.", None, None
            progress(0.1, desc="Preparing uploaded media...")
            audio_path = convert_to_wav(input_file)
            if audio_path != input_file:
                tmp_files.append(audio_path)
        else:
            if not youtube_url or not youtube_url.strip():
                return "⚠️ Please enter a valid YouTube URL first.", None, None
            progress(0.1, desc="Downloading YouTube audio...")
            try:
                audio_path = download_youtube_audio(youtube_url.strip())
                tmp_files.append(audio_path)
            except Exception as yt_err:
                err_msg = str(yt_err)
                if "Sign in to confirm you're not a bot" in err_msg:
                    return (
                        "❌ YouTube Bot Wall: YouTube detected a cloud server IP and requires bot verification.\n\n"
                        "💡 Solution: Please download or have the video/audio file and upload it directly using the 'Upload File' option above!",
                        None, None
                    )
                return f"❌ Error downloading YouTube video: {err_msg}", None, None

        progress(0.4, desc=f"Loading Faster-Whisper ({model_size})...")
        progress(0.6, desc="Transcribing speech to text...")
        seg_list, lang, lang_prob = transcribe_core(audio_path, model_size)

        if not seg_list:
            return "No speech detected in the provided media file.", None, None

        full_text_list = []
        srt_lines = []
        for idx, seg in enumerate(seg_list, start=1):
            full_text_list.append(seg["text"])
            start_str = format_timestamp(seg["start"])
            end_str = format_timestamp(seg["end"])
            srt_lines.append(f"{idx}\n{start_str} --> {end_str}\n{seg['text']}\n")

        full_transcript = " ".join(full_text_list)
        srt_content = "\n".join(srt_lines)

        progress(0.9, desc="Generating output files...")
        out_dir = Path(tempfile.mkdtemp())
        txt_path = str(out_dir / "transcript.txt")
        srt_path = str(out_dir / "transcript.srt")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_transcript)
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        summary = f"🌐 Detected Language: {lang} (Confidence: {lang_prob:.0%})\n\n{full_transcript}"
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

def update_visibility(mode):
    if mode == "Upload File":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

# Custom CSS for modern clean UI
custom_css = """
.container { max-width: 950px; margin: auto; }
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
                source_mode = gr.Radio(
                    choices=["Upload File", "YouTube URL"],
                    value="Upload File",
                    label="Choose Input Source"
                )
                input_file = gr.File(
                    label="Upload Video or Audio (MP4, MKV, MOV, MP3, WAV)",
                    file_types=[".mp4", ".mkv", ".mov", ".avi", ".webm", ".mp3", ".wav", ".m4a"],
                    visible=True
                )
                youtube_url = gr.Textbox(
                    label="Enter YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    visible=False
                )
                
                source_mode.change(
                    fn=update_visibility,
                    inputs=[source_mode],
                    outputs=[input_file, youtube_url]
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
            inputs=[source_mode, input_file, youtube_url, model_size],
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
