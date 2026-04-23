#!/usr/bin/env python3
"""
PC receiver for Raspberry Pi audio uploads with real-time Whisper transcription.
Run on your laptop/desktop:
    python3 pc_receiver.py
"""

from __future__ import annotations

import os
import signal
import threading
import time
import shutil
import wave
import socket
from dataclasses import dataclass
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from werkzeug.serving import make_server
import torch
from transformers import pipeline

app = Flask(__name__)
SAVE_DIR = os.path.join(os.getcwd(), "received_audio")
MODEL_ID = "openai/whisper-large-v3-turbo"
POLL_INTERVAL_SECONDS = 0.05
CLEANUP_EVERY_TRANSCRIBED = 10
TARGET_SAMPLING_RATE = 16000
GRADIO_HOST = "0.0.0.0"
GRADIO_PORT = 7860
GRADIO_REFRESH_SECONDS = 1
MAX_TRANSCRIPT_LINES = 300
AUDIO_LEVEL_WINDOW_SIZE = 120
MODEL_MIN_PEAK_FOR_ASR = 3500.0
MODEL_MIN_DBFS_FOR_ASR = -50.0
MODEL_ACTIVITY_ABS_AMPLITUDE = 700.0
MODEL_MIN_ACTIVITY_RATIO_FOR_ASR = 0.01
MODEL_MAX_WORDS_PER_SECOND = 7.0
MODEL_WORD_BURST_ALLOWANCE = 8
MODEL_MAX_CONSECUTIVE_SAME_WORD = 4
MODEL_MIN_WORDS_FOR_DIVERSITY_CHECK = 18
MODEL_MIN_UNIQUE_WORD_RATIO = 0.35


@dataclass
class ChunkRecord:
    file_name: str
    file_path: str
    received_at: float
    rms: float = 0.0
    peak: float = 0.0
    dbfs: float = -120.0
    in_progress: bool = False
    transcribed: bool = False
    text: str = ""


# Rolling buffer and coordination.
file_buffer: deque[ChunkRecord] = deque()
buffer_lock = threading.Lock()
shutdown_event = threading.Event()
asr_pipeline = None
transcribed_since_cleanup = 0
transcript_log: deque[str] = deque(maxlen=MAX_TRANSCRIPT_LINES)
transcript_lock = threading.Lock()
audio_level_history: deque[float] = deque(maxlen=AUDIO_LEVEL_WINDOW_SIZE)
audio_stats_lock = threading.Lock()
current_audio_stats = {
    "dbfs": -120.0,
    "rms": 0.0,
    "peak": 0.0,
}
last_asr_action = (
    "waiting "
    f"(peak>={MODEL_MIN_PEAK_FOR_ASR:.0f}, "
    f"dbfs>={MODEL_MIN_DBFS_FOR_ASR:.1f}, "
    f"activity>={MODEL_MIN_ACTIVITY_RATIO_FOR_ASR:.3f})"
)


def handle_shutdown_signal(signum, _frame):
    """Handle Ctrl-C / termination signals and trigger clean shutdown."""
    print(f"[*] Received signal {signum}. Shutting down...")
    shutdown_event.set()


class FlaskServerThread(threading.Thread):
    """Run Flask in a stoppable WSGI server thread."""

    def __init__(self, flask_app: Flask, host: str = "0.0.0.0", port: int = 5000):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self._server = make_server(host, port, flask_app)
        self._server.timeout = 1

    def run(self):
        print(f"[*] Receiver server listening on http://{self.host}:{self.port}")
        print(f"[*] Upload endpoint: http://{self.host}:{self.port}/upload")
        while not shutdown_event.is_set():
            self._server.handle_request()
        self._server.server_close()

    def shutdown(self):
        shutdown_event.set()


def load_whisper_to_gpu():
    """Load Whisper ASR pipeline to GPU using a local cache directory."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available. This script is configured to load Whisper on GPU."
        )

    device = "cuda:0"
    # Define the local cache path
    local_cache_path = os.path.join(os.getcwd(), "huggingface_cache")
    
    print(f"[*] Loading model '{MODEL_ID}' on {device}...")
    print(f"[*] Using local cache directory: {local_cache_path}")
    
    # Pass cache_dir to the pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_ID,
        device=device,
        torch_dtype=torch.float16,
        model_kwargs={"cache_dir": local_cache_path}
    )
    print("[+] Loading complete. Whisper is ready on GPU.")
    return pipe


def cleanup_transcribed_chunks_locked():
    """
    Unbounded rolling buffer:
    - Never delete untranscribed/in-progress chunks.
    - Every CLEANUP_EVERY_TRANSCRIBED completed transcriptions, remove all
      finished chunks from memory and disk.
    """
    global transcribed_since_cleanup

    if transcribed_since_cleanup < CLEANUP_EVERY_TRANSCRIBED:
        return

    kept_records: deque[ChunkRecord] = deque()
    removed_count = 0
    for record in file_buffer:
        if record.transcribed and not record.in_progress:
            removed_count += 1
            try:
                if os.path.exists(record.file_path):
                    os.remove(record.file_path)
            except Exception as e:
                print(f"[-] Cleanup error for {record.file_path}: {e}")
            continue
        kept_records.append(record)

    file_buffer.clear()
    file_buffer.extend(kept_records)
    transcribed_since_cleanup = 0

    if removed_count > 0:
        try:
            print(f"[*] Cleanup complete: removed {removed_count} transcribed chunks.")
        except Exception as e:
            print(f"[-] Cleanup logging error: {e}")


def get_newest_untranscribed_chunk():
    """
    Return newest untranscribed chunk and drop older pending chunks to keep
    end-to-end latency low for realtime display.
    """
    with buffer_lock:
        pending = [r for r in file_buffer if not r.transcribed and not r.in_progress]
        if not pending:
            return None, 0

        newest = pending[-1]
        stale_pending = pending[:-1]
        dropped = 0

        if stale_pending:
            stale_ids = {id(r) for r in stale_pending}
            kept_records: deque[ChunkRecord] = deque()
            for record in file_buffer:
                if id(record) in stale_ids:
                    dropped += 1
                    try:
                        if os.path.exists(record.file_path):
                            os.remove(record.file_path)
                    except Exception as e:
                        print(f"[-] Drop stale chunk cleanup error for {record.file_path}: {e}")
                    continue
                kept_records.append(record)
            file_buffer.clear()
            file_buffer.extend(kept_records)

        newest.in_progress = True
        return newest, dropped


def load_wav_for_asr(file_path: str):
    """
    Load WAV without ffmpeg and return Transformers ASR input dict:
    {"array": float32 mono samples in [-1, 1], "sampling_rate": int}
    """
    with wave.open(file_path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sampwidth == 1:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sampwidth == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if rate != TARGET_SAMPLING_RATE:
        # NumPy-only linear resample to avoid torchaudio dependency.
        if len(audio) > 1:
            old_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
            new_len = max(1, int(len(audio) * TARGET_SAMPLING_RATE / rate))
            new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
            audio = np.interp(new_x, old_x, audio).astype(np.float32)
        rate = TARGET_SAMPLING_RATE

    return {"array": audio, "sampling_rate": rate}


def append_transcript_line(transcript: str):
    text = (transcript or "").strip()
    if not text or text == ".":
        return

    normalized = " ".join(text.lower().split())
    with transcript_lock:
        if transcript_log:
            previous_normalized = " ".join(transcript_log[-1].lower().split())
            if normalized == previous_normalized:
                return
        transcript_log.append(text)


def sanitize_transcript_text(transcript: str, chunk_seconds: float) -> str:
    text = " ".join((transcript or "").split()).strip()
    if not text:
        return ""

    words = text.split(" ")
    max_allowed_words = int(chunk_seconds * MODEL_MAX_WORDS_PER_SECOND) + MODEL_WORD_BURST_ALLOWANCE
    if len(words) > max_allowed_words:
        return ""

    collapsed_words: list[str] = []
    previous_norm = ""
    run_length = 0
    for word in words:
        normalized = "".join(ch for ch in word.lower() if ch.isalnum())
        if normalized and normalized == previous_norm:
            run_length += 1
        else:
            previous_norm = normalized
            run_length = 1

        if run_length <= MODEL_MAX_CONSECUTIVE_SAME_WORD:
            collapsed_words.append(word)

    cleaned_text = " ".join(collapsed_words).strip()
    tokenized = ["".join(ch for ch in w.lower() if ch.isalnum()) for w in collapsed_words]
    tokenized = [w for w in tokenized if w]
    if len(tokenized) >= MODEL_MIN_WORDS_FOR_DIVERSITY_CHECK:
        unique_ratio = len(set(tokenized)) / len(tokenized)
        if unique_ratio < MODEL_MIN_UNIQUE_WORD_RATIO:
            return ""

    return cleaned_text


def compute_chunk_gate_stats(asr_input: dict[str, np.ndarray | int]) -> dict[str, float]:
    audio = np.asarray(asr_input["array"], dtype=np.float32)
    if audio.size == 0:
        return {
            "peak": 0.0,
            "rms": 0.0,
            "dbfs": -120.0,
            "activity_ratio": 0.0,
        }

    amplitude = np.abs(audio) * 32768.0
    peak = float(np.max(amplitude))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))) * 32768.0)
    dbfs = float(20.0 * np.log10(max(rms / 32768.0, 1e-12)))
    activity_ratio = float(np.mean(amplitude >= MODEL_ACTIVITY_ABS_AMPLITUDE))

    return {
        "peak": peak,
        "rms": rms,
        "dbfs": dbfs,
        "activity_ratio": activity_ratio,
    }


def get_asr_skip_reason(gate_stats: dict[str, float]) -> str | None:
    if gate_stats["peak"] < MODEL_MIN_PEAK_FOR_ASR:
        return f"peak<{MODEL_MIN_PEAK_FOR_ASR:.0f}"
    if gate_stats["dbfs"] < MODEL_MIN_DBFS_FOR_ASR:
        return f"dbfs<{MODEL_MIN_DBFS_FOR_ASR:.1f}"
    if gate_stats["activity_ratio"] < MODEL_MIN_ACTIVITY_RATIO_FOR_ASR:
        return f"activity<{MODEL_MIN_ACTIVITY_RATIO_FOR_ASR:.3f}"
    return None


def parse_float_or_default(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def update_audio_stats(rms: float, peak: float, dbfs: float):
    with audio_stats_lock:
        current_audio_stats["rms"] = rms
        current_audio_stats["peak"] = peak
        current_audio_stats["dbfs"] = dbfs
        audio_level_history.append(dbfs)


def get_transcript_view() -> str:
    with transcript_lock:
        if not transcript_log:
            return "Waiting for transcriptions..."
        return " ".join(transcript_log)


def get_audio_level_view() -> str:
    global last_asr_action
    with audio_stats_lock:
        dbfs = current_audio_stats["dbfs"]
        rms = current_audio_stats["rms"]
        peak = current_audio_stats["peak"]
    return (
        f"Current chunk level: {dbfs:.1f} dBFS | RMS: {rms:.0f} | Peak: {peak:.0f} "
        f"| ASR thresholds: peak>={MODEL_MIN_PEAK_FOR_ASR:.0f}, dbfs>={MODEL_MIN_DBFS_FOR_ASR:.1f}, "
        f"activity>={MODEL_MIN_ACTIVITY_RATIO_FOR_ASR:.3f} | Last ASR action: {last_asr_action}"
    )


def get_audio_plot():
    with audio_stats_lock:
        levels = list(audio_level_history)

    fig, ax = plt.subplots(figsize=(9, 2.8))
    if levels:
        x = np.arange(len(levels))
        ax.plot(x, levels, color="#1f77b4", linewidth=2)
        ax.scatter(x[-1], levels[-1], color="#d62728", s=45, zorder=3)
        ymin = min(-80.0, float(np.min(levels)) - 3.0)
        ymax = max(0.0, float(np.max(levels)) + 3.0)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0, max(len(levels) - 1, 1))
    else:
        ax.set_ylim(-80, 0)
        ax.set_xlim(0, 1)
        ax.text(0.5, -40, "Waiting for audio levels...", ha="center", va="center")

    ax.set_title("Audio Level (dBFS) - Sliding Window")
    ax.set_xlabel(f"Recent Chunks (last {AUDIO_LEVEL_WINDOW_SIZE})")
    ax.set_ylabel("dBFS")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def get_dashboard_data():
    return get_transcript_view(), get_audio_level_view(), get_audio_plot()


def get_local_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "127.0.0.1"


def build_gradio_ui() -> gr.Blocks:
    css = """
    #transcript_box textarea {
        height: 460px !important;
        overflow-y: auto !important;
        font-family: Consolas, monospace !important;
    }
    """
    with gr.Blocks(title="Live Transcript", css=css) as demo:
        gr.Markdown("## Live Transcript")
        gr.Markdown("Continuous real-time text stream with live audio level monitoring.")
        transcript_box = gr.Textbox(
            label="Transcript",
            value="Waiting for transcriptions...",
            lines=22,
            max_lines=22,
            interactive=False,
            elem_id="transcript_box",
        )
        level_box = gr.Markdown("Current chunk level: waiting for audio...")
        level_plot = gr.Plot(label="Audio Level Plot")
        # Gradio API differs by version. Prefer load(..., every=...),
        # then fall back to Timer.tick polling.
        try:
            demo.load(
                fn=get_dashboard_data,
                inputs=None,
                outputs=[transcript_box, level_box, level_plot],
                every=GRADIO_REFRESH_SECONDS,
            )
        except TypeError:
            demo.load(
                fn=get_dashboard_data,
                inputs=None,
                outputs=[transcript_box, level_box, level_plot],
            )
            if hasattr(gr, "Timer"):
                timer = gr.Timer(value=GRADIO_REFRESH_SECONDS)
                timer.tick(
                    fn=get_dashboard_data,
                    inputs=None,
                    outputs=[transcript_box, level_box, level_plot],
                )
            else:
                # Last-resort fallback for very old Gradio versions.
                gr.HTML(
                    "<script>setInterval(function(){window.location.reload();},1000);</script>"
                )
    return demo


def transcribe_worker():
    global last_asr_action
    print("[*] Transcription worker started.")
    while not shutdown_event.is_set():
        record, dropped_count = get_newest_untranscribed_chunk()
        if record is None:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        if dropped_count > 0:
            print(f"[*] Dropped {dropped_count} stale pending chunk(s) to stay realtime.")

        transcript = ""
        try:
            asr_input = load_wav_for_asr(record.file_path)
            gate_stats = compute_chunk_gate_stats(asr_input)
            skip_reason = get_asr_skip_reason(gate_stats)
            if skip_reason is not None:
                transcript = ""
                last_asr_action = f"skipped ({skip_reason})"
                continue

            # Some Transformers versions may mutate/pop keys from the input dict.
            # Compute duration before calling the pipeline.
            chunk_seconds = len(asr_input["array"]) / float(asr_input["sampling_rate"])

            result = asr_pipeline(
                asr_input,
                generate_kwargs={"task": "transcribe"},
            )
            raw_transcript = result.get("text", "").strip()
            transcript = sanitize_transcript_text(raw_transcript, chunk_seconds)
            if transcript:
                last_asr_action = "transcribed chunk"
                append_transcript_line(transcript)
            else:
                last_asr_action = "discarded likely hallucination"
        except Exception as e:
            transcript = f"[transcription_error] {e}"
            last_asr_action = "error"
            print(f"[-] Failed to transcribe {record.file_name}: {e}")
            append_transcript_line(transcript)
        finally:
            global transcribed_since_cleanup
            with buffer_lock:
                record.text = transcript
                record.transcribed = True
                record.in_progress = False
                transcribed_since_cleanup += 1
                cleanup_transcribed_chunks_locked()

    print("[*] Transcription worker stopped.")


def clear_existing_audio():
    """Removes all files in the SAVE_DIR to ensure a fresh start."""
    if os.path.exists(SAVE_DIR):
        print(f"[*] Cleaning up existing audio in {SAVE_DIR}...")
        for filename in os.listdir(SAVE_DIR):
            file_path = os.path.join(SAVE_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"[-] Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"[*] Created directory: {SAVE_DIR}")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "missing file field"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    base = secure_filename(file.filename)
    stamped_name = f"{int(time.time() * 1000)}_{base}"
    path = os.path.join(SAVE_DIR, stamped_name)
    os.makedirs(SAVE_DIR, exist_ok=True)

    file.save(path)
    rms = parse_float_or_default(request.form.get("rms"), 0.0)
    peak = parse_float_or_default(request.form.get("peak"), 0.0)
    dbfs = parse_float_or_default(request.form.get("dbfs"), -120.0)
    chunk_ts = parse_float_or_default(request.form.get("chunk_timestamp"), time.time())

    update_audio_stats(rms=rms, peak=peak, dbfs=dbfs)

    with buffer_lock:
        file_buffer.append(
            ChunkRecord(
                file_name=stamped_name,
                file_path=path,
                received_at=chunk_ts,
                rms=rms,
                peak=peak,
                dbfs=dbfs,
            )
        )

    return jsonify(
        {
            "status": "ok",
            "saved": stamped_name,
            "stats": {"rms": rms, "peak": peak, "dbfs": dbfs},
        }
    ), 200


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "running"}), 200


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    clear_existing_audio()
    asr_pipeline = load_whisper_to_gpu()
    worker = threading.Thread(target=transcribe_worker, daemon=True)
    worker.start()


    print("[*] Starting receiver server...")
    server = FlaskServerThread(app, host="0.0.0.0", port=5000)
    server.start()

    print("[*] Starting Gradio transcript UI...")
    demo = build_gradio_ui()
    demo.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        prevent_thread_lock=True,
        show_api=False,
    )
    local_url = f"http://127.0.0.1:{GRADIO_PORT}"
    lan_url = f"http://{get_local_ip()}:{GRADIO_PORT}"
    print(f"[*] Open transcript UI in browser: {local_url}")
    print(f"[*] LAN access (same network): {lan_url}")
    print("[*] Press Ctrl-C to stop.")

    try:
        while not shutdown_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("[*] Ctrl-C detected. Exiting...")
        shutdown_event.set()
    finally:
        demo.close()
        server.shutdown()
        server.join(timeout=2.0)
        shutdown_event.set()
        worker.join(timeout=2.0)
        print("[*] Receiver stopped cleanly.")
