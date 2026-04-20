#!/usr/bin/env python3
"""
Raspberry Pi Node 1: Continuous audio capture -> denoise -> HTTP upload.

Architecture:
- Producer thread: records 3-second chunks from USB mic and pushes raw bytes into a queue.
- Consumer thread: denoises chunk with wavelet transform, writes WAV in /dev/shm, uploads via HTTP POST.

Designed for:
card 1, device 0 (USB PnP Sound Device) as reported by `arecord -l`.
"""

from __future__ import annotations

import os
import queue
import signal
import threading
import time
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyaudio
import pywt
import requests


# -----------------------------
# Configuration
# -----------------------------
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3
FRAMES_PER_CHUNK = int(RATE / CHUNK * RECORD_SECONDS)

# USB mic reported as card 1, device 0.
# In PyAudio, this is usually a device index discovered by name.
INPUT_DEVICE_KEYWORD = "default"

# Queue size controls buffering depth between recording and processing/upload.
QUEUE_MAXSIZE = 8

# Store temporary WAV files in RAM disk to avoid SD-card wear.
TMP_DIR = "/dev/shm"

# Receiver endpoint running on your computer.
UPLOAD_URL = "http://172.20.10.3:5000/upload"
HTTP_TIMEOUT_SECONDS = 10

# Wavelet denoising settings
WAVELET = "db8"
WAVELET_LEVEL = 2
THRESHOLD = 100

# Stop conditions
ENABLE_TYPED_STOP = True



@dataclass
class AudioChunk:
    timestamp: float
    raw_bytes: bytes


shutdown_event = threading.Event()
audio_queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=QUEUE_MAXSIZE)


def on_signal(signum, _frame):
    print(f"[main] Signal {signum} received, shutting down...")
    shutdown_event.set()


def stop_on_user_input():
    print("[main] Type anything then press Enter to stop early.")
    try:
        _ = input()
    except EOFError:
        return
    if not shutdown_event.is_set():
        print("[main] User requested stop.")
        shutdown_event.set()


def find_input_device_index(pa: pyaudio.PyAudio, keyword: str) -> Optional[int]:
    keyword = keyword.lower()
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        name = str(info.get("name", "")).lower()
        max_input = int(info.get("maxInputChannels", 0))
        if max_input > 0 and keyword in name:
            return i
    return None


def denoise_audio(raw_data: bytes) -> np.ndarray:
    audio_array = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)

    coeffs = pywt.wavedec(audio_array, WAVELET, level=WAVELET_LEVEL)
    coeffs[1:] = [pywt.threshold(c, value=THRESHOLD, mode="soft") for c in coeffs[1:]]
    reconstructed = pywt.waverec(coeffs, WAVELET)

    reconstructed = np.clip(reconstructed, -32768, 32767).astype(np.int16)
    return reconstructed


def write_wav(path: str, samples: np.ndarray):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(samples.tobytes())


def compute_audio_stats(samples: np.ndarray) -> dict[str, float]:
    arr = samples.astype(np.float64)
    if arr.size == 0:
        return {"rms": 0.0, "peak": 0.0, "dbfs": -120.0}

    rms = float(np.sqrt(np.mean(arr**2)))
    peak = float(np.max(np.abs(arr)))
    dbfs = 20.0 * np.log10(max(rms / 32768.0, 1e-12))
    return {"rms": rms, "peak": peak, "dbfs": dbfs}


def send_to_pc(file_path: str, stats: dict[str, float], chunk_ts: float) -> int:
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "audio/wav")}
        data = {
            "chunk_timestamp": f"{chunk_ts:.6f}",
            "rms": f"{stats['rms']:.6f}",
            "peak": f"{stats['peak']:.6f}",
            "dbfs": f"{stats['dbfs']:.6f}",
        }
        response = requests.post(
            UPLOAD_URL,
            files=files,
            data=data,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
    return response.status_code


def producer_thread(pa: pyaudio.PyAudio):
    # Recommended: Set INPUT_DEVICE_KEYWORD = "default" in your config
    stream = None
    try:
        # 1. Try to find the device index based on your keyword
        input_device_index = find_input_device_index(pa, INPUT_DEVICE_KEYWORD)
        
        # 2. Fallback: If keyword search fails but we know Card 1 exists, 
        # we can try to force index 8 (the 'default' from your test.py)
        if input_device_index is None:
            print(f"[producer] Keyword '{INPUT_DEVICE_KEYWORD}' not found. Falling back to default.")
            # In your specific system, index 8 was 'default'
            input_device_index = 8 

        print(f"[producer] Opening stream on device index: {input_device_index}")
        
        stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=CHUNK,
        )
        print("[producer] Recording started.")

        try:
            while not shutdown_event.is_set():
                frames = []
                # Calculate how many reads we need for 3 seconds
                for _ in range(FRAMES_PER_CHUNK):
                    if shutdown_event.is_set():
                        break
                    try:
                        # overflow=False prevents the script from crashing if the CPU lags
                        frame = stream.read(CHUNK, exception_on_overflow=False)
                        frames.append(frame)
                    except Exception as e:
                        print(f"[producer] Read error: {e}")
                        break

                if not frames or shutdown_event.is_set():
                    continue

                raw_bytes = b"".join(frames)
                audio_array = np.frombuffer(raw_bytes, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_array.astype(np.float64)**2))
                peak = np.max(np.abs(audio_array))
                print(f"[producer] Audio level - RMS: {rms:.2f}, Peak: {peak}")

                chunk = AudioChunk(timestamp=time.time(), raw_bytes=raw_bytes)

                try:
                    audio_queue.put(chunk, timeout=1)
                    print(f"[producer] queued chunk ts={chunk.timestamp:.3f} q={audio_queue.qsize()}")
                except queue.Full:
                    print("[producer] queue full, dropping oldest chunk.")
                    try:
                        audio_queue.get_nowait()
                        audio_queue.task_done()
                    except queue.Empty:
                        pass
                    audio_queue.put(chunk, timeout=1)
        finally:
            if stream is not None:
                try:
                    if stream.is_active():
                        stream.stop_stream()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
    except Exception as e:
        print(f"[producer] Fatal error: {e}")
        shutdown_event.set()
    finally:
        print("[producer] stopped.")


def consumer_thread():
    os.makedirs(TMP_DIR, exist_ok=True)
    print("[consumer] Started.")

    while not shutdown_event.is_set() or not audio_queue.empty():
        try:
            chunk = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        file_name = f"audio_{chunk.timestamp:.6f}.wav"
        file_path = os.path.join(TMP_DIR, file_name)

        try:
            denoised = denoise_audio(chunk.raw_bytes)
            write_wav(file_path, denoised)
            stats = compute_audio_stats(denoised)
            status_code = send_to_pc(file_path, stats, chunk.timestamp)
            print(f"[consumer] uploaded {file_name} -> HTTP {status_code}")
        except requests.RequestException as e:
            print(f"[consumer] upload failed for {file_name}: {e}")
        except Exception as e:
            print(f"[consumer] processing failed for {file_name}: {e}")
        finally:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            audio_queue.task_done()

    print("[consumer] stopped.")


def main():
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    pa = pyaudio.PyAudio()

    producer = threading.Thread(target=producer_thread, args=(pa,), daemon=False)
    consumer = threading.Thread(target=consumer_thread, daemon=False)
    producer.start()
    consumer.start()

    if ENABLE_TYPED_STOP:
        input_monitor = threading.Thread(target=stop_on_user_input, daemon=True)
        input_monitor.start()

    try:
        while not shutdown_event.is_set():
            time.sleep(0.5)
    finally:
        shutdown_event.set()
        producer.join()
        consumer.join()
        time.sleep(0.1)
        try:
            pa.terminate()
        except Exception:
            pass
        print("[main] exit complete.")


if __name__ == "__main__":
    main()
