## Edge-to-Whisper: Real-Time Denoised Audio Transcription

### Team Member Names
- `Mo Jiang`
- `Junsoo Kim`

### Instructions to Compile/Execute Program(s)
1. Install dependencies:
   `pip install -r requirements.txt`
2. On Raspberry Pi, run:
   `python3 rpi_audio_node.py`
   - Configure the target IP address to your PC's IP in the script/settings.
3. On PC, run:
   `python pc_receiver.py`
4. Open the generated Gradio link in a web browser to view real-time transcription.

### External Libraries Used
- `pyaudio`
- `PyWavelets`
- `requests`
- `numpy`
- `flask`
- `torch`
- `transformers`
- `openai-whisper`
- `torchaudio`
- `gradio`
- `matplotlib`

### Core Idea
This project is a two-node IoT system designed for high-fidelity speech reconstruction. It uses a Raspberry Pi as an edge node to capture audio and apply a Discrete Wavelet Transform to remove background noise, satisfying the rubric's requirement for non-trivial signal processing. The cleaned audio is then sent via HTTP POST to a PC, where a CUDA-accelerated Whisper model (`openai/whisper-large-v3-turbo`) performs real-time transcription. The results are displayed on a Gradio web dashboard to provide a clear, live visualization of the data.

### Method
1. `rpi_audio_node.py` (Edge Data Acquisition & Signal Conditioning)
    This script runs on the Raspberry Pi to handle high-frequency data collection from the USB microphone. It performs temporal-spectral denoising using a Discrete Wavelet Transform to isolate linguistic signals from environmental noise before transmission, fulfilling the requirement for non-trivial edge processing. Cleaned audio chunks are then dispatched to the server node via HTTP POST requests.
2. `pc_receiver.py` (Neural Inference & Visualization)
    This script runs on the PC to act as the central hub for heavy computational offloading. It hosts a Flask-based REST API that asynchronously buffers incoming data and feeds it into a CUDA-accelerated Transformer model (`whisper-large-v3-turbo`) for real-time speech-to-text inference. Finally, it renders a reactive Gradio dashboard to provide a polished web interface for data visualization.

### AI Usage Statement
- We came up with the project idea and wrote most of the core code ourselves (PC-side model loading, transformer input feeding and display, Gradio UI, silence thresholding, and filtering).
- AI helped optimize implementation details such as thread creation, multithreading, graceful shutdown handling, and server-side debug printing.
- AI also suggested integration ideas for Gradio and web interfacing.
