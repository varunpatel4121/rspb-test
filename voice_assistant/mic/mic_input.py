# mic_input.py (with OpenAI Whisper)
import openai
import sounddevice as sd
import numpy as np
import webrtcvad
import wave
import tempfile
import os
from config import OPENAI_API_KEY

client = openai.OpenAI(api_key=OPENAI_API_KEY)

RATE = 16000  # 🎧 Sample rate in Hz (16,000 samples per second). Standard for speech recognition models like Whisper.
CHANNELS = 1  # 🎙️ Mono audio (1 channel), which is typical and sufficient for voice input.
FRAME_DURATION_MS = 30  # 🧱 Each audio frame is 30 milliseconds long. Common setting for voice activity detection (VAD).
FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)  
# 📏 Number of audio samples per frame.
# = 16,000 samples/sec * 0.03 sec = 480 samples per frame
MAX_SILENCE_DURATION_SEC = 2  # ⏸️ If silence is detected for this long (1.2 seconds), stop recording.
MAX_SILENT_FRAMES = int(MAX_SILENCE_DURATION_SEC * 1000 / FRAME_DURATION_MS)  
# 🔢 How many frames of silence equals 1.2 seconds.
# = 1200 ms / 30 ms = 40 silent frames before stopping

INITIAL_SILENCE_DURATION_SEC = 4  # Allow up to 4 seconds of silence at the start
INITIAL_SILENT_FRAMES = int(INITIAL_SILENCE_DURATION_SEC * 1000 / FRAME_DURATION_MS)

vad = webrtcvad.Vad(1)  # 0 = sensitive, 3 = aggressive


def save_audio_to_wav(audio, rate):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        with wave.open(tmp.name, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(rate)
            wf.writeframes(audio.tobytes())
        return tmp.name

def record_until_pause():
    print("🎤 Speak now (press Ctrl+C to stop).")
    frames = []
    silence_count = 0
    speech_started = False

    with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype='int16', blocksize=FRAME_SIZE) as stream:
        while True:
            audio_block, _ = stream.read(FRAME_SIZE)
            frames.append(audio_block.copy())

            is_speech = vad.is_speech(audio_block.tobytes(), RATE)

            if not is_speech:
                silence_count += 1
            else:
                silence_count = 0
                speech_started = True

            if not speech_started:
                if silence_count > INITIAL_SILENT_FRAMES:
                    print("🛑 Silence detected (no speech). Stopping.")
                    break
            else:
                if silence_count > MAX_SILENT_FRAMES:
                    print("🛑 Silence detected. Stopping.")
                    break

    return np.concatenate(frames)

def transcribe_with_whisper(audio_data):
    wav_path = save_audio_to_wav(audio_data, RATE)
    with open(wav_path, "rb") as f:
        print("🧠 Transcribing...")
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
            language="en"  # Force English to avoid random language outputs
        )
    os.remove(wav_path)
    # Filter out empty, '.', or whitespace-only transcriptions
    if not result or result.strip() == '' or result.strip() == '.':
        return ''
    return result

def get_audio_input():
    audio_data = record_until_pause()
    return transcribe_with_whisper(audio_data)