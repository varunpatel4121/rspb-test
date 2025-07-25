import sounddevice as sd
import numpy as np
import wave
import tempfile
import os
import time
from faster_whisper import WhisperModel
from openai import OpenAI
from config import OPENAI_API_KEY
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play
import pyttsx3
import os
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import subprocess, tempfile, os, textwrap

# === Settings ===
RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 30
FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)
MODEL_SIZE = "small"
COMPUTE_TYPE = "int8"

# This doesn't use OpenAi's model. This is just a python library.
engine = pyttsx3.init()
engine.setProperty("rate", 190)  # Adjust voice speed

PIPER_EXEC = "/Users/varunpatel/Projects/piper/build/piper"
PIPER_MODEL = "/Users/varunpatel/piper_models/en_US/amy/en_US-amy-low.onnx"
ESPEAK_PATH = "/opt/homebrew/opt/espeak-ng/share/espeak-ng-data"  # <- adjust if Intel

# === Load Models ===
print("🔁 Loading FasterWhisper...")
whisper_model = WhisperModel(MODEL_SIZE, compute_type=COMPUTE_TYPE)
client = OpenAI(api_key=OPENAI_API_KEY)

# === Record with silence detection ===
def record_until_pause(threshold=15000, max_pause_ms=800, min_record_ms=5500):
    print("🎙️ Speak now... (auto stops on silence)")
    buffer = []
    silent_chunks = 0
    max_silent_chunks = int(max_pause_ms / FRAME_DURATION_MS)
    min_record_chunks = int(min_record_ms / FRAME_DURATION_MS)
    total_chunks = 0

    try:
        with sd.InputStream(
            samplerate=RATE, channels=CHANNELS, dtype="int16", blocksize=FRAME_SIZE
        ) as stream:
            while True:
                block, _ = stream.read(FRAME_SIZE)
                buffer.append(block.copy())
                volume_norm = np.linalg.norm(block) * 10

                total_chunks += 1

                if volume_norm < threshold:
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                # Only allow silence to stop after min_record_chunks
                if (
                    total_chunks > min_record_chunks
                    and silent_chunks > max_silent_chunks
                ):
                    print("🛑 Silence detected.")
                    break
    except Exception as e:
        print(f"[ERROR] Audio input error: {e}")
        return np.array([])

    return np.concatenate(buffer)


# === Save to WAV ===
def save_audio(audio_data, rate=RATE):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(audio_data.tobytes())
        return tmp.name


# === Transcribe ===
def transcribe(path):
    print("🧠 Transcribing...")
    segs, info = whisper_model.transcribe(path)
    text = " ".join(s.text for s in segs)
    print(f"🌍 Language: {info.language}")
    return text, info.language   # ← return language code

# ─── voice_assistant_prompt.py ────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are “Nova,” a natural-sounding voice assistant. 
Speak like a friendly human in everyday conversation—no corporate jargon.

**Length**
• Default: ≤ 2 crisp sentences (≈ 25 words). 
• If user explicitly asks for detail (“explain”, “more”), allow up to 4 sentences.

**Style & Tone**
• Use contractions (“I’m”, “you’re”). 
• Prefer upbeat openers: “Sure”, “Absolutely”, “Here’s…”. 
• Read numbers as words: 2025 → “twenty twenty-five”; 3.14 → “three point one four”. 
• If the answer is a joke, end with a quick punch-line pause “… 😄”. 
• For lists, give max 3 bullet items.

**Language**
• Detect the user’s language automatically and reply in that language. 
• If unsure, default to English.

**Formatting for TTS**
• One blank line between paragraphs; avoid markdown headers. 
• No code fences unless the user explicitly asks for code.
""".strip()


# === Ask ChatGPT ===
def ask_chatgpt(user_text, user_lang, history, extra_tone=None, model="gpt-4o"):
    """
    prompt_text : the user transcript
    history     : running chat history list
    extra_tone  : optional modifier ("gentle", "excited", etc.)
    """
    # Detect language if you captured it from Whisper
    # Example assumes 'detected_lang' is available in outer scope
    lang_line = (
        f"\n(The user’s language code is {user_lang}.)" if user_lang else ""
    )

    # Optional tone tweak
    tone_line = ""
    if extra_tone == "gentle":
        tone_line = "\nSpeak extra gently and encouraging."
    elif extra_tone == "excited":
        tone_line = "\nUse lively wording—exclamation welcome!"

    system_msg = SYSTEM_PROMPT + lang_line + tone_line

    history.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(
        model=model,  # gpt-4o or gpt-4o-mini
        messages=[{"role": "system", "content": system_msg}] + history,
    )

    reply_text = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply_text})
    return reply_text, history


def speak_text_py(text):
    engine.say(text)
    engine.runAndWait()


def speak_text_os(text):
    os.system(f'say -v Ava "{text}"')  # Ava (Enhanced) will now be used if installed


def speak_text_google(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tts.save(tmp.name)
        audio = AudioSegment.from_file(tmp.name)
        play(audio)
        os.remove(tmp.name)


def speak_text_piper(text: str):
    """Generate and play speech with Piper (blocking)."""
    # Piper expects stdin → WAV out
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        subprocess.run(
            [PIPER_EXEC, "--model", PIPER_MODEL, "--output_file", tmp.name],
            input=text.encode(),
            check=True,
        )
        subprocess.run(["afplay", tmp.name])  # macOS playback
        os.unlink(tmp.name)


def speak_audio_openai(text: str):
    """
    Generate speech with OpenAI TTS and play it.
    Block until playback is done.
    """
    try:
        resp = client.audio.speech.create(
            model="gpt-4o-mini-tts", voice="shimmer", input=text, speed=1.5
        )
        # resp.content is raw MP3 bytes
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(resp.content)
            tmp.flush()
            audio = AudioSegment.from_file(tmp.name, format="mp3")
            play(audio)  # block until finished
            os.remove(tmp.name)
    except Exception as e:
        print(f"[TTS-1 error] {e}")


# === Main Loop ===
def run_assistant():
    print("\n🤖 Voice Assistant Ready. Press Ctrl+C to exit.\n")
    history = []

    while True:
        try:
            audio_data = record_until_pause()
            wav_path = save_audio(audio_data)
            text, lang = transcribe(wav_path)
            os.remove(wav_path)

            if text.strip():
                print(f"\n📝 You said: {text}")
                reply, history = ask_chatgpt(
                    text,  # user text from Whisper
                    lang,
                    history,
                    extra_tone=None,  # or "gentle", "excited"
                )
                print(f"\n🤖 ChatGPT: {reply}\n")
                speak_audio_openai(reply)  # 🔊 TTS added here
            else:
                print("❌ No clear speech detected.")

            time.sleep(0.3)
        except KeyboardInterrupt:
            print("\n👋 Exiting. Goodbye!")
            break


if __name__ == "__main__":
    run_assistant()
