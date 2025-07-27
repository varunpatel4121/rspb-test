import time
import sounddevice as sd
import numpy as np
import webrtcvad

# Use PipeWire/Pulse backend so default source (your USB mic) is used.
# From your listing: device "pulse" exists (index 5); we can select by name.
IN_DEV = 'pulse'            # or 5
RATE   = 16000              # VAD wants 8k/16k/32k/48k; 16k is a good default
CHANNELS = 1

# For VAD we will analyze 20 ms chunks (frame length must be 10/20/30 ms)
VAD = webrtcvad.Vad(2)      # 0=least aggressive, 3=most aggressive
FRAME_MS   = 20
FRAME_SAMP = RATE * FRAME_MS // 1000

def record_until_pause_vad(max_pause_ms=800, min_record_ms=600, max_utter_ms=15000, debug=True):
    """
    Capture mic until we detect >= max_pause_ms of non-speech *after* at least min_record_ms of speech.
    Also enforce a hard cap of max_utter_ms to avoid hangs.
    Returns an int16 numpy array at 16 kHz.
    """
    print("🎙️ Speak now... (auto stops on silence)")
    sd.check_input_settings(device=IN_DEV, samplerate=RATE, channels=CHANNELS, dtype='int16')

    voiced_ms = 0
    silence_ms = 0
    started = time.time()
    last_dbg = started
    chunks = []  # store raw int16 frames for playback/transcription

    # Let PortAudio pick an optimal internal blocksize (avoid stalls)
    with sd.InputStream(device=IN_DEV, samplerate=RATE, channels=CHANNELS,
                        dtype='int16', blocksize=0, latency='low') as stream:
        while True:
            # Read exactly FRAME_SAMP samples (20 ms)
            block, _ = stream.read(FRAME_SAMP, exception_on_overflow=False)

            # If we ever see no data for too long, bail out
            if block.size == 0:
                if time.time() - started > 10:
                    print("[WARN] No audio data received for 10s — check input device")
                    return np.array([], dtype=np.int16)
                continue

            chunks.append(block.copy())

            # VAD needs bytes of 16‑bit mono PCM at the exact RATE we set above
            is_speech = VAD.is_speech(block.tobytes(), RATE)
            if is_speech:
                voiced_ms += FRAME_MS
                silence_ms = 0
            else:
                silence_ms += FRAME_MS

            # Debug 1×/s
            if debug and time.time() - last_dbg > 1.0:
                print(f"   speech={voiced_ms:4d} ms  silence={silence_ms:4d} ms")
                last_dbg = time.time()

            # Stop conditions
            elapsed = (time.time() - started) * 1000
            if voiced_ms >= min_record_ms and silence_ms >= max_pause_ms:
                print("🛑 Silence detected.")
                break
            if elapsed >= max_utter_ms:
                print("⏱️ Max utterance time reached; stopping.")
                break

    return np.concatenate(chunks)
