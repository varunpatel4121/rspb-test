import speech_recognition as sr

def transcribe_google(audio):
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio)
        print(f"📝 Transcribed: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ Google could not understand audio.")
    except sr.RequestError as e:
        print(f"❌ Could not request results from Google: {e}")
    return ""