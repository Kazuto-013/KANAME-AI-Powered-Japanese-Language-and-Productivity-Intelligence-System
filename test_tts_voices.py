import pyttsx3
e = pyttsx3.init()
voices = e.getProperty("voices")
print(f"Available voices: {len(voices)}")
for v in voices:
    langs = getattr(v, "languages", [])
    print(f"  - name={v.name}")
    print(f"    id={v.id}")
    print(f"    languages={langs}")
e.stop()
print("pyttsx3 OK")
