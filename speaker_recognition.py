import os
import torchaudio
import soundfile as sf
from flask import Flask, request, jsonify
from speechbrain.pretrained import SpeakerRecognition
from scipy.spatial.distance import cosine

app = Flask(__name__)

model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"dtype": "float16"})

voiceprints = {}
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

def extract_embedding(audio_path):
    try:
        # ✅ Ensure file is small and valid
        metadata = torchaudio.info(audio_path)
        if metadata.num_frames / metadata.sample_rate > 10:  # ✅ Limit to 10 sec
            return None, "Audio too long (max 10 sec)"

        signal, fs = torchaudio.load(audio_path)
        embedding = model.encode_batch(signal).squeeze().tolist()
        return embedding, None
    except Exception as e:
        return None, str(e)

@app.route("/register", methods=["POST"])
def register():
    phone_number = request.form.get("phone_number")
    audio_file = request.files.get("audio")

    if not phone_number or not audio_file:
        return jsonify({"error": "Missing phone_number or audio file"}), 400

    # ✅ Log incoming file size
    audio_file.seek(0, os.SEEK_END)
    file_size = audio_file.tell()
    audio_file.seek(0)
    print(f"📂 Received file: {file_size} bytes for {phone_number}")

    if file_size > 5_000_000:  # ✅ Limit to 5MB
        return jsonify({"error": "File too large"}), 400

    original_path = os.path.join(data_dir, f"{phone_number}_original.wav")
    audio_file.save(original_path)

    # ✅ Convert to WAV if necessary
    try:
        data, samplerate = sf.read(original_path)
        audio_path = os.path.join(data_dir, f"{phone_number}.wav")
        sf.write(audio_path, data, samplerate, format="WAV")
    except Exception as e:
        return jsonify({"error": f"Invalid audio format: {str(e)}"}), 400

    # ✅ Extract voice embedding
    embedding, error = extract_embedding(audio_path)
    if error:
        return jsonify({"error": error}), 400

    voiceprints[phone_number] = embedding
    return jsonify({"message": "Voiceprint registered", "phone_number": phone_number})

# ✅ Increase Gunicorn timeout for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

