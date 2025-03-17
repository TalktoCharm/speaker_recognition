import os
import torchaudio
import soundfile as sf
from flask import Flask, request, jsonify
from speechbrain.pretrained import SpeakerRecognition
from scipy.spatial.distance import cosine

app = Flask(__name__)

# ✅ Lazy load model only when needed
model = None

def get_model():
    global model
    if model is None:
        print("🔄 Loading Speaker Recognition Model...")
        model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"dtype": "float16", "device": "cpu"}  # ✅ Uses less memory
        )
    return model

voiceprints = {}
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

def extract_embedding(audio_path):
    try:
        metadata = torchaudio.info(audio_path)
        if metadata.num_frames / metadata.sample_rate > 10:  # ✅ Limit to 10 sec
            return None, "Audio too long (max 10 sec)"

        signal, fs = torchaudio.load(audio_path)
        model = get_model()  # ✅ Load model only if needed
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

    original_path = os.path.join(data_dir, f"{phone_number}_original.wav")
    audio_file.save(original_path)

    try:
        data, samplerate = sf.read(original_path)
        audio_path = os.path.join(data_dir, f"{phone_number}.wav")
        sf.write(audio_path, data, samplerate, format="WAV")
    except Exception as e:
        return jsonify({"error": f"Invalid audio format: {str(e)}"}), 400

    embedding, error = extract_embedding(audio_path)
    if error:
        return jsonify({"error": error}), 400

    voiceprints[phone_number] = embedding
    return jsonify({"message": "Voiceprint registered", "phone_number": phone_number})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
