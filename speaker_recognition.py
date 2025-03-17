from flask import Flask, request, jsonify
import torchaudio
from speechbrain.pretrained import SpeakerRecognition
from scipy.spatial.distance import cosine

app = Flask(__name__)
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# Extract speaker embedding
def extract_embedding(audio_path):
    signal, fs = torchaudio.load(audio_path)
    embedding = model.encode_batch(signal).squeeze().tolist()
    return embedding

# Store voiceprints (mock database)
voiceprints = {}

@app.route("/register", methods=["POST"])
def register():
    phone_number = request.form["phone_number"]
    audio_file = request.files["audio"]
    
    audio_path = f"data/{phone_number}.wav"
    audio_file.save(audio_path)

    embedding = extract_embedding(audio_path)
    voiceprints[phone_number] = embedding

    return jsonify({"message": "Voiceprint registered", "phone_number": phone_number})

@app.route("/verify", methods=["POST"])
def verify():
    phone_number = request.form["phone_number"]
    audio_file = request.files["audio"]

    if phone_number not in voiceprints:
        return jsonify({"match": False, "message": "Phone number not registered"})

    audio_path = f"temp/{phone_number}_test.wav"
    audio_file.save(audio_path)
    new_embedding = extract_embedding(audio_path)

    stored_embedding = voiceprints[phone_number]
    similarity = 1 - cosine(stored_embedding, new_embedding)

    return jsonify({"match": similarity > 0.85, "similarity": similarity})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
