import os
import torchaudio
import soundfile as sf
from flask import Flask, request, jsonify
from speechbrain.pretrained import SpeakerRecognition
from scipy.spatial.distance import cosine

app = Flask(__name__)

# Load the pre-trained speaker recognition model
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# Dictionary to store voiceprints (Replace with a database in production)
voiceprints = {}

# Ensure data directory exists
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Function to extract voice embeddings
def extract_embedding(audio_path):
    signal, fs = torchaudio.load(audio_path)
    embedding = model.encode_batch(signal).squeeze().tolist()
    return embedding

#  Fix 1 & 2: Improved `/register` endpoint
@app.route("/register", methods=["POST"])
def register():
    phone_number = request.form.get("phone_number")
    audio_file = request.files.get("audio")

    if not phone_number or not audio_file:
        return jsonify({"error": "Missing phone_number or audio file"}), 400

    # Paths for saving audio
    original_path = os.path.join(data_dir, f"{phone_number}_original")
    audio_path = os.path.join(data_dir, f"{phone_number}.wav")

    #  Save the original file first
    audio_file.save(original_path)

    #  Fix 2: Convert the file to WAV if needed
    try:
        data, samplerate = sf.read(original_path)
        sf.write(audio_path, data, samplerate, format='WAV')
    except Exception as e:
        return jsonify({"error": f"Invalid audio format: {str(e)}"}), 400

    #  Fix 1: Ensure the file is valid before processing
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        return jsonify({"error": "Uploaded file is empty or corrupt"}), 400

    try:
        # Load and process the audio file
        embedding = extract_embedding(audio_path)
    except Exception as e:
        return jsonify({"error": f"Failed to process audio file: {str(e)}"}), 400

    # Store the extracted embedding
    voiceprints[phone_number] = embedding

    return jsonify({"message": "Voiceprint registered", "phone_number": phone_number})

#  `/verify` endpoint for speaker authentication
@app.route("/verify", methods=["POST"])
def verify():
    phone_number = request.form.get("phone_number")
    audio_file = request.files.get("audio")

    if not phone_number or not audio_file:
        return jsonify({"error": "Missing phone_number or audio file"}), 400

    if phone_number not in voiceprints:
        return jsonify({"error": "Phone number not registered"}), 404

    # Paths for saving test audio
    test_original_path = os.path.join(data_dir, f"{phone_number}_test_original")
    test_audio_path = os.path.join(data_dir, f"{phone_number}_test.wav")

    # Save the uploaded audio file
    audio_file.save(test_original_path)

    # Convert to WAV format
    try:
        data, samplerate = sf.read(test_original_path)
        sf.write(test_audio_path, data, samplerate, format='WAV')
    except Exception as e:
        return jsonify({"error": f"Invalid audio format: {str(e)}"}), 400

    # Load and process the test audio file
    try:
        new_embedding = extract_embedding(test_audio_path)
    except Exception as e:
        return jsonify({"error": f"Failed to process audio file: {str(e)}"}), 400

    # Compare embeddings using cosine similarity
    stored_embedding = voiceprints[phone_number]
    similarity = 1 - cosine(stored_embedding, new_embedding)

    return jsonify({"match": similarity > 0.85, "similarity": similarity})

#  Ensure Render uses the correct port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
