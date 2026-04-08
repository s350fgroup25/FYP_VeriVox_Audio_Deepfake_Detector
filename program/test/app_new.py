from flask import Flask, request, jsonify, render_template
import os
import json
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/home/carmen/asvspoof/S_audio'
ALLOWED_EXTENSIONS = {'flac', 'wav', 'mp3'}

app = Flask(
    __name__,
    template_folder='/home/carmen/asvspoof/www/templates',
    static_folder='/home/carmen/asvspoof/www/static',
    static_url_path='/static'
)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

# Global model variables
model = None
feature_extractor = None
device = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_once():
    global model, feature_extractor, device
    try:
        import torch
        from safetensors.torch import load_file
        from transformers import Wav2Vec2FeatureExtractor
        from model_sentence1 import Model

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading model on {device}...")

        model = Model(model_type='wavlm', device=device).to(device)
        state_dict = load_file('/home/carmen/asvspoof/program/wavlm-epoch50.safetensors')
        model.load_state_dict(state_dict)
        model.eval()

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base')
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

# Load model ONCE at startup
model_loaded = load_model_once()

@app.route('/')
def index():
    if not model_loaded:
        return "❌ Model failed to load. Check server logs.", 500
    return render_template('indexT.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'})

    # Clear old files
    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    return jsonify({'success': True, 'filename': filename})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    filename = data.get('filename')
    if not filename:
        return jsonify({'success': False, 'error': 'Filename is required'})
    
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'})

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'File not found'})

    try:
        import torch
        import soundfile as sf
        import librosa  # For resampling

        # Read audio
        waveform, sr = sf.read(filepath)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # ✅ RESAMPLE to 16kHz (WavLM requirement)
        if sr != 16000:
            print(f"Resampling from {sr}Hz to 16000Hz")
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            sr = 16000

        inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            logits = model(input_values=input_values)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        result = {
            "prob_real": float(probs[0, 1]),
            "prob_fake": float(probs[0, 0]),
            "label": bool(probs[0, 1] >= 0.5)  # ✅ Native Python bool (fixes JSON error)
        }

        return jsonify({'success': True, 'result': json.dumps(result)})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)

