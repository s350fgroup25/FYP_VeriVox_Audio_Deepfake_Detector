from gtts import gTTS
import tempfile
from flask import Flask, request, jsonify, render_template
import os
import json
from werkzeug.utils import secure_filename
import subprocess
import time
from flask import send_from_directory
import ffmpeg
import torch
from safetensors.torch import load_file
from transformers import Wav2Vec2FeatureExtractor
from model import HFReadyModel  # 🔄 use new model class

UPLOAD_FOLDER = '/home/carmen/asvspoof/S_audio'
ALLOWED_EXTENSIONS = {'flac', 'wav', 'mp3', 'webm', 'm4a'}

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

# ---------- Audio cleanup ----------

# ✅ 強化版自動清理：強制保持只有最新1個音檔
def cleanup_s_audio():
    """強制保持 S_audio 只有最新 1 個音檔 + 詳細日誌"""
    try:
        print("🚀 === STARTING CLEANUP ===")
        audio_ext = {'.wav', '.flac', '.mp3', '.webm', '.m4a'}
        all_files = os.listdir(UPLOAD_FOLDER)
        audio_files = []

        # 掃描所有音檔
        for f in all_files:
            if len(f.rsplit('.', 1)) > 1 and f.rsplit('.', 1)[1].lower() in audio_ext:
                full_path = os.path.join(UPLOAD_FOLDER, f)
                if os.path.isfile(full_path):
                    ctime = os.path.getctime(full_path)
                    audio_files.append((f, ctime))
                    print(f"  📁 Found: {f} (created: {time.ctime(ctime)})")

        print(f"🔍 Total audio files found: {len(audio_files)}")

        if len(audio_files) > 1:
            # 按建立時間排序（舊→新），保留最新
            audio_files.sort(key=lambda x: x[1])
            print(f"📋 Oldest to newest: {[f[0] for f in audio_files]}")

            # 刪除除了最新的所有檔案
            for i, (old_file, _) in enumerate(audio_files[:-1]):
                old_path = os.path.join(UPLOAD_FOLDER, old_file)
                try:
                    os.remove(old_path)
                    print(f"🧹 DELETED #{i+1}: {old_file}")
                except Exception as e:
                    print(f"❌ Failed to delete {old_file}: {e}")
        else:
            print("✅ Already only 1 file or empty")

        # 驗證結果
        remaining = [f for f in os.listdir(UPLOAD_FOLDER)
                     if len(f.rsplit('.', 1)) > 1 and f.rsplit('.', 1)[1].lower() in audio_ext]
        print(f"✅ CLEANUP DONE: {len(remaining)} audio files remaining: {remaining}")
        print("🚀 === CLEANUP COMPLETE ===\n")

    except Exception as e:
        print(f"❌ CLEANUP ERROR: {e}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model_once():
    """Load the new wav2vec2-xls-r-300m based model only once."""
    global model, feature_extractor, device
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading model on {device}...")

        # 1) init backbone + classifier
        model = HFReadyModel(device=device).to(device)

        # 2) load new safetensors weights
        state_dict = load_file('/home/carmen/asvspoof/program/model.safetensors')
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        # 3) feature extractor for wav2vec2-xls-r-300m
        # If offline, you can point this to a local folder instead
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            '/home/carmen/asvspoof/program/wav2vec2-xls-r-300m-feature'
        )

        print("✅ New wav2vec2-xls-r-300m model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


# Load model ONCE at startup
model_loaded = load_model_once()

# ---------- Video → Audio config ----------

VIDEO_EXT = {'mp4', 'm4v', 'mov', 'webm', 'mkv', 'm4p', 'm4b'}

def allowed_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in VIDEO_EXT

# ---------- Routes ----------

@app.route('/')
def index():
    if not model_loaded:
        return "❌ Model failed to load. Check server logs.", 500
    return render_template('home.html')


@app.route('/upload')
def upload_page():
    if not model_loaded:
        return "❌ Model failed to load. Check server logs.", 500
    return render_template('index.html')


@app.route('/record')
def record_page():
    if not model_loaded:
        return "❌ Model failed to load. Check server logs.", 500
    return render_template('record.html')


@app.route('/convert')
def convert_page():
    if not model_loaded:
        return "❌ Model failed to load. Check server logs.", 500
    return render_template('convert.html')


@app.route('/convert', methods=['POST'])
def convert_audio():
    print("\n🎵 === CONVERT ROUTE ===")
    cleanup_s_audio()  # ✅ 強制清理

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})

    original_name = secure_filename(file.filename).rsplit('.', 1)[0]
    timestamp = int(time.time())
    converted_path = os.path.join(UPLOAD_FOLDER, f"convert_{original_name}_{timestamp}.wav")
    input_path = os.path.join(UPLOAD_FOLDER, f"temp_{timestamp}_{secure_filename(file.filename)}")

    try:
        file.save(input_path)
        print(f"🔄 Converting {file.filename} → {os.path.basename(converted_path)}")

        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(stream, converted_path, ar=16000, ac=1, acodec='pcm_s16le')
        ffmpeg.run(stream, overwrite_output=True, quiet=True)

        os.unlink(input_path)
        print(f"✅ Converted: {os.path.getsize(converted_path)} bytes")
        print("🎵 === CONVERT COMPLETE ===\n")

        return jsonify({
            'success': True,
            'filename': os.path.basename(converted_path),
            'size': os.path.getsize(converted_path),
            'original': file.filename
        })
    except Exception as e:
        print(f"❌ Convert error: {e}")
        if os.path.exists(input_path):
            os.unlink(input_path)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/download/<filename>')
def download_file(filename):
    print(f"📥 Serving: {filename}")
    try:
        return send_from_directory(UPLOAD_FOLDER, filename, mimetype='audio/wav')
    except FileNotFoundError:
        return "File not found", 404


@app.route('/fake')
def fake_page():
    if not model_loaded:
        return "❌ Model failed to load. Check server logs.", 500
    return render_template('fake.html')


@app.route('/tts', methods=['POST'])
def tts_generate():
    print("\n🤖 === TTS ROUTE ===")
    cleanup_s_audio()  # ✅ 強制清理

    data = request.get_json()
    text = data.get('text', '').strip()

    if not text or len(text) > 1000:
        return jsonify({'success': False, 'error': 'Text empty or too long (max 1000 chars)'})

    timestamp = int(time.time())
    filename = f"fake_{timestamp}.wav"
    output_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        tts = gTTS(text=text, lang='en', slow=False)
        mp3_path = os.path.join(UPLOAD_FOLDER, f"temp_{timestamp}.mp3")
        tts.save(mp3_path)

        stream = ffmpeg.input(mp3_path)
        stream = ffmpeg.output(stream, output_path, ar=16000, ac=1, acodec='pcm_s16le')
        ffmpeg.run(stream, overwrite_output=True, quiet=True)

        os.unlink(mp3_path)

        print(f"✅ TTS generated: {filename} ({os.path.getsize(output_path)} bytes)")
        print("🤖 === TTS COMPLETE ===\n")
        return jsonify({
            'success': True,
            'filename': filename,
            'size': os.path.getsize(output_path)
        })

    except Exception as e:
        print(f"❌ TTS error: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/upload', methods=['POST'])
def upload_file():
    print("\n📤 === UPLOAD ROUTE ===")
    cleanup_s_audio()  # ✅ 強制清理

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    print(f"✅ Uploaded: {filename} ({os.path.getsize(filepath)} bytes)")
    print("📤 === UPLOAD COMPLETE ===\n")
    return jsonify({'success': True, 'filename': filename})

@app.route('/analyze', methods=['POST'])
def analyze():
    """音頻分析 + 每次完成立即內存清理（支持並發）"""
    start_time = time.time()
    
    data = request.json
    filename = data.get('filename')
    
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'})

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'File not found'})

    wav_path = filepath.rsplit('.', 1)[0] + '.wav'

    try:
        import soundfile as sf
        import librosa
        import subprocess

        # 🔥 處理webm轉wav
        if filename.lower().endswith('.webm'):
            print(f"🔄 Converting {filename} to WAV...")
            result = subprocess.run([
                'ffmpeg', '-y', '-i', filepath, '-ar', '16000', '-ac', '1', wav_path
            ], check=True, capture_output=True, text=True)
            print(f"✅ FFmpeg: {result.stderr}")
            filepath = wav_path

        # 🔥 讀取音頻
        waveform, sr = sf.read(filepath)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # 🔥 重採樣到16kHz
        if sr != 16000:
            print(f"🔄 Resampling {sr}Hz → 16kHz")
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            sr = 16000

        # 🔥 特徵提取
        inputs = feature_extractor(
            waveform, sampling_rate=sr, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(device)

        # 🔥 模型推理
        with torch.no_grad():
            logits = model(input_values=input_values)

        # 🔥 計算概率
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # 🔥 準備前端數據（CSV導出用）
        result_data = {
            "prob_real": float(probs[0, 1]),
            "prob_fake": float(probs[0, 0]),
            "label": bool(probs[0, 1] >= 0.5)
        }

        # 🔥 🔥 🔥 關鍵：每次analyze完立即清理內存 🔥 🔥 🔥
        print(f"🧹 清理內存... (Real:{result_data['prob_real']:.3f}, Time:{time.time()-start_time:.1f}s)")
        
        # 1. 清理大內存對象
        cleanup_vars = ['waveform', 'input_values', 'logits', 'probs', 'inputs']
        for var_name in cleanup_vars:
            if var_name in locals():
                try:
                    exec(f"del {var_name}")
                except:
                    pass
        
        # 2. PyTorch GPU內存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 3. Python垃圾回收
        import gc
        gc.collect()
        
        # 4. 清理臨時wav文件
        if os.path.exists(wav_path) and wav_path != filepath:
            os.remove(wav_path)

        total_time = time.time() - start_time
        print(f"✅ 清理完成！總時間: {total_time:.1f}s")
        
        # ✅ 返回完整數據（前端CSV導出正常）
        return jsonify({
            'success': True, 
            'result': json.dumps(result_data),
            'processing_time': total_time
        })

    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg錯誤: {e.stderr}")
        import gc; gc.collect()
        return jsonify({'success': False, 'error': f'FFmpeg failed: {e.stderr}'})
    except Exception as e:
        print(f"❌ 分析錯誤: {str(e)}")
        import gc; gc.collect()
        return jsonify({'success': False, 'error': str(e)})


# ---------- Video → Audio pages ----------

@app.route('/video2audio')
def video2audio_page():
    if not model_loaded:
        return "❌ Model failed to load. Check server logs.", 500
    return render_template('video_to_audio.html')

# 🔥 ➕ 新增這一個：Realtime Monitor 頁面
@app.route('/realtime')
def realtime_page():
    if not model_loaded:
        return "❌ Model failed to load. Check server logs.", 500
    return render_template('realtime.html')

@app.route('/realtime_continuous')
def realtime_continuous_page():
    if not model_loaded:
        return "❌ Model failed to load. Check server logs.", 500
    return render_template('realtime_continuous.html')

@app.route('/video2audio/convert', methods=['POST'])
def video2audio_convert():
    print("\n🎬 === VIDEO→AUDIO CONVERT ROUTE ===")
    # 你可以選擇是否清理 S_audio，這裡不清理，以免刪掉分析用的檔案
    # cleanup_s_audio()

    if 'video' not in request.files:
        return "No file part", 400

    file = request.files['video']
    if file.filename == '':
        return "No file selected", 400

    if not allowed_video(file.filename):
        return "Invalid video type", 400

    target_format = request.form.get('format', 'mp3')  # 'mp3' or 'wav'
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    base_name = os.path.splitext(filename)[0]

    # 存到 S_audio 資料夾
    input_path = os.path.join(UPLOAD_FOLDER, f"video_{timestamp}_{filename}")
    file.save(input_path)

    output_filename = f"{base_name}_{timestamp}.{target_format}"
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)

    try:
        stream = ffmpeg.input(input_path)
        if target_format == 'mp3':
            stream = ffmpeg.output(stream.audio, output_path, acodec='libmp3lame')
        else:  # wav
            stream = ffmpeg.output(stream.audio, output_path, acodec='pcm_s16le')

        ffmpeg.run(stream, overwrite_output=True, quiet=True)

        # 轉完可以刪掉原影片
        os.unlink(input_path)

        print(f"✅ Video converted to {output_filename}")
        print("🎬 === VIDEO→AUDIO CONVERT COMPLETE ===\n")

        # 直接觸發下載
        return send_from_directory(UPLOAD_FOLDER, output_filename, as_attachment=True)

    except Exception as e:
        print(f"❌ Video→Audio convert error: {e}")
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
        return f"Convert error: {e}", 500


@app.route('/video2audio/convert_api', methods=['POST'])
def video2audio_convert_api():
    print("\n🎬 === VIDEO→AUDIO CONVERT (API) ===")

    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    if not allowed_video(file.filename):
        return jsonify({'success': False, 'error': 'Invalid video type'}), 400

    target_format = request.form.get('format', 'wav')  # for analyze, WAV is easier
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    base_name = os.path.splitext(filename)[0]

    input_path = os.path.join(UPLOAD_FOLDER, f"video_{timestamp}_{filename}")
    file.save(input_path)

    output_filename = f"{base_name}_{timestamp}.{target_format}"
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)

    try:
        stream = ffmpeg.input(input_path)
        if target_format == 'mp3':
            stream = ffmpeg.output(stream.audio, output_path, acodec='libmp3lame')
        else:  # wav
            stream = ffmpeg.output(stream.audio, output_path, acodec='pcm_s16le')

        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        os.unlink(input_path)

        print(f"✅ Video converted to {output_filename}")
        print("🎬 === VIDEO→AUDIO CONVERT (API) COMPLETE ===\n")

        return jsonify({
            'success': True,
            'filename': output_filename
        })

    except Exception as e:
        print(f"❌ Video→Audio convert error: {e}")
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
        return jsonify({'success': False, 'error': str(e)}), 500

# 🔥 ➕ 新增這一個：自動清理暫存音檔 API（給 realtime_monitor.js 用）
@app.route('/delete_temp', methods=['POST'])
def delete_temp():
    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({'success': False, 'error': 'No filename'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"🧹 Deleted realtime temp file: {filename}")
        return jsonify({'success': True})
    except Exception as e:
        print(f"❌ Failed to delete temp file {filename}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(
        host='0.0.0.0', 
        port=5001, 
        threaded=True,        # 🔥 啟用多線程
        debug=False
    )
