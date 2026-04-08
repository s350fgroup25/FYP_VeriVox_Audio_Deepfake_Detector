from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import subprocess
import json

UPLOAD_FOLDER = '/home/carmen/asvspoof/S_audio'
ALLOWED_EXTENSIONS = {'flac', 'wav', 'mp3'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB

app = Flask(__name__,
            template_folder='/home/carmen/asvspoof/www/templates',
            static_folder='/home/carmen/asvspoof/www/static')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part in request'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Unsupported file type'})
    
    # 清空資料夾確保只有一個上傳檔案
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
    
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    return jsonify({'success': True, 'filename': filename})

@app.route('/analyze', methods=['POST'])
def analyze_file():
    data = request.json
    filename = data.get('filename')
    if not filename:
        return jsonify({'success': False, 'error': 'Filename is required'})

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'File not found on server'})

    result = subprocess.run(['python3', '/home/carmen/asvspoof/program/eval_TF_.py', file_path],
                            capture_output=True, text=True)

    if result.returncode != 0:
        return jsonify({'success': False, 'error': 'Model evaluation failed', 'details': result.stderr})

    # Parse the JSON string output from eval_TF.py
    result_dict = json.loads(result.stdout)

    # Add success key if you want
    result_dict['success'] = True

    # Return as JSON response directly, so frontend gets proper boolean values
    return jsonify(result_dict)

if __name__ == '__main__':
    app.run(debug=True,port=5000)

