# Audio Deepfake Detection System

A WavLM-based audio deepfake detection web application deployed on Raspberry Pi.

## Project Overview

This project implements an **audio deepfake detection system** capable of identifying Logical Access (LA) and Deepfake (DF) attacks. The system uses the ASVspoof 2019 dataset for initial model training and the more authoritative ASVspoof 2021 dataset for final evaluation.

### Key Features

- 🌐 **Web Interface**: Upload audio files for deepfake detection  
- 🎙️ **Real-time Recording**: Detect fake audio through microphone input  
- 📊 **Batch Evaluation**: Large-scale dataset evaluation with EER calculation  
- 📈 **Result Visualization**: Generate detection reports and charts  

## Introduction Video (Project Video)
[![VeriVox Audio Deepfake Detector Demo](https://img.youtube.com/vi/svC8c3vcg5Y/0.jpg)](https://youtu.be/svC8c3vcg5Y)

**Click the image above or [click here](https://youtu.be/svC8c3vcg5Y) to watch the demo.**

## Demo Video
---

## Project Structure
```text
asvspoof/
│
├── app.py                       # Flask web server entry point
├── program/                     # Core program code
│   ├── model.py                 # WavLM model loading and inference
│   ├── model_sentence1.py       # Sentence-level detection model v1
│   ├── model_sentence2.py       # Sentence-level detection model v2
│   ├── eval_platform.py         # Main evaluation platform
│   ├── eval_platform_N.py       # N-sample evaluation version
│   ├── eval_platform_one.py     # Single sample evaluation
│   ├── eval_2021_DF.py          # 2021 DF dataset evaluation
│   ├── eval_2021_LA.py          # 2021 LA dataset evaluation
│   ├── eval-sentence.py         # Sentence-level evaluation
│   ├── train-sentence.py        # Model training script
│   ├── train-sentence-low-memory.py   # Low memory training (Raspberry Pi)
│   ├── single_evaluate.py       # Single audio evaluation
│   ├── eer1.py                  # EER calculation script
│   ├── report_generator.py      # Report generator
│   ├── regen_df_report.py       # DF report regeneration
│   ├── retry_failed_files.py    # Retry failed files
│   ├── dataset_sentence.py      # Sentence dataset processing
│   ├── requirements.txt         # Python dependencies
│   └── __pycache__/             # Python cache (ignored)
│
├── preprocess/                  # Data preprocessing scripts
│   ├── prepare_eval_dataset_100.py     # Prepare 100 evaluation samples
│   ├── prepare_eval_dataset_100_df.py  # DF data preparation
│   ├── generate_la_eval_csv.py         # Generate LA evaluation CSV
│   ├── create_my_df_mapping.py         # Create DF mapping
│   └── *.csv                          # Test set combination files
│
├── www/                          # Web frontend files
│   ├── templates/                # HTML templates
│   │   ├── index.html            # Main page
│   │   ├── home.html             # Home page
│   │   ├── convert.html          # Audio conversion page
│   │   ├── record.html           # Recording page
│   │   ├── record_odd.html       # Recording page (backup)
│   │   ├── realtime.html         # Real-time detection
│   │   ├── realtime_continuous.html    # Continuous real-time detection
│   │   ├── fake.html             # Fake detection page
│   │   ├── video_to_audio.html   # Video to audio conversion
│   │   ├── video_to_audio_odd.html     # Video to audio (backup)
│   │   └── test/                 # Test pages
│   │
│   └── static/                   # Static assets
│       ├── actions.js            # Main interaction logic
│       ├── actions_odd.js        # Backup interaction logic
│       ├── ai-result.js          # AI result display
│       ├── convert.js            # Audio conversion logic
│       ├── fake.js               # Fake detection logic
│       ├── realtime_monitor.js   # Real-time monitoring
│       ├── realtime_continuous.js# Continuous real-time logic
│       ├── record_actions.js     # Recording operations
│       ├── video2audio.js        # Video to audio logic
│       └── images/               # Image resources
│
├── results/                      # Evaluation results
│   ├── *.csv                     # Evaluation result tables
│   └── *.json                    # Evaluation summary JSON
│
├── testSet/                      # Demo audio samples
│   ├── covent/                   # Genuine audio samples
│   ├── fake/                     # Fake audio samples
│   ├── record/                   # Recording samples
│   └── sample/                   # Example audio
│
├── asvspoof2021/                 # ⚠️ Dataset (too large, not uploaded)
│   ├── DF/                       # Deepfake audio
│   └── LA/                       # Logical access audio
│
└── datasets/                     # ⚠️ Training dataset (too large, not uploaded)
    ├── LA/                       # ASVspoof 2019 LA data
    └── models/                   # Pretrained models
        └── wavlm-epoch50/        # WavLM model files
```

---

## File Descriptions

### Core Files

| File | Description |
|------|-------------|
| `app.py` | Flask web server, handles routing, file uploads, and model inference |
| `program/model.py` | Loads WavLM pretrained model, provides audio deepfake detection |
| `program/eval_platform.py` | Core evaluation platform, calculates EER and accuracy |
| `program/requirements.txt` | Python dependencies (see below) |

### Web Frontend Files

| File | Description |
|------|-------------|
| `www/templates/index.html` | Main interface with upload and detection |
| `www/templates/realtime.html` | Real-time recording detection interface |
| `www/templates/video_to_audio.html` | Convert video to audio and detect |
| `www/static/actions.js` | Handles audio upload, API calls, result display |
| `www/static/realtime_monitor.js` | Real-time microphone audio stream processing |
| `www/static/convert.js` | Audio format conversion logic |

### Data Preprocessing Files

| File | Description |
|------|-------------|
| `preprocess/prepare_eval_dataset_100.py` | Randomly extract 100 audio samples for evaluation |
| `preprocess/generate_la_eval_csv.py` | Generate LA dataset evaluation CSV |
| `preprocess/create_my_df_mapping.py` | Create DF audio file to label mapping |

### Evaluation Scripts

| File | Description |
|------|-------------|
| `program/eval_2021_DF.py` | Evaluate model on ASVspoof 2021 DF dataset |
| `program/eval_2021_LA.py` | Evaluate model on ASVspoof 2021 LA dataset |
| `program/eer1.py` | Calculate Equal Error Rate (EER) |
| `program/report_generator.py` | Generate CSV/JSON evaluation reports |

### Training Scripts

| File | Description |
|------|-------------|
| `program/train-sentence.py` | Fine-tune WavLM model at sentence level |
| `program/train-sence-low-memory.py` | Low memory training (optimized for Raspberry Pi) |

---

## Installation & Deployment

### 1. Clone the Repository

```bash
git clone https://github.com/s350fgroup25/FYP_VeriVox_Audio_Deepfake_Detector.git
cd FYP_VeriVox_Audio_Deepfake_Detector
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# or
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r program/requirements.txt
```

### 4. Download Model Files

Due to size limitations, model files are not included in the repository. Download them from the following links and place them in `program/`:

- `wavlm-epoch50.safetensors`: [Download link to be provided]

### 5. Run the Web Server

```bash
python app.py
```

Access `http://raspberry_pi_IP:5000` in your browser.

---

## Dependencies (`requirements.txt`)

```text
flask>=2.0.0
torch>=1.12.0
torchaudio>=0.12.0
transformers>=4.25.0
librosa>=0.9.0
numpy>=1.21.0
scipy>=1.7.0
safetensors>=0.3.0
soundfile>=0.10.0
pydub>=0.25.0
```

---

## Usage Instructions

### Web Interface

1. Open browser and navigate to `http://raspberry_pi_IP:5000`  
2. Click “Choose File” to upload audio (WAV/FLAC/MP3)  
3. Click “Detect”  
4. System returns “Genuine” or “Fake” result with confidence score  

### Batch Evaluation

```bash
cd program
python eval_platform.py --dataset ../preprocess/eval_100_dataset.csv
```

### Training Model

```bash
cd program
python train-sentence-low-memory.py --epochs 10 --batch-size 4
```

---

## Dataset Information

| Dataset | Purpose | Status |
|--------|---------|--------|
| ASVspoof 2019 LA | Initial model training | Download separately |
| ASVspoof 2021 DF/LA | Final evaluation (more authoritative) | Download separately |

Download: [ASVspoof Official Website](https://www.asvspoof.org/)

---

## Evaluation Metrics

- EER (Equal Error Rate): Lower is better  
- Accuracy: Detection accuracy  
- t-DCF: Tandem Detection Cost Function  

Results are saved as CSV and JSON files in the `results/` directory.

---

## Important Notes

- Raspberry Pi Performance: Use `train-sentence-low-memory.py` for training  
- Model Files: `.safetensors` files must be downloaded separately  
- Datasets: `asvspoof2021/` and `datasets/` are not included  
- HTTPS: `cert.pem` not uploaded; use HTTP for local testing  

---

## Author

- Name: Yu Ching Ting   
- Project: ASVspoof Audio Deepfake Detection System  
- Date: April 2026  

---

## License

To be added.

---
