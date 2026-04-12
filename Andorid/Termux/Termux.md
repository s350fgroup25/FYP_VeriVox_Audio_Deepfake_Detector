# Termux Deployment Workflow Report

## VeriVox Android (Termux) Deployment Complete Guide

---

## 1. Environment Setup

### 1.1 Install Termux

| Step | Action | Note |
|------|--------|------|
| 1 | Uninstall Google Play version of Termux | Too old, not working |
| 2 | Download Termux from **F-Droid** | https://f-droid.org/packages/com.termux/ |
| 3 | Install Termux 0.118.0 or higher | |
| 4 | Open Termux app | |

### 1.2 Verify Termux Version

```bash
echo $TERMUX_VERSION
```

**Expected output:**
```
0.118.0
```

---

## 2. Basic Environment Setup

### 2.1 Update Package Manager

```bash
pkg update && pkg upgrade -y
```

### 2.2 Grant Storage Access

```bash
termux-setup-storage
```

> A permission popup will appear. Click **Allow**

### 2.3 Install Python

```bash
pkg install python -y
```

Verify:
```bash
python --version
```

**Expected output:**
```
Python 3.11.x or higher
```

### 2.4 Install System Dependencies

```bash
pkg install ffmpeg -y
pkg install libsndfile -y
pkg install cmake ninja -y
pkg install openssh -y
```

---

## 3. Python Package Installation

### 3.1 Upgrade pip

```bash
pip install --upgrade pip
```

### 3.2 Install PyTorch

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3.3 Install Other Dependencies

```bash
pip install transformers
pip install safetensors
pip install soundfile
pip install numpy
pip install Flask
```

### 3.4 Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers OK')"
python -c "import safetensors; print('Safetensors OK')"
python -c "import soundfile; print('Soundfile OK')"
```

---

## 4. Project File Preparation

### 4.1 Create Project Directory

```bash
mkdir -p ~/storage/shared/ASV
cd ~/storage/shared/ASV
```

### 4.2 Python Files to Upload to GitHub

| File Name | Purpose | Required |
|-----------|---------|----------|
| `single.py` | Single audio inference entry | ✅ Required |
| `model.py` | Model definition (HFReadyModel) | ✅ Required |
| `model.safetensors` | Model weights file | ✅ Required |
| `wav2vec2-xls-r-300m-feature/` | Feature extractor folder | ✅ Required |
| `wav2vec2-xls-r-300m-model/` | Wav2Vec2 model folder | ✅ Required |
| `requirements.txt` | Dependency list | Recommended |
| `README.md` | Project documentation | Recommended |

### 4.3 Copy Files from Raspberry Pi to Termux

```bash
# Copy single files
scp pi@192.168.x.x:/home/carmen/asvspoof/program/single.py ~/storage/shared/ASV/
scp pi@192.168.x.x:/home/carmen/asvspoof/program/model.py ~/storage/shared/ASV/
scp pi@192.168.x.x:/home/carmen/asvspoof/program/model.safetensors ~/storage/shared/ASV/

# Copy folders
scp -r pi@192.168.x.x:/home/carmen/asvspoof/program/wav2vec2-xls-r-300m-feature ~/storage/shared/ASV/
scp -r pi@192.168.x.x:/home/carmen/asvspoof/program/wav2vec2-xls-r-300m-model ~/storage/shared/ASV/
```

### 4.4 Verify Files

```bash
ls -la ~/storage/shared/ASV/
```

**Expected output:**
```
model.safetensors
model.py
single.py
wav2vec2-xls-r-300m-feature/
wav2vec2-xls-r-300m-model/
```

---

## 5. Run single.py

### 5.1 Enter Directory

```bash
cd ~/storage/shared/ASV
```

### 5.2 Set Environment Variable (Optional, remove warning)

```bash
export OPENBLAS_CORETYPE=ARMV8
```

### 5.3 Run Inference

```bash
python single.py --audio record_sample2.wav --backend pytorch
```

### 5.4 One-Line Command

```bash
cd ~/storage/shared/ASV && export OPENBLAS_CORETYPE=ARMV8 && python single.py --audio record_sample2.wav
```

---

## 6. Expected Output

### 6.1 Successful Output Example

```
[1] Model init & load: 2.35 s
[2] FeatureExtractor load: 0.12 s
[3] Audio loading: 0.05 s
[4] Feature extraction: 0.08 s
[5] Inference & softmax: 0.15 s
Positive class probability: 0.3500
```

### 6.2 Failed Output Example (Weight Mismatch)

```
Wav2Vec2Model LOAD REPORT from: ./wav2vec2-xls-r-300m
Key                          | Status
-----------------------------+----------
project_q.weight             | UNEXPECTED
quantizer.codevectors        | UNEXPECTED
...
Positive class probability: 0.9924 (WRONG)
```

---

## 7. Files to Upload to GitHub

### 7.1 Required Files Structure

```
ASV/
├── single.py                 # Single audio inference script
├── model.py                  # Model definition (HFReadyModel)
├── model.safetensors         # Model weights (~1.2 GB)
├── wav2vec2-xls-r-300m-feature/   # Feature extractor folder
│   ├── preprocessor_config.json
│   └── config.json
├── wav2vec2-xls-r-300m-model/     # Wav2Vec2 model folder
│   ├── model.safetensors
│   └── config.json
└── requirements.txt          # Dependency list
```

### 7.2 requirements.txt Content

```txt
torch>=2.0.0
transformers>=4.30.0
safetensors>=0.3.0
soundfile>=0.12.0
numpy>=1.24.0
Flask>=2.3.0
```

### 7.3 Upload to GitHub Commands

```bash
# On Raspberry Pi, package all files
cd ~/asvspoof/program
tar -czf termux_files.tar.gz single.py model.py model.safetensors wav2vec2-xls-r-300m-feature wav2vec2-xls-r-300m-model requirements.txt

# Upload to GitHub Releases or use Git LFS for large files
```

---

## 8. Common Issues and Solutions

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'torch'` | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `soundfile` installation fails | `pkg install libsndfile` then retry |
| `scipy` / `librosa` installation fails | Use NumPy manual resampling instead |
| HuggingFace TLS error | Use local model folder, not online download |
| Weight loading UNEXPECTED keys | Use `strict=False` when loading |
| Inference mismatch (0.99 vs 0.35) | ❌ NOT RESOLVED, Android deployment failed |

---

## 9. Summary

| Item | Status |
|------|--------|
| Termux installation | ✅ Success |
| Python environment | ✅ Success |
| PyTorch installation | ✅ Success |
| Dependency packages | ⚠️ Partial (scipy/librosa failed) |
| Model loading | ⚠️ Partial (strict=False) |
| Inference correctness | ❌ Failed (0.99 vs 0.35) |

**Conclusion:** Termux environment can run `single.py`, but the inference results do not match Raspberry Pi. Android deployment was not successful.

---

This report can be saved as `TERMUX_DEPLOYMENT.md` in your GitHub repository.
