# ASV ONNX Test - Android Audio Classification App

An Android application that performs real-time audio classification using ONNX Runtime. The app allows users to select audio files and runs inference through a pre-trained ONNX model to classify the audio into two categories.

## 📱 Features

- Select audio files from device storage (.wav, raw PCM)
- Run ONNX model inference on selected audio
- Display classification results (Class 0 or Class 1)
- Show confidence scores and logits
- Support for various audio formats (16-bit and 32-bit PCM)
- Automatic audio preprocessing (normalization, padding/truncation)

## 🏗️ Tech Stack

- **Language**: Kotlin
- **UI Framework**: Jetpack Compose
- **ML Runtime**: ONNX Runtime Android 1.19.0
- **Minimum SDK**: API 24 (Android 7.0)
- **Target SDK**: API 36
- **Build Tool**: Gradle (Kotlin DSL)

## 📋 Prerequisites

- Android Studio Ladybug (2024.2.1) or later
- JDK 11 or later
- Android device or emulator with API 24+

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ASVOnnxTest.git
cd ASVOnnxTest
```

### 2. Add your model file

Place your ONNX model file in the `app/src/main/assets/` directory and name it `model.onnx`

```
app/src/main/assets/model.onnx
```

### 3. Build and run

Open the project in Android Studio and click **Run** (▶️) or use Gradle:

```bash
./gradlew assembleDebug
./gradlew installDebug
```

## 📁 Project Structure

```
ASVOnnxTest/
├── app/
│   ├── src/main/
│   │   ├── java/hk/omyu/asvonnxtest/
│   │   │   └── MainActivity.kt          # Main application logic
│   │   ├── res/                          # Android resources
│   │   └── assets/                       # Model and test audio files
│   │       ├── model.onnx                # Your ONNX model (required)
│   │       └── test.wav                  # Test audio file (optional)
│   └── build.gradle.kts                  # App-level build configuration
├── gradle/
└── build.gradle.kts                      # Project-level build configuration
```

## 🎯 How It Works

### Audio Processing Pipeline

1. **File Selection**: User picks an audio file from device storage
2. **Preprocessing**:
   - Reads WAV header or raw PCM data
   - Converts to mono (averages channels if needed)
   - Normalizes audio (zero mean, unit variance)
   - Pads or truncates to 77,824 samples
3. **Inference**: Runs ONNX model using ONNX Runtime
4. **Output**: Returns logits for 2 classes with confidence scores

### Model Input/Output

| Property | Value |
|----------|-------|
| **Input Shape** | `[1, 77824]` (batch_size, audio_samples) |
| **Input Type** | Float32 |
| **Output Shape** | `[1, 2]` (batch_size, num_classes) |
| **Output Type** | Float32 (logits) |

## 📖 Usage

1. **Launch the app**
2. **Tap "Select Audio File"** to choose an audio file from your device
3. **Tap "Run Inference"** to process the selected audio
4. **View results**:
   - Predicted class (Class 0 or Class 1)
   - Raw logit values
   - Confidence level

### Example Output

```
File: speech_sample.wav
logits length: 2
max logit: 5.2146597
argmax: 0
✅ Class 0 predicted
```

## 🔧 Configuration

### Model Requirements

Your ONNX model must:
- Accept input shape `[1, 77824]` of Float32
- Output `[1, 2]` logits (2 classes)
- Use input name `"input_values"` (can be changed in code)

### Audio Requirements

Supported audio formats:
- WAV (16-bit or 32-bit PCM)
- Raw PCM (16-bit)
- Mono or stereo (automatically converted to mono)
- Any sample rate (automatically handled)

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Model not found** | Ensure `model.onnx` is in `app/src/main/assets/` |
| **Out of memory** | Model is 1.2GB; ensure device has sufficient RAM |
| **NaN in output** | Check audio file format; try a different audio file |
| **File picker doesn't open** | Check storage permissions in Android settings |

### Logging

Enable debug logging to see detailed processing steps:

```bash
adb logcat | grep ONNXTest
```

## 📦 Dependencies

```kotlin
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.0")
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("androidx.activity:activity-compose:1.8.0")
    implementation("androidx.compose.ui:ui:1.5.4")
    implementation("androidx.compose.material3:material3:1.1.2")
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [ONNX Runtime](https://github.com/microsoft/onnxruntime) for Android inference
- [Jetpack Compose](https://developer.android.com/jetpack/compose) for modern UI

## 📞 Support

For issues or questions:
1. Check the [Issues](https://github.com/yourusername/ASVOnnxTest/issues) page
2. Enable debug logging and capture logcat output
3. Include model details and audio file information

---

**Note**: You need to provide your own `model.onnx` file. The app will copy it from assets to external storage on first run (requires ~1.2GB free space).
```

## Also Create a `.gitignore` file:

```gitignore
# Android
*.iml
.gradle/
/local.properties
/.idea/
.DS_Store
/build/
/captures/
.externalNativeBuild/
.cxx/

# Java
*.class

# Kotlin
*.kotlin_module

# Logs
*.log

# OS
Thumbs.db
Desktop.ini

# Model files (too large for GitHub)
*.onnx
*.pb
*.tflite

# Audio files
*.wav
*.mp3
*.m4a

# Keystore
*.jks
*.keystore
```

## Quick Commands for Git Upload:

```bash
# Initialize git repository
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: ASV ONNX Test Android app"

# Add remote repository (replace with your repo URL)
git remote add origin https://github.com/yourusername/ASVOnnxTest.git

# Push to GitHub
git branch -M main
git push -u origin main
