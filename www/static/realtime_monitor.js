// Realtime monitor: 4s chunks -> 16kHz WAV -> upload -> analyze
// - Threshold from UI select
// - Optional auto delete temp file on server

let mediaStream = null;
let audioContext = null;
let mediaSource = null;
let scriptNode = null;
let recordingLoop = false;

const CHUNK_SECONDS = 4;          // 每段長度 4 秒
const SAMPLE_RATE = 16000;        // 16kHz mono
const AUTO_DELETE_SERVER_FILE = true;  // 若 true，分析後請求後端刪除暫存檔

let bufferData = [];              // 收集 Float32 audio samples
let chunkStartTime = 0;           // 當前 chunk 開始時間（AudioContext 時間）

let startBtn, stopBtn, confirmBtn, statusEl, resultEl, timerEl, lastFileEl, previewAudio, thresholdSelect;

document.addEventListener('DOMContentLoaded', () => {
    startBtn = document.getElementById('startBtn');
    stopBtn = document.getElementById('stopBtn');
    confirmBtn = document.getElementById('confirmBtn');
    statusEl = document.getElementById('status');
    resultEl = document.getElementById('result');
    timerEl = document.getElementById('timer');
    lastFileEl = document.getElementById('lastFile');
    previewAudio = document.getElementById('previewAudio');
    thresholdSelect = document.getElementById('thresholdSelect');

    startBtn.onclick = startMonitor;
    stopBtn.onclick = stopMonitor;
    confirmBtn.onclick = () => {
        // 清除 fake 提示，重新啟動監聽
        resultEl.textContent = '';
        resultEl.className = '';
        lastFileEl.textContent = '';
        previewAudio.style.display = 'none';
        previewAudio.src = '';
        confirmBtn.style.display = 'none';
        startMonitor();
    };
});

async function startMonitor() {
    if (recordingLoop) return;

    try {
        statusEl.textContent = '🎙️ 正在請求麥克風權限...';
        resultEl.textContent = '';
        resultEl.className = '';
        lastFileEl.textContent = '';
        previewAudio.style.display = 'none';
        previewAudio.src = '';

        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: false,
                noiseSuppression: false,
                channelCount: 1
            }
        });

        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: SAMPLE_RATE
        });

        mediaSource = audioContext.createMediaStreamSource(mediaStream);
        const bufferSize = 4096;
        scriptNode = audioContext.createScriptProcessor(bufferSize, 1, 1);

        recordingLoop = true;
        bufferData = [];
        chunkStartTime = audioContext.currentTime;

        startBtn.disabled = true;
        stopBtn.disabled = false;
        confirmBtn.style.display = 'none';
        statusEl.textContent = '🟢 監聽中，每 4 秒分析一次 (16kHz WAV)...';

        // 顯示監聽總時間
        const startedAt = Date.now();
        const timerId = setInterval(() => {
            if (!recordingLoop) {
                clearInterval(timerId);
                return;
            }
            const sec = Math.floor((Date.now() - startedAt) / 1000);
            const mm = String(Math.floor(sec / 60)).padStart(2, '0');
            const ss = String(sec % 60).padStart(2, '0');
            timerEl.textContent = `Monitoring: ${mm}:${ss}`;
        }, 500);

        scriptNode.onaudioprocess = (event) => {
            if (!recordingLoop) return;

            const input = event.inputBuffer.getChannelData(0); // Float32Array
            bufferData.push(new Float32Array(input));         // 存 buffer

            const elapsedChunk = audioContext.currentTime - chunkStartTime;
            if (elapsedChunk >= CHUNK_SECONDS) {
                // 取一段 chunk 出來處理
                const chunk = mergeFloat32Arrays(bufferData);
                bufferData = [];
                chunkStartTime = audioContext.currentTime;

                // 轉成 16kHz wav Blob
                const wavBlob = encodeWAV(chunk, SAMPLE_RATE);

                // 上傳 + 分析（不 await，讓下一段持續錄）
                handleChunk(wavBlob);
            }
        };

        mediaSource.connect(scriptNode);
        scriptNode.connect(audioContext.destination);

    } catch (err) {
        console.error('Mic error:', err);
        statusEl.textContent = `❌ 麥克風錯誤: ${err.message}`;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        recordingLoop = false;
    }
}

function stopMonitor() {
    recordingLoop = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    timerEl.textContent = 'Idle';

    try {
        if (scriptNode) scriptNode.disconnect();
        if (mediaSource) mediaSource.disconnect();
        if (audioContext) audioContext.close();
        if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
    } catch (e) {
        console.warn('Cleanup error:', e);
    }

    statusEl.textContent = '⏹ 已停止監聽';
}

// 合併多個 Float32Array
function mergeFloat32Arrays(chunks) {
    let length = 0;
    for (const c of chunks) length += c.length;
    const result = new Float32Array(length);
    let offset = 0;
    for (const c of chunks) {
        result.set(c, offset);
        offset += c.length;
    }
    return result;
}

// 從 record_actions.js 移植的 encodeWAV (16kHz mono)
function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    function writeString(offset, str) {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset + i, str.charCodeAt(i));
        }
    }

    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = sampleRate * numChannels * bitsPerSample / 8;
    const blockAlign = numChannels * bitsPerSample / 8;

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

// 處理每一個 4 秒 chunk：上傳 → analyze → 依 threshold 顯示結果
async function handleChunk(wavBlob) {
    try {
        statusEl.textContent = '📤 上傳 4 秒音訊 (WAV) 並分析中...';

        const tempFile = new File([wavBlob], `rt_${Date.now()}.wav`, { type: 'audio/wav' });

        const formData = new FormData();
        formData.append('file', tempFile);

        // 1) 上傳
        const uploadRes = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const uploadJson = await uploadRes.json();

        if (!uploadJson.success) {
            statusEl.textContent = `❌ Upload failed: ${uploadJson.error}`;
            return;
        }

        const serverFilename = uploadJson.filename;
        console.log('Uploaded realtime chunk as:', serverFilename);

        // 2) 分析
        const analyzeRes = await fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: serverFilename })
        });
        const analyzeJson = await analyzeRes.json();

        if (!analyzeJson.success) {
            statusEl.textContent = `❌ Analysis failed: ${analyzeJson.error}`;
            return;
        }

        const output = JSON.parse(analyzeJson.result);
        console.log('Realtime output:', output);

        // 🎚 讀取 UI 上的 threshold (Real >= threshold 才算 REAL)
        const th = parseFloat(thresholdSelect.value || '0.5');  // 預設 0.5
        const isReal = output.prob_real >= th;

        if (isReal) {
            resultEl.textContent = '✅ REAL';
            resultEl.className = 'green';
            statusEl.textContent =
                `🟢 最新 4 秒為真實語音 (Real ${(output.prob_real*100).toFixed(1)}% ≥ ${(th*100).toFixed(0)}%)，持續監聽中...`;
            // 不停，下一段會自動繼續
        } else {
            resultEl.textContent = '❌ FAKE';
            resultEl.className = 'red';
            statusEl.textContent =
                `🚨 偵測到 Fake 語音 (Real ${(output.prob_real*100).toFixed(1)}% < ${(th*100).toFixed(0)}%)，已暫停監聽。`;

            // 播放該段
            previewAudio.src = `/download/${serverFilename}`;
            previewAudio.style.display = 'block';
            lastFileEl.textContent = `Last chunk on server: ${serverFilename}`;

            // 停止監聽
            stopMonitor();
            confirmBtn.style.display = 'inline-block';

            // 若啟用自動刪檔：通知後端刪除暫存檔
            if (AUTO_DELETE_SERVER_FILE) {
                try {
                    await fetch('/delete_temp', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ filename: serverFilename })
                    });
                    console.log('Requested delete of temp file:', serverFilename);
                } catch (e) {
                    console.warn('Delete temp failed:', e);
                }
            }
        }
    } catch (err) {
        console.error('Realtime handleChunk error:', err);
        statusEl.textContent = `❌ Realtime error: ${err.message}`;
    }
}

