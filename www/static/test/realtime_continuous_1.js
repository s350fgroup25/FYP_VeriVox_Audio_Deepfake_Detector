// 🔥 連續監聽：每4秒分析一次，永不停止，完美匹配你的 Model (Real: 0.0069, Fake: 0.9931)
let mediaStream = null;
let audioContext = null;
let mediaSource = null;
let scriptNode = null;
let isMonitoring = false;

const CHUNK_SECONDS = 4;
const SAMPLE_RATE = 16000;
let bufferData = [];
let chunkStartTime = 0;
let totalTime = 0;
let chunkCounter = 0;
let silentCount = 0;

let startBtn, stopBtn, statusEl, timerEl, resultsContainer, thresholdSelect;

document.addEventListener('DOMContentLoaded', () => {
    startBtn = document.getElementById('startBtn');
    stopBtn = document.getElementById('stopBtn');
    statusEl = document.getElementById('status');
    timerEl = document.getElementById('timer');
    resultsContainer = document.getElementById('resultsContainer');
    thresholdSelect = document.getElementById('thresholdSelect');

    startBtn.onclick = startContinuousMonitor;
    stopBtn.onclick = stopContinuousMonitor;
});

async function startContinuousMonitor() {
    if (isMonitoring) return;

    try {
        statusEl.textContent = '🎙️ 請求麥克風權限...';
        resultsContainer.innerHTML = '<div style="opacity: 0.7;">開始連續監聽...</div>';
        
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: { 
                echoCancellation: false, 
                noiseSuppression: false, 
                channelCount: 1,
                sampleRate: SAMPLE_RATE
            }
        });

        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
        mediaSource = audioContext.createMediaStreamSource(mediaStream);
        const bufferSize = 4096;
        scriptNode = audioContext.createScriptProcessor(bufferSize, 1, 1);

        isMonitoring = true;
        bufferData = [];
        chunkStartTime = audioContext.currentTime;
        chunkCounter = 0;
        totalTime = Date.now();
        silentCount = 0;

        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusEl.textContent = '🔊 確認錄音音量...';

        // 總時間計時器
        const timerId = setInterval(() => {
            if (!isMonitoring) clearInterval(timerId);
            const sec = Math.floor((Date.now() - totalTime) / 1000);
            const mm = String(Math.floor(sec / 60)).padStart(2, '0');
            const ss = String(sec % 60).padStart(2, '0');
            timerEl.textContent = `總監聽時間: ${mm}:${ss}`;
        }, 1000);

        scriptNode.onaudioprocess = (event) => {
            if (!isMonitoring) return;

            const input = event.inputBuffer.getChannelData(0);
            
            // 🔥 音量檢查（RMS）
            let rms = 0;
            for (let i = 0; i < input.length; i++) {
                rms += input[i] * input[i];
            }
            rms = Math.sqrt(rms / input.length);

            if (rms < 0.01) silentCount++;
            else silentCount = 0;

            bufferData.push(new Float32Array(input));

            const elapsedChunk = audioContext.currentTime - chunkStartTime;
            if (elapsedChunk >= CHUNK_SECONDS) {
                const chunk = mergeFloat32Arrays(bufferData);
                console.log(`🔍 Chunk ${chunkCounter}: ${chunk.length} samples, RMS=${(rms*100).toFixed(2)}%`);
                
                bufferData = [];
                chunkStartTime = audioContext.currentTime;
                chunkCounter++;

                statusEl.textContent = `📤 分析第 ${chunkCounter} 段 (${((chunkCounter-1)*4).toString().padStart(2,'0')}s-${(chunkCounter*4).toString().padStart(2,'0')}s, RMS=${(rms*100).toFixed(1)}%)`;
                
                const wavBlob = encodeWAV(chunk, SAMPLE_RATE);
                analyzeChunk(wavBlob, chunkCounter, rms);
            }
        };

        mediaSource.connect(scriptNode);
        scriptNode.connect(audioContext.destination);

    } catch (err) {
        console.error('❌ Mic error:', err);
        statusEl.textContent = `❌ 麥克風錯誤: ${err.message}`;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        isMonitoring = false;
    }
}

function stopContinuousMonitor() {
    isMonitoring = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    timerEl.textContent = '已停止';
    statusEl.textContent = '⏹ 監聽已停止';

    try {
        if (scriptNode) scriptNode.disconnect();
        if (mediaSource) mediaSource.disconnect();
        if (audioContext) audioContext.close();
        if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
    } catch (e) {
        console.warn('Cleanup error:', e);
    }
}

// 🔥 合併多個 Float32Array（修復截斷問題）
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

// 🔥 WAV Encoder（16kHz mono）
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

// 🔥 完美匹配你 Model：只看 prob_real >= threshold
async function analyzeChunk(wavBlob, chunkNum, rms) {
    try {
        const formData = new FormData();
        formData.append('file', new File([wavBlob], `rt_${chunkNum}.wav`, { type: 'audio/wav' }));

        // 1. 上傳
        const uploadRes = await fetch('/upload', { method: 'POST', body: formData });
        const uploadJson = await uploadRes.json();

        if (!uploadJson.success) {
            addResultLine(chunkNum, '❌', 'Upload failed', 'red', rms);
            return;
        }

        // 2. 你 Model 分析
        const analyzeRes = await fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: uploadJson.filename })
        });
        const analyzeJson = await analyzeRes.json();

        console.log('🔍 Model raw:', analyzeJson);

        if (analyzeJson.success) {
            const output = JSON.parse(analyzeJson.result);
            console.log(`🎯 Chunk ${chunkNum}: Real=${output.prob_real.toFixed(4)}, Fake=${output.prob_fake.toFixed(4)}`);
            
            // 🔥 只按 prob_real 判斷 (你的需求)
            const th = parseFloat(thresholdSelect.value);
            const isReal = output.prob_real >= th;
            const displayProb = `${(output.prob_real*100).toFixed(1)}%`;
            
            addResultLine(chunkNum, isReal ? '✅ REAL' : '❌ FAKE', isReal ? 'green' : 'red', displayProb, rms);
        } else {
            addResultLine(chunkNum, '❌', 'Analyze failed', 'red', rms);
        }
    } catch (err) {
        console.error('Analyze error:', err);
        addResultLine(chunkNum, '❌', 'Network error', 'red', rms);
    }
}

// 🔥 顯示結果列
function addResultLine(chunkNum, status, colorClass, prob, rms) {
    const startSec = (chunkNum - 1) * 4;
    const endSec = chunkNum * 4;
    
    const div = document.createElement('div');
    div.className = `result-line result-${colorClass}`;
    div.innerHTML = `
        <span class="time-col">${startSec.toString().padStart(2,'0')}s-${endSec.toString().padStart(2,'0')}s</span>
        <span class="status-col">${status}</span>
        <span class="prob-col">${prob} | Vol:${(rms*100).toFixed(1)}%</span>
    `;
    
    resultsContainer.insertBefore(div, resultsContainer.firstChild);
    
    while (resultsContainer.children.length > 50) {
        resultsContainer.removeChild(resultsContainer.lastChild);
    }
    resultsContainer.scrollTop = 0;
}
EOF

