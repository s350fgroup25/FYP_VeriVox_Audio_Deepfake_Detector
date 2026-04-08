let mediaRecorder = null;           
let audioChunks = [];               
let recordedBlob = null;
let recordedFile = null;
let previewUrl = null;

document.addEventListener('DOMContentLoaded', function () {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        const status = document.getElementById('status');
        const recordBtn = document.getElementById('recordBtn');
        if (status) {
            status.textContent = '❌ This browser does not support microphone recording (no mediaDevices.getUserMedia).';
        }
        if (recordBtn) {
            recordBtn.disabled = true;
        }
        return;
    }
    
    const recordBtn = document.getElementById('recordBtn');
    const playBtn = document.getElementById('playBtn');
    const submitBtn = document.getElementById('submitBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const status = document.getElementById('status');
    const progressBar = document.getElementById('progressBar');
    const result = document.getElementById('result');
    const timeDisplay = document.getElementById('timeDisplay');
    const audioPlayer = document.getElementById('audioPlayer');

    let timer = null;
    let startTime = null;
    const MAX_SECONDS = 5;

    let audioContext = null;
    let mediaStream = null;
    let mediaSource = null;
    let scriptNode = null;
    let pcmData = [];     
    let recording = false;

    // 初始化
    playBtn.disabled = true;
    submitBtn.disabled = true;
    downloadBtn.style.display = 'none';
    progressBar.style.display = 'none';
    timeDisplay.textContent = '00:00.0';

    function formatTime(sec) {
        const s = Math.floor(sec * 10) / 10;
        return `${Math.floor(s/60).toString().padStart(2,'0')}:${(s%60).toFixed(1).padStart(4,'0')}`;
    }

    async function startRecording() {
        recordedBlob = null;
        recordedFile = null;
        pcmData = [];
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl = null;
        audioPlayer.src = '';
        result.textContent = '';
        result.className = '';
        downloadBtn.style.display = 'none';

        try {
            status.textContent = '🎙️ Requesting microphone...';

            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    channelCount: 1
                }
            });

            audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000
            });

            mediaSource = audioContext.createMediaStreamSource(mediaStream);
            const bufferSize = 4096;
            scriptNode = audioContext.createScriptProcessor(bufferSize, 1, 1);

            scriptNode.onaudioprocess = (event) => {
                if (!recording) return;
                const inputBuffer = event.inputBuffer.getChannelData(0);
                pcmData.push(new Float32Array(inputBuffer));
            };

            mediaSource.connect(scriptNode);
            scriptNode.connect(audioContext.destination);

            recording = true;
            status.textContent = '🎙️ Recording 16kHz WAV... Speak clearly!';
            recordBtn.textContent = '🛑 Stop';
            playBtn.disabled = true;
            submitBtn.disabled = true;
            downloadBtn.disabled = true;

            startTime = Date.now();
            timer = setInterval(() => {
                const elapsed = (Date.now() - startTime) / 1000;
                timeDisplay.textContent = formatTime(elapsed);
                if (elapsed >= MAX_SECONDS) stopRecording();
            }, 100);

        } catch (err) {
            console.error(err);
            status.textContent = `❌ Mic error: ${err.message}`;
        }
    }

    function stopRecording() {
        if (!recording) return;
        recording = false;

        if (timer) {
            clearInterval(timer);
            timer = null;
        }

        status.textContent = '⏳ Finalizing WAV...';
        recordBtn.disabled = true;

        try {
            if (scriptNode) scriptNode.disconnect();
            if (mediaSource) mediaSource.disconnect();
            if (audioContext) audioContext.close();
            if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
        } catch (e) {
            console.warn('Cleanup error:', e);
        }

        let length = 0;
        for (const chunk of pcmData) length += chunk.length;
        if (length === 0) {
            status.textContent = '❌ No audio captured';
            recordBtn.textContent = '🎤 Start 5s';
            recordBtn.disabled = false;
            return;
        }

        const monoData = new Float32Array(length);
        let offset = 0;
        for (const chunk of pcmData) {
            monoData.set(chunk, offset);
            offset += chunk.length;
        }

        recordedBlob = encodeWAV(monoData, 16000);
        console.log('🎵 Recorded WAV size:', recordedBlob.size, 'bytes');

        recordedFile = new File(
            [recordedBlob],
            `recording_${Date.now()}.wav`,
            { type: 'audio/wav' }
        );

        previewUrl = URL.createObjectURL(recordedBlob);
        audioPlayer.src = previewUrl;

        status.textContent = `✅ Recorded WAV ${Math.round(recordedBlob.size/1000)}KB - Ready!`;
        recordBtn.textContent = '🔄 Re-record';
        recordBtn.disabled = false;
        playBtn.disabled = false;
        submitBtn.disabled = false;
        downloadBtn.style.display = 'inline-block';
        downloadBtn.disabled = false;
    }

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

    downloadBtn.onclick = function() {
        if (recordedFile) {
            const url = URL.createObjectURL(recordedFile);
            const a = document.createElement('a');
            a.href = url;
            a.download = recordedFile.name;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            status.textContent = '💾 Downloaded WAV! Open with VLC/Media Player';
        }
    };

    // 🔥 UPDATED Submit：10% threshold, no %
    submitBtn.onclick = async function() {
        if (!recordedFile) {
            status.textContent = '❌ No recording to submit';
            return;
        }

        const uploadStart = Date.now();
        timeDisplay.textContent = '00:00.0';

        progressBar.style.display = 'block';
        progressBar.value = 0;
        status.textContent = '📤 Uploading...';
        result.textContent = '';
        result.className = '';
        submitBtn.disabled = true;
        recordBtn.disabled = true;
        playBtn.disabled = true;
        downloadBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', recordedFile);

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/upload');

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                progressBar.value = (e.loaded / e.total) * 50;
            }
        };

        xhr.onload = async () => {
            try {
                const res = JSON.parse(xhr.responseText);
                if (!res.success) {
                    status.textContent = `❌ Upload failed: ${res.error}`;
                } else {
                    status.textContent = '🔬 AI Analyzing...';
                    progressBar.value = 50;

                    const analyzeRes = await fetch('/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ filename: res.filename })
                    });
                    const analyzeData = await analyzeRes.json();

                    const totalTime = ((Date.now() - uploadStart) / 1000).toFixed(1);
                    timeDisplay.textContent = `Done: ${totalTime}s`;

                    if (analyzeData.success) {
                        const output = JSON.parse(analyzeData.result);
                        
                        // 🔥 NEW LOGIC: Real > 10% = REAL, else FAKE (NO %)
                        if (output.prob_real > 0.1) {
                            result.textContent = '✅ REAL AUDIO';
                            result.className = 'green';
                            console.log(`🎯 Record: Real=${output.prob_real.toFixed(4)}>0.1 → REAL`);
                        } else {
                            result.textContent = '❌ FAKE';
                            result.className = 'red';
                            console.log(`🎯 Record: Real=${output.prob_real.toFixed(4)}≤0.1 → FAKE`);
                        }
                        
                        status.textContent = '🎉 Analysis complete!';
                    } else {
                        status.textContent = `❌ Analysis failed: ${analyzeData.error}`;
                    }
                }
            } catch (err) {
                status.textContent = '❌ Network/Analysis error';
                console.error('Submit error:', err);
            } finally {
                progressBar.style.display = 'none';
                submitBtn.disabled = false;
                recordBtn.disabled = false;
                playBtn.disabled = false;
                downloadBtn.disabled = false;
            }
        };

        xhr.onerror = () => {
            status.textContent = '❌ Network error';
            progressBar.style.display = 'none';
            submitBtn.disabled = false;
            recordBtn.disabled = false;
            playBtn.disabled = false;
            downloadBtn.disabled = false;
        };

        xhr.send(formData);
    };

    recordBtn.onclick = () => recording ? stopRecording() : startRecording();
    playBtn.onclick = () => audioPlayer.play();
});

