let currentFile = null;
let analysisTimer = null;
let startTime = null;

document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const confirmBtn = document.getElementById('confirmBtn');
    const status = document.getElementById('status');
    const progressBar = document.getElementById('progressBar');
    const result = document.getElementById('result');
    const timeDisplay = document.getElementById('timeDisplay');

    // File selection
    fileInput.onchange = function() {
        currentFile = this.files[0];
        if (currentFile) {
            if (currentFile.size > 10 * 1024 * 1024) {
                status.textContent = '❌ File too large (max 10MB)';
                uploadBtn.disabled = true;
                return;
            }
            if (!['audio/wav','audio/flac','audio/mpeg'].includes(currentFile.type)) {
                status.textContent = '❌ Unsupported format (WAV/FLAC/MP3 only)';
                uploadBtn.disabled = true;
                return;
            }
            status.textContent = `✅ Ready: ${currentFile.name}`;
            uploadBtn.disabled = false;
            uploadBtn.textContent = '🚀 Upload & Analyze';
            confirmBtn.style.display = 'none';
            result.textContent = '';
            timeDisplay.textContent = '00:00.0';
        }
    };

    // Upload & Analyze (START TIMER)
    uploadBtn.onclick = async function() {
        if (!currentFile) return;

        // 🕒 Start processing timer
        startTime = Date.now();
        analysisTimer = setInterval(() => {
            const elapsed = (Date.now() - startTime) / 1000;
            const minutes = Math.floor(elapsed / 60);
            const seconds = Math.floor((elapsed % 60) * 10) / 10;
            timeDisplay.textContent = `${minutes.toString().padStart(2,'0')}:${seconds.toString().padStart(5,'0')}`;
        }, 100);

        uploadBtn.disabled = true;
        status.textContent = '📤 Uploading...';
        result.textContent = '';
        progressBar.style.display = 'block';
        progressBar.value = 0;
        confirmBtn.style.display = 'none';

        const formData = new FormData();
        formData.append('file', currentFile);

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/upload');

        xhr.upload.onprogress = e => {
            if (e.lengthComputable) {
                progressBar.value = (e.loaded / e.total) * 50;
            }
        };

        xhr.onload = async () => {
            const res = JSON.parse(xhr.responseText);
            if (res.success) {
                status.textContent = '🔬 Analyzing...';
                progressBar.value = 50;

                // Analyze
                const analyzeRes = await fetch('/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({filename: res.filename})
                });
                const analyzeData = await analyzeRes.json();

                // 🛑 Stop timer + show result
                clearInterval(analysisTimer);
                const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
                timeDisplay.textContent = `Done: ${totalTime}s`;

                if (analyzeData.success) {
                    const output = JSON.parse(analyzeData.result);
                    const label = output.label ? '✅ REAL' : '❌ FAKE';
                    result.textContent = label;
                    result.className = output.label ? 'green' : 'red';
                    status.textContent = `🎉 Real: ${(output.prob_real*100).toFixed(1)}%`;
                } else {
                    status.textContent = '❌ ' + analyzeData.error;
                    result.textContent = '';
                }
            } else {
                status.textContent = '❌ ' + res.error;
                timeDisplay.textContent = 'Error';
                result.textContent = '';
            }
            
            progressBar.style.display = 'none';
            uploadBtn.disabled = true;  // Wait for Confirm
            confirmBtn.style.display = 'inline-block';
        };

        xhr.onerror = () => {
            clearInterval(analysisTimer);
            timeDisplay.textContent = 'Error';
            status.textContent = '❌ Network error';
            progressBar.style.display = 'none';
            confirmBtn.style.display = 'inline-block';
        };

        xhr.send(formData);
    };

    // 🔄 Confirm resets for next file
    confirmBtn.onclick = function() {
        currentFile = null;
        result.textContent = '';
        result.className = '';
        status.textContent = '';
        timeDisplay.textContent = '00:00.0';
        fileInput.value = '';
        uploadBtn.textContent = 'Select File First';
        uploadBtn.disabled = true;
        confirmBtn.style.display = 'none';
        progressBar.style.display = 'none';
    };
});

