let filesToProcess = [];
let activeUploads = 0;
let totalFiles = 0;
let analysisTimer = null;
let startTime = null;

document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const confirmBtn = document.getElementById('confirmBtn');
    const status = document.getElementById('status');
    const resultsContainer = document.getElementById('resultsContainer');
    const timer = document.getElementById('timer');
    const timeDisplay = document.getElementById('timeDisplay');

    // 🔥 Multi-file selection (max 10)
    fileInput.onchange = function() {
        const files = Array.from(this.files);
        
        // Filter valid files only (max 10)
        filesToProcess = files.slice(0, 10).filter(file => {
            const validTypes = ['audio/wav','audio/flac','audio/mpeg'];
            const isValid = validTypes.includes(file.type) && file.size <= 10 * 1024 * 1024;
            return isValid;
        });

        totalFiles = filesToProcess.length;
        
        if (totalFiles === 0) {
            status.textContent = '❌ No valid files (WAV/FLAC/MP3 only, max 10MB each)';
            uploadBtn.disabled = true;
            return;
        }

        status.textContent = `✅ ${totalFiles} valid file(s) ready (Max 10)`;
        uploadBtn.disabled = false;
        uploadBtn.textContent = `🚀 Analyze ${totalFiles} Files`;
        
        // Clear previous results
        resultsContainer.innerHTML = '';
        confirmBtn.style.display = 'none';
        timer.style.display = 'none';
    };

    // 🔥 Batch Upload & Analyze ALL files simultaneously
    uploadBtn.onclick = async function() {
        if (filesToProcess.length === 0) return;

        // Start global timer
        startTime = Date.now();
        analysisTimer = setInterval(() => {
            const elapsed = (Date.now() - startTime) / 1000;
            const minutes = Math.floor(elapsed / 60);
            const seconds = Math.floor((elapsed % 60) * 10) / 10;
            timeDisplay.textContent = `${minutes.toString().padStart(2,'0')}:${seconds.toString().padStart(5,'0')}`;
        }, 100);

        uploadBtn.disabled = true;
        status.textContent = `🚀 Processing ${totalFiles} files simultaneously...`;
        timer.style.display = 'block';
        confirmBtn.style.display = 'none';
        activeUploads = totalFiles;

        // 🔥 Process ALL files in parallel
        const processPromises = filesToProcess.map((file, index) => 
            processFile(file, index, resultsContainer)
        );

        // Wait for all to complete
        await Promise.all(processPromises);
        
        clearInterval(analysisTimer);
        const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
        status.textContent = `🎉 All ${totalFiles} files processed in ${totalTime}s!`;
        confirmBtn.style.display = 'inline-block';
        timer.style.display = 'none';
    };

    // 🔥 Process single file (upload + analyze)
    async function processFile(file, index, resultsContainer) {
        const fileId = `file-${Date.now()}-${index}`;
        addFileRow(resultsContainer, file.name, fileId, true);

        try {
            // 1. Upload
            const formData = new FormData();
            formData.append('file', file);

            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/upload');

            // Update individual progress
            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    const percent = Math.round((e.loaded / e.total) * 50);
                    updateFileProgress(fileId, percent, '📤 Uploading...');
                }
            };

            await new Promise((resolve, reject) => {
                xhr.onload = () => {
                    if (xhr.status === 200) {
                        const res = JSON.parse(xhr.responseText);
                        if (res.success) {
                            analyzeFile(res.filename, fileId, resultsContainer)
                                .then(resolve)
                                .catch(reject);
                        } else {
                            reject(new Error(res.error));
                        }
                    } else {
                        reject(new Error(`HTTP ${xhr.status}`));
                    }
                };
                xhr.onerror = reject;
                xhr.send(formData);
            });

        } catch (error) {
            console.error(`File ${file.name}:`, error);
            updateFileResult(fileId, '❌ ERROR', 'red', error.message);
        }
    }

    // 🔥 Analyze single file
    async function analyzeFile(filename, fileId, resultsContainer) {
        updateFileProgress(fileId, 50, '🔬 Analyzing...');
        
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({filename: filename})
        });
        
        const analyzeData = await response.json();
        
        if (analyzeData.success) {
            const output = JSON.parse(analyzeData.result);
            
            // 🔥 10% threshold logic
            const isReal = output.prob_real > 0.1;
            updateFileResult(
                fileId, 
                isReal ? '✅ REAL AUDIO' : '❌ FAKE', 
                isReal ? 'green' : 'red',
                filename
            );
        } else {
            updateFileResult(fileId, '❌ ANALYSIS FAILED', 'red', analyzeData.error);
        }
    }

    // 🔥 Add file result row
    function addFileRow(container, filename, fileId, uploading = false) {
        const div = document.createElement('div');
        div.id = fileId;
        div.className = 'file-result';
        div.innerHTML = `
            <div class="file-name">📁 ${filename}</div>
            <div class="progress-row">
                <progress id="${fileId}-progress" value="0" max="100"></progress>
                <div id="${fileId}-status">⏳ Preparing...</div>
            </div>
            <div id="${fileId}-result" style="display:none;"></div>
        `;
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }

    // 🔥 Update file progress/status
    function updateFileProgress(fileId, percent, message) {
        const progress = document.getElementById(`${fileId}-progress`);
        const statusEl = document.getElementById(`${fileId}-status`);
        if (progress) progress.value = percent;
        if (statusEl) statusEl.textContent = message;
    }

    // 🔥 Update final result + download button
    function updateFileResult(fileId, resultText, resultClass, filename) {
        const resultEl = document.getElementById(`${fileId}-result`);
        const statusEl = document.getElementById(`${fileId}-status`);
        const row = document.getElementById(fileId);
        
        resultEl.innerHTML = `
            <div class="result ${resultClass}">${resultText}</div>
            <button class="download-btn" onclick="window.downloadFile('${filename}')">
                💾 Download WAV
            </button>
        `;
        resultEl.style.display = 'block';
        statusEl.textContent = '✅ Complete!';
        row.className = `file-result ${resultClass}`;
        
        activeUploads--;
        if (activeUploads === 0) {
            status.textContent = '🎉 All files completed!';
        }
    }

    // 🔥 Global download function
    window.downloadFile = function(filename) {
        window.location.href = `/download/${filename}`;
    };

    // 🔥 Clear all results
    confirmBtn.onclick = function() {
        filesToProcess = [];
        resultsContainer.innerHTML = '';
        status.textContent = '✅ Cleared! Ready for new files.';
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Select Files First';
        confirmBtn.style.display = 'none';
        timer.style.display = 'none';
        fileInput.value = '';
    };
});

