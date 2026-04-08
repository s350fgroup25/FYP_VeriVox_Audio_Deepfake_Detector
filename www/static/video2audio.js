document.addEventListener('DOMContentLoaded', function() {
    const lines = ['line1', 'line2', 'line3', 'line4', 'line5'];
    const elements = {
        fileInput: document.getElementById('fileInput'),
        originalFileName: document.getElementById('originalFileName'),
        convertAnalyzeBtn: document.getElementById('convertAnalyzeBtn'),
        convertedPlayer: document.getElementById('convertedPlayer'),
        convertedFileName: document.getElementById('convertedFileName'),
        downloadBtn: document.getElementById('downloadBtn'),
        processStatus: document.getElementById('processStatus'),
        result: document.getElementById('result'),
        confidence: document.getElementById('confidence')
    };

    let currentFile = null;
    let convertedFilename = null;
    let aiResult = null;

    // Progressive UI - show next line
    function showNextLine(currentLine) {
        const currentIdx = lines.indexOf(currentLine);
        if (currentIdx < lines.length - 1) {
            document.getElementById(lines[currentIdx + 1]).style.display = 'block';
            console.log(`✅ Showed line: ${lines[currentIdx + 1]}`);
        }
    }

    // Line 1: File selected
    elements.fileInput.addEventListener('change', function(e) {
        if (e.target.files[0]) {
            currentFile = e.target.files[0];
            elements.originalFileName.textContent = `Selected: ${currentFile.name} (${(currentFile.size/1024/1024).toFixed(1)}MB)`;
            showNextLine('line1');  // → Line 2 (Convert+Analyze button)
        }
    });

    // Line 2: ONE CLICK Convert + Analyze - 🔥 NEW 10% THRESHOLD LOGIC!
    elements.convertAnalyzeBtn.addEventListener('click', async function() {
        if (!currentFile) {
            alert('Please select a video file first');
            return;
        }

        // Disable button and show processing
        elements.convertAnalyzeBtn.disabled = true;
        elements.convertAnalyzeBtn.textContent = '⏳ Processing...';
        elements.processStatus.textContent = '🎬 Step 1/2: Extracting audio from video...';

        try {
            console.log('🚀 Starting video conversion...');

            // Step 1: Convert video → WAV
            const formData = new FormData();
            formData.append('video', currentFile);
            formData.append('format', 'wav');

            const convertResponse = await fetch('/video2audio/convert_api', {
                method: 'POST',
                body: formData
            });

            if (!convertResponse.ok) {
                throw new Error(`Convert HTTP ${convertResponse.status}`);
            }

            const convertData = await convertResponse.json();
            console.log('Convert result:', convertData);

            if (!convertData.success) {
                throw new Error(convertData.error || 'Conversion failed');
            }

            convertedFilename = convertData.filename;
            console.log('✅ Converted filename:', convertedFilename);

            // Update UI for audio player
            elements.convertedFileName.textContent = `✅ ${convertedFilename}`;
            elements.convertedPlayer.src = `/download/${convertedFilename}`;

            // Show audio player FIRST
            showNextLine('line2');  // → Line 3 (audio player)
            elements.processStatus.textContent = '🔍 Step 2/2: AI Deepfake analysis...';

            console.log('🚀 Starting AI analysis...');

            // Step 2: Analyze with AI
            const analyzeResponse = await fetch('/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({filename: convertedFilename})
            });

            if (!analyzeResponse.ok) {
                throw new Error(`Analyze HTTP ${analyzeResponse.status}`);
            }

            const analyzeData = await analyzeResponse.json();
            console.log('Analyze result:', analyzeData);

            if (!analyzeData.success) {
                throw new Error(analyzeData.error || 'AI analysis failed');
            }

            // 🔥 NEW LOGIC: Real > 10% = REAL, else FAKE (NO % shown)
            aiResult = JSON.parse(analyzeData.result);
            console.log('AI Result:', aiResult);
            console.log(`🎯 Threshold check: Real=${aiResult.prob_real.toFixed(4)} > 0.1? ${aiResult.prob_real > 0.1}`);

            if (aiResult.prob_real > 0.1) {
                // Real > 10% → Show REAL
                elements.result.innerHTML = '<span style="color: #28a745; font-size: 32px;">✅ REAL AUDIO</span>';
                console.log('✅ UI: Showing REAL AUDIO');
            } else {
                // Real ≤ 10% → Show FAKE
                elements.result.innerHTML = '<span style="color: #dc3545; font-size: 32px;">❌ FAKE</span>';
                console.log('❌ UI: Showing FAKE');
            }

            // 隱藏百分比 - 乾淨顯示
            elements.confidence.innerHTML = '';

            // Show ALL remaining lines
            showNextLine('line3');  // → Line 4 (download)
            showNextLine('line4');  // → Line 5 (results)

            elements.processStatus.textContent = '🎉 All done! Audio ready + AI analyzed.';
            elements.convertAnalyzeBtn.textContent = '✅ Done!';
            elements.convertAnalyzeBtn.style.background = '#28a745';

        } catch (error) {
            console.error('❌ Full error:', error);
            elements.processStatus.innerHTML = `
                <div style="color: #dc3545;">
                    ❌ Error: ${error.message}<br>
                    <small>Check browser console (F12) for details</small>
                </div>
            `;
        } finally {
            elements.convertAnalyzeBtn.disabled = false;
            elements.convertAnalyzeBtn.textContent = '🎬 Convert + AI Analyze';
            elements.convertAnalyzeBtn.style.background = '#28a745';
        }
    });

    // Line 4: Download button
    elements.downloadBtn.addEventListener('click', function() {
        if (convertedFilename) {
            console.log('📥 Downloading:', convertedFilename);
            window.location.href = `/download/${convertedFilename}`;
        } else {
            alert('No file to download yet');
        }
    });

    // Debug: Log all elements found
    console.log('🎬 Video2Audio loaded. Elements:', elements);
});

