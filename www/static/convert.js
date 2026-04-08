document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const originalFileName = document.getElementById('originalFileName');
    const originalFileNameDisplay = document.getElementById('originalFileNameDisplay');
    const fileInfo = document.getElementById('fileInfo');
    const originalPlayer = document.getElementById('originalPlayer');
    const convertedPlayer = document.getElementById('convertedPlayer');
    const convertedFileName = document.getElementById('convertedFileName');
    const convertBtn = document.getElementById('convertBtn');  // 🔥 Convert + AI
    const convertStatus = document.getElementById('convertStatus');
    const downloadBtn = document.getElementById('downloadBtn');
    const submitBtn = document.getElementById('submitBtn');    // 隱藏
    const submitStatus = document.getElementById('submitStatus');
    const result = document.getElementById('result');
    const confidence = document.getElementById('confidence');

    let originalFile = null;
    let convertedFilename = null;

    // Line 1: File select
    fileInput.onchange = function(e) {
        originalFile = e.target.files[0];
        if (!originalFile) return;

        const fileSizeKB = (originalFile.size / 1024).toFixed(1);
        originalFileName.textContent = `Selected: ${originalFile.name} (${fileSizeKB}KB)`;
        originalFileNameDisplay.textContent = originalFile.name;
        fileInfo.textContent = `${originalFile.name} - ${fileSizeKB}KB`;

        // Show Line 2 & 3
        document.getElementById('line2').style.display = 'block';
        document.getElementById('line3').style.display = 'block';
        document.getElementById('line4').style.display = 'block';

        // Play original
        const url = URL.createObjectURL(originalFile);
        originalPlayer.src = url;

        // Reset converted & AI result
        convertedFilename = null;
        convertedPlayer.src = '';
        convertedFileName.textContent = '';
        result.textContent = '';
        result.className = '';
        confidence.textContent = '';
        document.getElementById('line5').style.display = 'none';
        document.getElementById('line6').style.display = 'block';  // Download永遠顯示
        document.getElementById('line7').style.display = 'none';
    };

    // 🔥 Line 4: ONE-CLICK Convert + AI Analyze (50% threshold for convert page)
    convertBtn.onclick = async function() {
        if (!originalFile) return alert('Please select a file first');

        convertBtn.disabled = true;
        convertBtn.textContent = '⏳ Convert + Analyze...';
        convertStatus.textContent = '🎵 Step 1/2: Converting to 16kHz WAV...';
        document.getElementById('line4').scrollIntoView();

        try {
            // Step 1: Convert to WAV
            const formData = new FormData();
            formData.append('file', originalFile);

            const response = await fetch('/convert', { method: 'POST', body: formData });
            const res = await response.json();

            if (!res.success) {
                throw new Error(res.error);
            }

            convertedFilename = res.filename;
            convertedFileName.textContent = `✅ ${res.filename} (${(res.size/1024).toFixed(1)}KB)`;
            convertedPlayer.src = `/download/${res.filename}`;
            
            convertStatus.textContent = '🔍 Step 2/2: AI Analysis...';
            document.getElementById('line5').style.display = 'block';  // Show audio player

            // Step 2: AI Analyze (自動執行!)
            const analyzeRes = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: convertedFilename })
            });

            const analyzeData = await analyzeRes.json();

            if (analyzeData.success) {
                const output = JSON.parse(analyzeData.result);
                
                console.log(`🎯 Convert: Real=${output.prob_real.toFixed(4)}, Fake=${output.prob_fake.toFixed(4)}`);
                
                // 🔥 CONVERT PAGE: 50% threshold (不同於其他頁面的10%)
                if (output.prob_real > 0.5) {
                    result.textContent = '✅ REAL AUDIO';
                    result.className = 'green';
                    console.log('✅ UI: Showing REAL AUDIO (50% threshold)');
                } else {
                    result.textContent = '❌ FAKE';
                    result.className = 'red';
                    console.log('❌ UI: Showing FAKE (50% threshold)');
                }
                
                // 🔥 Convert page 保持 % 顯示 (特色保留!)
                confidence.textContent = `Real: ${(output.prob_real * 100).toFixed(1)}% | Fake: ${(output.prob_fake * 100).toFixed(1)}%`;
                
                document.getElementById('line7').style.display = 'block';
                convertStatus.textContent = '🎉 Convert + Analyze complete!';
                document.getElementById('line3').scrollIntoView();
            } else {
                convertStatus.textContent = `❌ Analysis failed: ${analyzeData.error}`;
            }

        } catch (err) {
            console.error('Convert+Analyze error:', err);
            convertStatus.textContent = `❌ Error: ${err.message}`;
        } finally {
            convertBtn.disabled = false;
            convertBtn.textContent = '🎵 Convert + AI Analyze';
        }
    };

    // Line 5: Download converted WAV
    downloadBtn.onclick = function() {
        if (convertedFilename) {
            const a = document.createElement('a');
            a.href = `/download/${convertedFilename}`;
            a.download = convertedFilename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            console.log('💾 Downloaded:', convertedFilename);
        }
    };

    // Line 6: SubmitBtn 永遠隱藏 (已整合到 convertBtn)
    submitBtn.style.display = 'none';
});

