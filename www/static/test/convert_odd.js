document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const originalFileName = document.getElementById('originalFileName');
    const originalFileNameDisplay = document.getElementById('originalFileNameDisplay');
    const fileInfo = document.getElementById('fileInfo');
    const originalPlayer = document.getElementById('originalPlayer');
    const convertedPlayer = document.getElementById('convertedPlayer');
    const convertedFileName = document.getElementById('convertedFileName');
    const convertBtn = document.getElementById('convertBtn');
    const convertStatus = document.getElementById('convertStatus');
    const downloadBtn = document.getElementById('downloadBtn');
    const submitBtn = document.getElementById('submitBtn');
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

        // Reset converted
        convertedFilename = null;
        convertedPlayer.src = '';
        convertedFileName.textContent = '';
        document.getElementById('line5').style.display = 'none';
        document.getElementById('line6').style.display = 'none';
        document.getElementById('line7').style.display = 'none';
        result.textContent = '';
    };

    // Line 4: Convert
    convertBtn.onclick = async function() {
        if (!originalFile) return alert('Please select a file first');

        convertBtn.disabled = true;
        convertStatus.textContent = '⏳ Converting to 16kHz WAV...';
        document.getElementById('line4').scrollIntoView();

        try {
            const formData = new FormData();
            formData.append('file', originalFile);

            const response = await fetch('/convert', { method: 'POST', body: formData });
            const res = await response.json();
            
            if (res.success) {
                convertedFilename = res.filename;
                convertedFileName.textContent = `✅ ${res.filename} (${(res.size/1024).toFixed(1)}KB)`;
                
                // Play converted WAV
                convertedPlayer.src = `/download/${res.filename}`;
                document.getElementById('line5').style.display = 'block';
                document.getElementById('line6').style.display = 'block';
                
                convertStatus.textContent = '✅ Converted successfully! Now play both audios.';
                document.getElementById('line3').scrollIntoView();
            } else {
                convertStatus.textContent = `❌ Error: ${res.error}`;
            }
        } catch (err) {
            convertStatus.textContent = `❌ Network error: ${err.message}`;
        } finally {
            convertBtn.disabled = false;
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

    // Line 6: Submit to AI
    submitBtn.onclick = async function() {
        if (!convertedFilename) return alert('Please convert file first');

        submitBtn.disabled = true;
        submitStatus.textContent = '📤 Submitting to AI Model...';
        document.getElementById('line7').style.display = 'block';

        try {
            const analyzeRes = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: convertedFilename })
            });
            
            const analyzeData = await analyzeRes.json();
            
            if (analyzeData.success) {
                const output = JSON.parse(analyzeData.result);
                const label = output.label ? '✅ REAL' : '❌ FAKE';
                result.textContent = label;
                result.className = output.label ? 'green' : 'red';
                confidence.textContent = `Real: ${(output.prob_real * 100).toFixed(1)}% | Fake: ${(output.prob_fake * 100).toFixed(1)}%`;
                submitStatus.textContent = '🎉 Analysis complete!';
            } else {
                submitStatus.textContent = `❌ Analysis failed: ${analyzeData.error}`;
            }
        } catch (err) {
            submitStatus.textContent = `❌ Error: ${err.message}`;
        } finally {
            submitBtn.disabled = false;
        }
    };
});

