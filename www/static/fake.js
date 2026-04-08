document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('textInput');
    const charCount = document.getElementById('charCount');
    const generateBtn = document.getElementById('generateBtn');
    const status = document.getElementById('status');
    const audioPlayer = document.getElementById('audioPlayer');
    const downloadBtn = document.getElementById('downloadBtn');
    const submitBtn = document.getElementById('submitBtn');
    const submitStatus = document.getElementById('submitStatus');
    const result = document.getElementById('result');
    const confidence = document.getElementById('confidence');

    let generatedFilename = null;

    // Character counter
    textInput.addEventListener('input', () => {
        const len = textInput.value.length;
        charCount.textContent = `${len} / 1000`;
    });

    // Line 1: Generate fake audio via TTS
    generateBtn.onclick = async () => {
        const text = textInput.value.trim();
        if (!text) {
            alert('Please enter some text first.');
            return;
        }

        generatedFilename = null;
        audioPlayer.src = '';
        document.getElementById('line2').style.display = 'none';
        document.getElementById('line3').style.display = 'none';
        document.getElementById('line4').style.display = 'none';
        result.textContent = '';
        confidence.textContent = '';

        generateBtn.disabled = true;
        status.textContent = '⏳ Generating fake speech (TTS → WAV 16kHz)...';

        try {
            const resp = await fetch('/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const data = await resp.json();

            if (!data.success) {
                status.textContent = `❌ TTS error: ${data.error}`;
                generateBtn.disabled = false;
                return;
            }

            generatedFilename = data.filename;  // e.g. fake_1700000000.wav
            status.textContent = `✅ Generated: ${generatedFilename}`;

            // Show player and buttons
            audioPlayer.src = `/download/${generatedFilename}`;
            document.getElementById('line2').style.display = 'block';
            document.getElementById('line3').style.display = 'block';
            document.getElementById('line4').style.display = 'block';

        } catch (err) {
            status.textContent = `❌ Network error: ${err.message}`;
        } finally {
            generateBtn.disabled = false;
        }
    };

    // Line 3: Download WAV
    downloadBtn.onclick = () => {
        if (!generatedFilename) {
            alert('No generated audio to download.');
            return;
        }
        const a = document.createElement('a');
        a.href = `/download/${generatedFilename}`;
        a.download = generatedFilename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    };

    // Line 4: Submit to existing /analyze
    submitBtn.onclick = async () => {
        if (!generatedFilename) {
            alert('Please generate fake audio first.');
            return;
        }

        submitBtn.disabled = true;
        submitStatus.textContent = '📤 Submitting to ASVspoof model...';
        result.textContent = '';
        confidence.textContent = '';

        try {
            const resp = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: generatedFilename })
            });
            const data = await resp.json();

            if (!data.success) {
                submitStatus.textContent = `❌ Analysis failed: ${data.error}`;
                submitBtn.disabled = false;
                return;
            }

            const output = JSON.parse(data.result);
            const labelText = output.label ? '✅ REAL' : '❌ FAKE';
            result.textContent = labelText;
            result.className = output.label ? 'green' : 'red';
            confidence.textContent = `Real: ${(output.prob_real * 100).toFixed(1)}% | Fake: ${(output.prob_fake * 100).toFixed(1)}%`;
            submitStatus.textContent = '🎉 Analysis complete!';
        } catch (err) {
            submitStatus.textContent = `❌ Network error: ${err.message}`;
        } finally {
            submitBtn.disabled = false;
        }
    };
});

