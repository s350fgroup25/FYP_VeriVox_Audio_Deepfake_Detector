let uploadedFile = null, uploadedFilename = null;

document.getElementById('uploadBtn').onclick = function() {
    const f = document.getElementById('fileInput').files[0];
    if (!f) return alert('請選擇檔案');
    if (f.size > 10*1024*1024) return alert('檔案過大');
    if (!['audio/flac','audio/wav','audio/mp3'].includes(f.type)) return alert('格式錯誤');
    const fd = new FormData();
    fd.append('file', f);
    fetch('/upload', {method:'POST', body: fd})
    .then(r=>r.json()).then(data=>{
        if(data.success) {
            uploadedFile = f;
            uploadedFilename = data.filename;
            document.getElementById('uploadMsg').textContent = data.filename + ' successful';
        } else {
            alert('上載失敗: ' + data.error);
        }
    });
};

document.getElementById('cancelBtn').onclick = function() {
    uploadedFile = null; uploadedFilename = null;
    document.getElementById('uploadMsg').textContent = '';
    document.getElementById('fileInput').value = '';
};

document.getElementById('submitBtn').onclick = function() {
    if (!uploadedFilename) return alert('請先成功上載音檔！');
    document.getElementById('progressBox').style.display = '';
    let pct = 0, bar = document.getElementById('progressBar');
    bar.value = 0;
    let ti = setInterval(()=>{ pct+=10; bar.value=pct;
        if (pct>=100) {
            clearInterval(ti);
            document.getElementById('confirmBtn').style.display = '';
        }
    }, 200);
};

document.getElementById('confirmBtn').onclick = function() {
    fetch('/analyze', {method:'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({filename: uploadedFilename})})
    .then(r=>r.json()).then(data=>{
        let r = document.getElementById('result');
        r.textContent = data.result ? 'Real' : 'Fake';
        r.className = data.result ? 'green' : 'red';
    });
};

document.getElementById('resetBtn').onclick = function() {
    location.reload();
};

