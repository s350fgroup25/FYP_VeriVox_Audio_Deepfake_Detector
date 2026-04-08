// 🔥 AI RESULT DISPLAY - 統一邏輯 (Real >10% = Real, else Fake)
window.showAIResult = function(elementId, resultData) {
    const result = JSON.parse(resultData.result);
    
    // 閾值設定 (隨時可改)
    const REAL_THRESHOLD = 0.1;  // 10% - 這裡修改即可！
    
    const resultElement = document.getElementById(elementId);
    
    if (result.prob_real > REAL_THRESHOLD) {
        resultElement.innerHTML = '<span style="color: #28a745; font-size: 32px;">✅ REAL AUDIO</span>';
    } else {
        resultElement.innerHTML = '<span style="color: #dc3545; font-size: 32px;">❌ FAKE</span>';
    }
    
    // 隱藏信心度 (乾淨顯示)
    const confidenceElement = document.getElementById('confidence');
    if (confidenceElement) {
        confidenceElement.innerHTML = '';
    }
    
    console.log(`🎯 AI Result: Real=${result.prob_real.toFixed(4)}, Fake=${result.prob_fake.toFixed(4)}, Label=${result.prob_real > REAL_THRESHOLD ? 'REAL' : 'FAKE'}`);
};

