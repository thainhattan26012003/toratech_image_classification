// Khi chạy sau Nginx (cùng domain), dùng relative path. Khi dev local (FE 3003) dùng BE 1234.
const API_BASE = (window.location.hostname === 'localhost' && window.location.port === '3003')
  ? 'http://localhost:1234'
  : '';

const modelArchitecture = document.getElementById('modelArchitecture');
const modelFile = document.getElementById('modelFile');
const classNames = document.getElementById('classNames');
const btnLoadModel = document.getElementById('btnLoadModel');
const modelStatus = document.getElementById('modelStatus');
const imageFiles = document.getElementById('imageFiles');
const btnPredict = document.getElementById('btnPredict');
const resultsEl = document.getElementById('results');

let modelLoaded = false;

function setModelStatus(message, type = 'info') {
  modelStatus.textContent = message;
  modelStatus.className = 'status ' + type;
}

async function loadModel() {
  const file = modelFile.files[0];
  if (!file) {
    setModelStatus('Vui lòng chọn file .pth', 'error');
    return;
  }
  const names = classNames.value.trim();
  if (!names) {
    setModelStatus('Vui lòng nhập tên các class', 'error');
    return;
  }
  setModelStatus('Đang tải model...', 'info');
  btnLoadModel.disabled = true;
  try {
    const form = new FormData();
    form.append('architecture', modelArchitecture.value);
    form.append('class_names', names);
    form.append('file', file);
    const res = await fetch(API_BASE + '/api/model', {
      method: 'POST',
      body: form,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setModelStatus(data.detail || 'Lỗi khi load model', 'error');
      return;
    }
    setModelStatus(data.message || 'Đã load model thành công!', 'success');
    modelLoaded = true;
    btnPredict.disabled = false;
  } catch (err) {
    setModelStatus('Lỗi kết nối backend: ' + err.message, 'error');
  } finally {
    btnLoadModel.disabled = false;
  }
}

async function predict() {
  const files = imageFiles.files;
  if (!files || files.length === 0) {
    resultsEl.innerHTML = '<p class="loading">Vui lòng chọn ít nhất một ảnh.</p>';
    return;
  }
  resultsEl.innerHTML = '<p class="loading">Đang phân loại...</p>';
  const form = new FormData();
  for (let i = 0; i < files.length; i++) {
    form.append('files', files[i]);
  }
  try {
    const res = await fetch(API_BASE + '/api/predict-batch', {
      method: 'POST',
      body: form,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      resultsEl.innerHTML = '<p class="error">' + (data.detail || 'Lỗi từ server') + '</p>';
      return;
    }
    renderResults(files, data.results || []);
  } catch (err) {
    resultsEl.innerHTML = '<p class="error">Lỗi kết nối backend: ' + err.message + '</p>';
  }
}

function renderResults(fileList, results) {
  const list = Array.from(fileList);
  resultsEl.innerHTML = '';
  list.forEach((file, i) => {
    const result = results[i] || {};
    const card = document.createElement('div');
    card.className = 'result-card';
    const imgUrl = URL.createObjectURL(file);
    let content = '<img src="' + imgUrl + '" alt=""/>';
    content += '<div class="filename">' + escapeHtml(file.name) + '</div>';
    if (result.error) {
      content += '<div class="error">' + escapeHtml(result.error) + '</div>';
    } else {
      content += '<div class="prediction">' + escapeHtml(result.prediction || '') + '</div>';
      content += '<div class="confidence">Confidence: ' + (result.confidence != null ? (result.confidence * 100).toFixed(2) + '%' : '') + '</div>';
    }
    card.innerHTML = content;
    card.querySelector('img').onload = () => URL.revokeObjectURL(imgUrl);
    resultsEl.appendChild(card);
  });
}

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

btnLoadModel.addEventListener('click', loadModel);
btnPredict.addEventListener('click', predict);
