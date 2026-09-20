/**
 * DigitNet — Handwritten Digit Recognition Frontend
 * Vanilla JS logic for drawing canvas, file uploads, MNIST 28x28 preview,
 * mock predictions, dynamic class probability rendering, and backend bridge.
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const tabDraw = document.getElementById('tab-draw');
    const tabUpload = document.getElementById('tab-upload');
    const drawPanel = document.getElementById('draw-panel');
    const uploadPanel = document.getElementById('upload-panel');

    const canvas = document.getElementById('drawing-canvas');
    const ctx = canvas.getContext('2d');
    const canvasHint = document.getElementById('canvas-hint');

    const btnClear = document.getElementById('btn-clear');
    const btnUndo = document.getElementById('btn-undo');
    const btnPredict = document.getElementById('btn-predict');

    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const dropzoneEmpty = document.getElementById('dropzone-empty');
    const dropzonePreview = document.getElementById('dropzone-preview');
    const previewImg = document.getElementById('preview-img');
    const btnRemoveUpload = document.getElementById('btn-remove-upload');

    const mnistPreviewCanvas = document.getElementById('mnist-preview-canvas');
    const mnistCtx = mnistPreviewCanvas.getContext('2d');

    const predictedDigitEl = document.getElementById('predicted-digit');
    const predictedConfidenceEl = document.getElementById('predicted-confidence');
    const probabilityChartEl = document.getElementById('probability-chart');
    const exampleGridEl = document.getElementById('example-grid');
    const toastEl = document.getElementById('toast');
    const toastMessageEl = document.getElementById('toast-message');

    // --- State Variables ---
    let activeMode = 'draw'; // 'draw' or 'upload'
    let isDrawing = false;
    let hasDrawn = false;
    let undoStack = [];
    const maxUndoSteps = 25;
    let uploadedImageElement = null;
    let toastTimeout = null;

    // --- 1. Mode Tab Switching ---
    function setActiveMode(mode) {
        activeMode = mode;
        if (mode === 'draw') {
            tabDraw.classList.add('active');
            tabDraw.setAttribute('aria-selected', 'true');
            tabUpload.classList.remove('active');
            tabUpload.setAttribute('aria-selected', 'false');
            drawPanel.classList.add('active');
            uploadPanel.classList.remove('active');
        } else {
            tabUpload.classList.add('active');
            tabUpload.setAttribute('aria-selected', 'true');
            tabDraw.classList.remove('active');
            tabDraw.setAttribute('aria-selected', 'false');
            uploadPanel.classList.add('active');
            drawPanel.classList.remove('active');
        }
    }

    tabDraw.addEventListener('click', () => setActiveMode('draw'));
    tabUpload.addEventListener('click', () => setActiveMode('upload'));

    // --- 2. Drawing Canvas Engine ---
    function initCanvas() {
        const rect = canvas.getBoundingClientRect();
        // Scaling for crisp DPI displays
        const dpi = window.devicePixelRatio || 1;
        canvas.width = rect.width * dpi;
        canvas.height = rect.height * dpi;
        ctx.scale(dpi, dpi);

        // Canvas default background (White)
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, rect.width, rect.height);

        // Stroke settings for digit drawing
        ctx.strokeStyle = '#111827';
        ctx.lineWidth = 20;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        saveCanvasState();
    }

    function saveCanvasState() {
        if (undoStack.length >= maxUndoSteps) {
            undoStack.shift();
        }
        undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    }

    function getCanvasCoordinates(e) {
        const rect = canvas.getBoundingClientRect();
        let clientX = e.clientX;
        let clientY = e.clientY;

        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        }

        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }

    function startDrawing(e) {
        e.preventDefault();
        isDrawing = true;
        hasDrawn = true;
        canvasHint.classList.add('hidden');
        saveCanvasState();

        const coords = getCanvasCoordinates(e);
        ctx.beginPath();
        ctx.moveTo(coords.x, coords.y);
        // Draw a tiny dot for clicks without drag
        ctx.lineTo(coords.x + 0.1, coords.y + 0.1);
        ctx.stroke();
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();

        const coords = getCanvasCoordinates(e);
        ctx.lineTo(coords.x, coords.y);
        ctx.stroke();
    }

    function stopDrawing(e) {
        if (!isDrawing) return;
        isDrawing = false;
        ctx.closePath();
    }

    // Canvas Event Listeners (Mouse, Touch, Pointer)
    canvas.addEventListener('pointerdown', startDrawing);
    canvas.addEventListener('pointermove', draw);
    canvas.addEventListener('pointerup', stopDrawing);
    canvas.addEventListener('pointerleave', stopDrawing);
    canvas.addEventListener('pointercancel', stopDrawing);

    // Clear Canvas
    function clearCanvas() {
        const rect = canvas.getBoundingClientRect();
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, rect.width, rect.height);
        hasDrawn = false;
        canvasHint.classList.remove('hidden');
        undoStack = [];
        saveCanvasState();
    }

    btnClear.addEventListener('click', () => {
        if (activeMode === 'draw') {
            clearCanvas();
        } else {
            resetUpload();
        }
    });

    // Undo Canvas
    btnUndo.addEventListener('click', () => {
        if (activeMode !== 'draw' || undoStack.length === 0) return;

        // Pop current state
        undoStack.pop();
        if (undoStack.length > 0) {
            const previousState = undoStack[undoStack.length - 1];
            ctx.putImageData(previousState, 0, 0);
        } else {
            clearCanvas();
        }
    });

    // --- 3. Image Upload Engine ---
    dropzone.addEventListener('click', (e) => {
        if (e.target !== btnRemoveUpload) {
            fileInput.click();
        }
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileUpload(e.target.files[0]);
        }
    });

    function handleFileUpload(file) {
        if (!file.type.match('image.*')) {
            showToast('Please upload a valid PNG, JPG, or JPEG image file.');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            dropzoneEmpty.classList.add('hidden');
            dropzonePreview.classList.remove('hidden');

            const img = new Image();
            img.onload = () => {
                uploadedImageElement = img;
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    function resetUpload() {
        fileInput.value = '';
        previewImg.src = '';
        uploadedImageElement = null;
        dropzonePreview.classList.add('hidden');
        dropzoneEmpty.classList.remove('hidden');
    }

    btnRemoveUpload.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    // --- 4. MNIST 28x28 Preprocessing Preview ---
    function updateMNISTPreview(sourceCanvasOrImage) {
        // Clear 28x28 black preview canvas
        mnistCtx.fillStyle = '#000000';
        mnistCtx.fillRect(0, 0, 28, 28);

        if (!sourceCanvasOrImage) return;

        // Offscreen canvas for 28x28 resize
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');

        // Draw resized source
        tempCtx.drawImage(sourceCanvasOrImage, 0, 0, 28, 28);
        const imgData = tempCtx.getImageData(0, 0, 28, 28);
        const data = imgData.data;

        // Standard MNIST: Black background (#000000), White stroke (#ffffff)
        // If drawn canvas (black on white), invert pixel values
        for (let i = 0; i < data.length; i += 4) {
            const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
            // If background is light (white stroke or white background), invert grayscale
            const inverted = 255 - avg;
            data[i] = inverted;     // Red
            data[i + 1] = inverted; // Green
            data[i + 2] = inverted; // Blue
            data[i + 3] = 255;      // Alpha
        }

        tempCtx.putImageData(imgData, 0, 0);
        mnistCtx.drawImage(tempCanvas, 0, 0);
    }

    // --- 5. Class Probabilities Visualizer ---
    function renderProbabilityChart(probabilities, targetDigit) {
        probabilityChartEl.innerHTML = '';

        probabilities.forEach((prob, digit) => {
            const percentage = (prob * 100).toFixed(1);
            const isHighlighted = digit === targetDigit;

            const barCol = document.createElement('div');
            barCol.className = `chart-bar-col ${isHighlighted ? 'highlighted' : ''}`;

            const barPercentage = document.createElement('span');
            barPercentage.className = 'bar-percentage';
            barPercentage.textContent = `${percentage}%`;

            const barFillContainer = document.createElement('div');
            barFillContainer.className = 'bar-fill-container';

            const barFill = document.createElement('div');
            barFill.className = 'bar-fill';
            // Scale bar height smoothly (min height 4% for visibility)
            const heightVal = Math.max(4, prob * 100);
            barFill.style.height = `${heightVal}%`;

            const barLabel = document.createElement('span');
            barLabel.className = 'bar-label';
            barLabel.textContent = digit;

            barFillContainer.appendChild(barFill);
            barCol.appendChild(barPercentage);
            barCol.appendChild(barFillContainer);
            barCol.appendChild(barLabel);

            probabilityChartEl.appendChild(barCol);
        });
    }

    // --- 6. Prediction Engine & Backend Bridge ---
    const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'http://localhost:8000'
        : '';

    /**
     * Sends the drawn canvas or uploaded image to the FastAPI /predict endpoint.
     * Returns real prediction, confidence, probabilities, and preprocessed preview.
     */
    async function predictDigit(inputSource) {
        let blob = null;

        if (inputSource instanceof HTMLCanvasElement) {
            blob = await new Promise((resolve) => inputSource.toBlob(resolve, 'image/png'));
        } else if (inputSource instanceof File || inputSource instanceof Blob) {
            blob = inputSource;
        }

        if (!blob) {
            throw new Error('Please draw a digit or upload an image first.');
        }

        const formData = new FormData();
        formData.append('file', blob, 'digit.png');

        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            let errorDetail = 'Prediction request failed';
            try {
                const errData = await response.json();
                if (errData && errData.detail) {
                    errorDetail = errData.detail;
                }
            } catch (_) {}
            throw new Error(errorDetail);
        }

        return await response.json();
    }

    // Predict Action Handler
    btnPredict.addEventListener('click', async () => {
        let inputSource = null;

        if (activeMode === 'draw') {
            if (!hasDrawn) {
                showToast('Please draw a digit or upload an image first.');
                return;
            }
            inputSource = canvas;
        } else {
            if (!uploadedImageElement && (!fileInput.files || fileInput.files.length === 0)) {
                showToast('Please draw a digit or upload an image first.');
                return;
            }
            inputSource = (fileInput.files && fileInput.files[0]) || uploadedImageElement;
        }

        // 1. Enter Loading State: disable button and show Analyzing...
        const originalBtnHTML = btnPredict.innerHTML;
        btnPredict.disabled = true;
        btnPredict.innerHTML = `
            <svg class="spinner" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="2" x2="12" y2="6"></line><line x1="12" y1="18" x2="12" y2="22"></line><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line><line x1="2" y1="12" x2="6" y2="12"></line><line x1="18" y1="12" x2="22" y2="12"></line><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line></svg>
            Analyzing...
        `;

        try {
            // 2. Fetch Real ML Model Prediction
            const result = await predictDigit(inputSource);

            // 3. Update 28x28 MNIST Preview Box (using server-preprocessed preview if available)
            if (result.preview) {
                const previewImg = new Image();
                previewImg.onload = () => {
                    mnistCtx.clearRect(0, 0, 28, 28);
                    mnistCtx.drawImage(previewImg, 0, 0, 28, 28);
                };
                previewImg.src = result.preview;
            } else {
                updateMNISTPreview(inputSource);
            }

            // 4. Update Prediction Result Metrics
            const digit = result.prediction !== undefined ? result.prediction : result.digit;
            predictedDigitEl.textContent = digit;
            predictedConfidenceEl.textContent = `${(result.confidence * 100).toFixed(2)}%`;

            // 5. Render Class Probabilities Chart
            renderProbabilityChart(result.probabilities, digit);
        } catch (error) {
            console.error('Prediction failed:', error);
            showToast(error.message || 'Unable to connect to backend server.');
        } finally {
            // Restore button state
            btnPredict.disabled = false;
            btnPredict.innerHTML = originalBtnHTML;
        }
    });

    // --- 7. Example Digits Tiles (0-9) ---
    function renderExampleDigitTiles() {
        exampleGridEl.innerHTML = '';
        for (let i = 0; i <= 9; i++) {
            const tile = document.createElement('div');
            tile.className = 'example-tile';
            tile.textContent = i;
            tile.setAttribute('title', `Try digit ${i}`);

            tile.addEventListener('click', async () => {
                // Switch to Draw tab
                setActiveMode('draw');
                // Clear existing canvas
                clearCanvas();
                // Draw template digit onto canvas
                drawExampleDigitOnCanvas(i);
                // Trigger real prediction
                btnPredict.click();
            });

            exampleGridEl.appendChild(tile);
        }
    }


    // Helper: Draw template digit curves on canvas
    function drawExampleDigitOnCanvas(digit) {
        hasDrawn = true;
        canvasHint.classList.add('hidden');

        const rect = canvas.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;
        const cx = w / 2;
        const cy = h / 2;

        ctx.strokeStyle = '#111827';
        ctx.lineWidth = 22;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();

        // Simple vector strokes for template digits
        switch(digit) {
            case 0:
                ctx.ellipse(cx, cy, w * 0.22, h * 0.32, 0, 0, 2 * Math.PI);
                break;
            case 1:
                ctx.moveTo(cx - 20, cy - h * 0.25);
                ctx.lineTo(cx, cy - h * 0.3);
                ctx.lineTo(cx, cy + h * 0.3);
                break;
            case 2:
                ctx.arc(cx - 5, cy - h * 0.15, w * 0.18, Math.PI, 0.2);
                ctx.lineTo(cx - w * 0.2, cy + h * 0.28);
                ctx.lineTo(cx + w * 0.2, cy + h * 0.28);
                break;
            case 3:
                ctx.arc(cx - 5, cy - h * 0.14, w * 0.16, -Math.PI * 0.8, Math.PI * 0.5);
                ctx.arc(cx - 5, cy + h * 0.14, w * 0.17, -Math.PI * 0.5, Math.PI * 0.8);
                break;
            case 4:
                ctx.moveTo(cx + w * 0.1, cy + h * 0.28);
                ctx.lineTo(cx + w * 0.1, cy - h * 0.3);
                ctx.lineTo(cx - w * 0.22, cy + h * 0.05);
                ctx.lineTo(cx + w * 0.22, cy + h * 0.05);
                break;
            case 5:
                ctx.moveTo(cx + w * 0.18, cy - h * 0.28);
                ctx.lineTo(cx - w * 0.15, cy - h * 0.28);
                ctx.lineTo(cx - w * 0.15, cy - h * 0.02);
                ctx.arc(cx - 2, cy + h * 0.1, w * 0.18, -Math.PI * 0.5, Math.PI * 0.7);
                break;
            case 6:
                ctx.arc(cx, cy + h * 0.1, w * 0.18, 0, Math.PI * 2);
                ctx.moveTo(cx + w * 0.18, cy + h * 0.08);
                ctx.bezierCurveTo(cx + w * 0.1, cy - h * 0.2, cx, cy - h * 0.3, cx - w * 0.08, cy - h * 0.25);
                break;
            case 7:
                ctx.moveTo(cx - w * 0.22, cy - h * 0.28);
                ctx.lineTo(cx + w * 0.22, cy - h * 0.28);
                ctx.lineTo(cx - w * 0.08, cy + h * 0.3);
                break;
            case 8:
                ctx.ellipse(cx, cy - h * 0.14, w * 0.16, h * 0.14, 0, 0, 2 * Math.PI);
                ctx.ellipse(cx, cy + h * 0.14, w * 0.19, h * 0.16, 0, 0, 2 * Math.PI);
                break;
            case 9:
                ctx.arc(cx, cy - h * 0.1, w * 0.18, 0, Math.PI * 2);
                ctx.moveTo(cx + w * 0.18, cy - h * 0.1);
                ctx.lineTo(cx + w * 0.08, cy + h * 0.3);
                break;
        }

        ctx.stroke();
        saveCanvasState();
    }

    // --- 8. Toast Alert Component ---
    function showToast(message) {
        toastMessageEl.textContent = message;
        toastEl.classList.add('show');

        if (toastTimeout) clearTimeout(toastTimeout);
        toastTimeout = setTimeout(() => {
            toastEl.classList.remove('show');
        }, 3200);
    }

    // --- 9. Backend Health Status Check ---
    async function checkBackendHealth() {
        const statusBadge = document.querySelector('.status-badge');
        try {
            const res = await fetch(`${API_BASE_URL}/health`);
            const data = await res.json();
            if (data.status === 'healthy' && statusBadge) {
                statusBadge.className = 'status-badge green-badge';
                statusBadge.innerHTML = '<span class="status-dot"></span> Model Loaded';
            }
        } catch (_) {
            if (statusBadge) {
                statusBadge.className = 'status-badge gray-badge';
                statusBadge.innerHTML = '<span class="status-dot offline"></span> Backend Offline';
            }
        }
    }

    // --- Initial Setup ---
    initCanvas();
    renderExampleDigitTiles();
    checkBackendHealth();

    // Render default state on page load
    const initialProbabilities = [0.000, 0.000, 0.001, 0.000, 0.000, 0.001, 0.002, 0.991, 0.001, 0.000];
    renderProbabilityChart(initialProbabilities, 7);
    
    // Draw initial digit 7 on preview canvas to match reference mockup
    drawExampleDigitOnCanvas(7);
    updateMNISTPreview(canvas);
});
