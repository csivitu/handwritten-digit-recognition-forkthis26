// Run with: node test_upload_independence.js
async function testUploadIndependence(source) {
    const elements = new Map();
    const reads = [];
    const requests = [];
    const context = new Proxy({}, {
        get: (_, key) => key === 'getImageData' ? () => ({ data: [] }) : () => {},
    });
    function element() {
        return {
            listeners: {}, style: {}, files: [], children: [],
            classList: { add() {}, remove() {} },
            addEventListener(event, handler) { this.listeners[event] = handler; },
            setAttribute() {}, appendChild(child) { this.children.push(child); },
            getContext() { return context; },
            getBoundingClientRect() { return { width: 300, height: 300 }; },
        };
    }
    let initialize;
    const document = {
        addEventListener(event, handler) { initialize = handler; },
        getElementById(id) {
            if (!elements.has(id)) elements.set(id, element());
            return elements.get(id);
        },
        createElement: element,
        querySelector: () => null,
    };
    class File {
        constructor(name) { this.name = name; this.type = 'image/png'; }
    }
    class FileReader {
        readAsDataURL(file) { reads.push({ file, reader: this }); }
    }
    class FormData {
        append(key, value) { this[key] = value; }
    }
    const fetch = async (url, options) => {
        if (options) requests.push(options.body.file);
        return { ok: true, json: async () => ({ prediction: 3, confidence: 1,
            probabilities: Array(10).fill(0), preview: 'preview' }) };
    };
    new Function('document', 'window', 'FileReader', 'File', 'Blob',
        'HTMLCanvasElement', 'FormData', 'fetch', 'Image', 'setTimeout',
        'clearTimeout', source)(document, { location: { hostname: 'localhost' } },
        FileReader, File, class {}, class {}, FormData, fetch, class {}, () => 0, () => {});
    initialize();

    const get = (id) => document.getElementById(id);
    const assert = (condition, message) => { if (!condition) throw new Error(message); };
    const picker = get('file-input');
    const pick = (file) => {
        picker.files = [file];
        picker.listeners.change({ target: picker });
    };
    const drop = (file) => get('dropzone').listeners.drop({
        preventDefault() {}, dataTransfer: { files: [file] },
    });
    const predict = () => get('btn-predict').listeners.click();
    const finishRead = (index) => reads[index].reader.onload({ target: { result: reads[index].file.name } });

    get('tab-upload').listeners.click();
    const first = new File('first.png');
    const second = new File('second.png');
    pick(first);
    finishRead(0);
    await predict();
    drop(second);
    // Predict immediately, before asynchronous preview loading finishes.
    await predict();
    assert(requests[0] === first && requests[1] === second,
        'A dropped file must replace a previous picker file in predictions');
    finishRead(1);
    finishRead(0);
    assert(get('preview-img').src === second.name, 'An old read must not overwrite the new preview');

    get('btn-remove-upload').listeners.click({ stopPropagation() {} });
    finishRead(1);
    await predict();
    assert(requests.length === 2, 'Reset must prevent reuse of any previous upload');
    assert(get('preview-img').src === '', 'An old read must not restore a cleared preview');

    pick(first);
    await predict();
    assert(requests[2] === first, 'A new picker selection must replace the dropped file');
    return 'Upload independence regressions passed';
}

if (typeof require !== 'undefined') {
    testUploadIndependence(require('node:fs').readFileSync('frontend/script.js', 'utf8'))
        .then(console.log).catch(error => { console.error(error); process.exitCode = 1; });
}
