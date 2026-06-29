import os
import cv2
import requests
import numpy as np
from flask import Flask, request, jsonify, send_file
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import tempfile
import io

app = Flask(__name__)

SOURCE_FACE_URL = "https://i.ibb.co/Tq8D18ZL/Ayesha-Sadiq.jpg"

print("Loading face analysis model...")
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1, det_size=(640, 640))

print("Loading swapper model...")
swapper = get_model('inswapper_128.onnx', download=True, download_zip=True)

def download_image(url):
    resp = requests.get(url, timeout=30)
    arr = np.frombuffer(resp.content, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

source_img = download_image(SOURCE_FACE_URL)
source_faces = face_app.get(source_img)
print(f"Source face loaded: {len(source_faces)} face(s) detected")

@app.route('/swap', methods=['POST'])
def swap_face():
    data = request.json
    target_url = data.get('target_url')
    
    if not target_url:
        return jsonify({'error': 'target_url required'}), 400
    
    try:
        target_img = download_image(target_url)
        target_faces = face_app.get(target_img)
        
        if not target_faces:
            return jsonify({'error': 'No face detected in target image'}), 400
        
        result = target_img.copy()
        for face in target_faces:
            result = swapper.get(result, face, source_faces[0], paste_back=True)
        
        _, buffer = cv2.imencode('.jpg', result)
        return send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='swapped.jpg'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
