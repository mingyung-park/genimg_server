from flask import Flask, request, jsonify
from generate_image import get_image
import base64
import io


app = Flask(__name__)

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.get("/")
async def root():
    return { "message" : "This is Flask" }
@app.route('/get_image', methods=['POST'])
async def process_request():
    try:
        # 받은 요청의 데이터를 확인
        request_data = request.json
        prompt = request_data.get('prompt')
        image = get_image(prompt)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return jsonify({'image': encoded_image}), 200
    except Exception as e:
        # 예외 발생 시 처리
        print("Exception occurred in process_request:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
