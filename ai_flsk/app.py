from flask import Flask, request, jsonify
from generate_image import get_image

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
        return jsonify({'image': image}), 200
    except Exception as e:
        # 예외 발생 시 처리
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=5000)