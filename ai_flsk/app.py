from flask import Flask, request, jsonify
from .generate_image import get_image
app = Flask(__name__)

@app.get("/")
async def root():
	return { "message" : "This is Flask" }


@app.route('/get_image', methods=['POST'])
def process_request():
    # 받은 요청의 데이터를 확인
    request_data = request.json
    prompt = request_data.get('prompt')

    # AI 모델을 사용하여 이미지 생성
    image = get_image(prompt)

    # 이미지 데이터를 반환
    return jsonify({'image': image})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)