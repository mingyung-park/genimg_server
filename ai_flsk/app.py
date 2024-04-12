from flask import Flask, request, jsonify
from generate_image import get_image
from sentiment_music import *
from bucket import S3_ACESS_KEY,S3_SECRET_ACCESS_KEY,S3_BUCKET_NAME,AWS_S3_REGION_NAME
import boto3
import uuid
import io

app = Flask(__name__)
s3 = boto3.client('s3',
                  aws_access_key_id=S3_ACESS_KEY,
                  aws_secret_access_key=S3_SECRET_ACCESS_KEY)

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
async def process_image_request():
    try:
        # 받은 요청의 데이터를 확인
        request_data = request.json
        prompt = request_data.get('prompt')
        image = get_image(prompt)
        
        image_key = str(uuid.uuid4())
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        buffered.seek(0)
        s3.upload_fileobj(buffered, Bucket=S3_BUCKET_NAME, Key=f'images/{image_key}', ExtraArgs={'ContentType':'image/jpeg'})
        image_url = f'https://{S3_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{image_key}'
        buffered.close()
        return jsonify({'image_url': image_url}), 200
    except Exception as e:
        print("Exception occurred in process_request:", e)
        return jsonify({"error": str(e)}), 500
    
@app.route('/get_sentiment', methods=['POST'])
async def process_sentiment_request():
    try:
        request_data = request.json
        content = request_data.get('content')
        _, emotion_label = get_emotion_label(content)
        
        return jsonify({'emotion_label': emotion_label}), 200
    except Exception as e:
        print("Exception occurred in process_request:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/get_music', methods=['POST'])
async def process_music_request():
    try:
        request_data = request.json
        content = request_data.get('content')
        most_similar_song_info, similar_songs_info = get_music(content)
        
        response_data = {
            'most_similar_song': {
                'title': most_similar_song_info[0],
                'artist': most_similar_song_info[1],
                'genre': most_similar_song_info[2]
            },
            'similar_songs': [{
                'title': song_info[0],
                'artist': song_info[1],
                'genre': song_info[2]
            } for song_info in similar_songs_info.values]
        }
        return jsonify(response_data), 200
    except Exception as e:
        print("Exception occurred in process_request:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
