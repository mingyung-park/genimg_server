from flask import Flask, request, jsonify
from transformers import BertModel, PreTrainedTokenizerFast, GPT2LMHeadModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from diffusers import StableDiffusionPipeline
from kobert_tokenizer import KoBERTTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm_notebook
# from generate_comment import *
# from sentiment_music import *
# from generate_image import *
from model_class import BERTDataset,BERTClassifier
from torch import nn
from bucket import S3_ACESS_KEY,S3_SECRET_ACCESS_KEY,S3_BUCKET_NAME,AWS_S3_REGION_NAME

import gluonnlp as nlp
import pandas as pd
import numpy as np
import torch
import boto3
import uuid
import io
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

app = Flask(__name__)

#모델 로드
#gpt model
print('gpt_load')
gpt_device = torch.device("cuda:0")
gpt_model = GPT2LMHeadModel.from_pretrained('Eunju2834/aicomment_kogpt2').to(gpt_device)
gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained('Eunju2834/aicomment_kogpt2')
U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

#sentiment model
print('sentiment_load')
sm_model_path = 'CustomKoBERTWithLayNorm_epoch20_F1_SD.pth'
sm_tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1') 
sm_bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
sm_vocab = nlp.vocab.BERTVocab.from_sentencepiece(sm_tokenizer.vocab_file, padding_token='[PAD]') 
sm_tok = sm_tokenizer.tokenize
sm_device = torch.device("cuda:0")
sm_model = BERTClassifier(sm_bertmodel, dr_rate=0.5).to(sm_device)
print('sentiment_load_state_dict')
sm_model.load_state_dict(torch.load(sm_model_path))
sm_model.eval()
sm_token = nlp.data.BERTSPTokenizer(sm_tokenizer, sm_vocab, lower=False)

emotion_arr=['불안','분노','슬픔','당황','기쁨']
csv_file_path = 'music_emotion.csv'
final_emotion = pd.read_csv(csv_file_path)
max_len = 100
batch_size = 64
warmup_ratio = 0.1
num_epochs = 20
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

#image model
print('image_model')
image_model_path = 'Eunju2834/LoRA_oilcanvas_style'
image_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
image_device = torch.device('cuda')
image_pipe.unet.load_attn_procs(image_model_path)
image_pipe.to(image_device)
neg_prompt = '''FastNegativeV2,(bad-artist:1.0), (loli:1.2),
    (worst quality, low quality:1.4), (bad_prompt_version2:0.8),
    bad-hands-5,lowres, bad anatomy, bad hands, ((text)), (watermark),
    error, missing fingers, extra digit, fewer digits, cropped,
    worst quality, low quality, normal quality, ((username)), blurry,
    (extra limbs), bad-artist-anime, badhandv4, EasyNegative,
    ng_deepnegative_v1_75t, verybadimagenegative_v1.3, BadDream,
    (three hands:1.1),(three legs:1.1),(more than two hands:1.4),
    (more than two legs,:1.2),badhandv4,EasyNegative,ng_deepnegative_v1_75t,verybadimagenegative_v1.3,(worst quality, low quality:1.4),text,words,logo,watermark,
    '''
    
    
def inference(sentence):
    dataset = [[sentence, '0']]
    test = BERTDataset(dataset, 0, 1, sm_tok, sm_vocab,max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, num_workers=2)

    output = []
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
      token_ids = token_ids.long().to(sm_device)
      segment_ids = segment_ids.long().to(sm_device)
      valid_length= valid_length
      label = label.long().to(sm_device)
      output.append(sm_model(token_ids, valid_length, segment_ids))
    softmax = torch.nn.Softmax(dim=1)
    output = softmax(output[0])[0].tolist()
    return output


def get_image(prompt):
    image = image_pipe(prompt, negative_prompt=neg_prompt,num_inference_steps=30, guidance_scale=7.5).images[0]
    return image


def get_comment(input_text): #koGPT2 모델을 활용하여 입력된 질문에 대한 대답을 생성하는 함수
    q = input_text
    a = ""
    sent = ""
    while True:
        input_ids = torch.LongTensor(gpt_tokenizer.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0).to(gpt_device)
        pred = gpt_model(input_ids)
        pred = pred.logits
        gen = gpt_tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().tolist())[-1]
        if gen == EOS:
            break
        a += gen.replace("▁", " ")
    return a


def get_emotion_label(content):
    emotion_pred = inference(content)
    max_value = max(emotion_pred)
    max_index = emotion_pred .index(max_value)
    return emotion_pred, emotion_arr[max_index]


def get_music(content):
    emotion_pred, max_index=get_emotion_label(content)
    df_user_sentiment = pd.DataFrame([emotion_pred],columns=emotion_arr)
    user_emotion_str = df_user_sentiment.apply(lambda x: ' '.join(map(str, x)), axis=1)
    music_emotion_str = final_emotion[emotion_arr].apply(lambda x: ' '.join(map(str, x)), axis=1)

    tfidf = TfidfVectorizer()
    user_tfidf_matrix = tfidf.fit_transform(user_emotion_str)
    music_tfidf_matrix = tfidf.transform(music_emotion_str)

    cosine_sim = cosine_similarity(user_tfidf_matrix, music_tfidf_matrix)
    
    most_similar_song_index = cosine_sim.argmax()
    most_similar_song_info = final_emotion.iloc[most_similar_song_index]

    num_additional_recommendations = 4
    similar_songs_indices = cosine_sim.argsort()[0][-num_additional_recommendations-1:-1][::-1]
    similar_songs_info = final_emotion.iloc[similar_songs_indices]

    return most_similar_song_info, similar_songs_info

# Flask
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
        s3.upload_fileobj(buffered, Bucket=S3_BUCKET_NAME, Key=f'images/{image_key}.jpg', ExtraArgs={'ContentType':'image/jpeg'})
        image_url = f'https://{S3_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/images/{image_key}.jpg'
        buffered.close()
        return jsonify({'image_url': image_url}), 200
    except Exception as e:
        print("Exception occurred in process_request:", e)
        return jsonify({"error": str(e)}), 500
        
@app.route('/get_comment', methods=['POST'])
async def process_comment_request():
    try:
        request_data = request.json
        content = request_data.get('content')
        comment = get_comment(content)
        
        return jsonify({'comment': comment}), 200
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
    app.run(host='0.0.0.0', port=5000, debug=False)
