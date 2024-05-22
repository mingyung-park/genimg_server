
import torch
import numpy as np
import pandas as pd
from torch import nn
import gluonnlp as nlp
from tqdm import tqdm_notebook
from transformers import BertModel
from torch.utils.data import Dataset
from kobert_tokenizer import KoBERTTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


max_len = 100
batch_size = 64
warmup_ratio = 0.1
num_epochs = 20
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer,vocab, max_len,
                 pad, pair):

        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=5,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier =  nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=hidden_size, out_features=512),
            nn.Linear(in_features=512, out_features=num_classes)
        )
        self.layer_norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device)) #BERT 모델에 입력을 전달하여 출력을 계산

        pooled_output = self.dropout(pooler)
        normalized_output = self.layer_norm(pooled_output)
        out=self.classifier(normalized_output)

        return out

model_path = './CustomKoBERTWithLayNorm_epoch20_F1.pth'
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
print('bert')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)

vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
tok = tokenizer.tokenize
model = torch.load(model_path)
model = BERTClassifier(bertmodel,  dr_rate=0.5)
model.eval()

token = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
def inference(sentence):
    dataset = [[sentence, '0']]
    test = BERTDataset(dataset, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, num_workers=2)

    output = []
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
      token_ids = token_ids.long().to(device)
      segment_ids = segment_ids.long().to(device)
      valid_length= valid_length
      label = label.long().to(device)
      output.append(model(token_ids, valid_length, segment_ids))
    softmax = torch.nn.Softmax(dim=1)
    output = softmax(output[0])[0].tolist()
    return output

emotion_arr=['불안','분노','슬픔','당황','기쁨']
csv_file_path = 'music_emotion.csv'
final_emotion = pd.read_csv(csv_file_path)

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

    # TF-IDF 벡터화
    tfidf = TfidfVectorizer()
    user_tfidf_matrix = tfidf.fit_transform(user_emotion_str)
    music_tfidf_matrix = tfidf.transform(music_emotion_str)

    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(user_tfidf_matrix, music_tfidf_matrix)

    # 가장 유사한 음악 선택
    most_similar_song_index = cosine_sim.argmax()
    most_similar_song_info = final_emotion.iloc[most_similar_song_index]

    # 선택된 음악과 유사한 음악 4곡 더 추천
    num_additional_recommendations = 4
    similar_songs_indices = cosine_sim.argsort()[0][-num_additional_recommendations-1:-1][::-1]
    similar_songs_info = final_emotion.iloc[similar_songs_indices]

    return most_similar_song_info, similar_songs_info
