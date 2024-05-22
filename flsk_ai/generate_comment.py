import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('Eunju2834/aicomment_kogpt2')
tokenizer = PreTrainedTokenizerFast.from_pretrained('Eunju2834/aicomment_kogpt2')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'
    
    
def get_comment(input_text): #koGPT2 모델을 활용하여 입력된 질문에 대한 대답을 생성하는 함수
    q = input_text
    a = ""
    sent = ""
    while True:
        input_ids = torch.LongTensor(tokenizer.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
        input_ids = input_ids.to(device)
        pred = model(input_ids)
        pred = pred.logits
        gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().tolist())[-1]
        if gen == EOS:
            break
        a += gen.replace("▁", " ")
    return a


# diary='''
# 물리적으로도 피곤하고 마음 속으로도 지친 하루였다. 하루종일 뜬눈으로 시간을 보냈다.
# 아침 일어나자마자부터 마음이 무겁고 뭔가 잘못된 기분이 들었다.
# 머릿속이 복잡해서 무엇부터 해야 할지 막막했다. 친구들과의 대화조차도 어색하게 느껴졌다.
# 하루 종일 날이 어두워져가는 것처럼 마음도 점점 어두워져만 갔다.
# 이럴 때마다 왜 이렇게 우울해지는 건지 모르겠다. 아무것도 재미있게 느껴지지 않았다.
# 음악도, 영화도, 책도, 아무것도 마음에 들지 않았다.
# 책상 위에는 무언가를 해야겠다는 생각으로 쌓아둔 책과 종이들이 더럽혀져만 가는데, 손을 대기 싫었다.
# '''
# print(get_comment(diary))
