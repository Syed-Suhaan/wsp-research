import sys, numpy as np, torch, pyworld as pw, librosa, scipy.fft
sys.path.insert(0,'/home/user/wsp-research')
torch.set_num_threads(4)
from eval import wsp_tools as W
from models.tiny_voca3 import TinyVoca3, use_checkpoint_frontend
use_checkpoint_frontend()
from tiny_voca.text import phonemize
SR=24000
BASE=W.tts_base_state_dict()
def model_from(sd):
    m=TinyVoca3(); m.load_state_dict(sd,strict=True); return m.eval()
def internals(m, text, voice):
    p=phonemize(text)
    tok=torch.tensor([p.ids]); st=torch.tensor([p.stress]); L=torch.tensor([len(p)])
    w,n,d=m.synthesize(tok,st,L,torch.tensor([voice]))
    pr=d['prosody'][0]; voiced=pr[:,1]>0
    return dict(logf0=float(pr[voiced,0].mean())+np.log(200), frames=int(d['frame_mask'].sum()), energy=float(pr[:,2].mean()))
MEL=librosa.filters.mel(sr=SR,n_fft=1024,n_mels=80,fmin=0,fmax=8000)
def feats(x):
    x=np.ascontiguousarray(x,dtype=np.float64)
    f0,t=pw.harvest(x,SR,f0_floor=50,f0_ceil=600,frame_period=10)
    sp=pw.cheaptrick(x,f0,t,SR,fft_size=1024)
    v=f0>0
    lm=np.log(sp[v]@MEL.T+1e-10)
    cep=scipy.fft.dct(lm,type=2,norm='ortho',axis=1)[:,:25]
    return dict(lf0=np.median(np.log(f0[v])), cep=cep.mean(0), lmel=lm.mean(0))
def synth(m,text,voice): return np.asarray(W.tts_synthesize(m,text,voice,0),dtype=np.float64)
