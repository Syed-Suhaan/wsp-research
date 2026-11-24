import torch, numpy as np, sys
sys.path.insert(0,'.')
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
from eval.common import predict_features, predict_logits
m=load_base()
xk=k_shot_examples(6); print('k',xk.shape)
for img in xk:
    print('\n'.join(''.join(' .:-=+*#%@'[min(9,int(v)//26)] for v in row[::1]) for row in img[::2]))
    print('-'*32)
x,y=load_val_old()
F=predict_features(m,x); Fk=predict_features(m,xk)
L=predict_logits(m,x); Lk=predict_logits(m,xk)
print('val acc',(L.argmax(1).numpy()==y).mean())
print('k logits'); print(Lk.numpy().round(2))
print('k softmax preds',Lk.argmax(1))
sd=base_state_dict()
W=sd['head.weight'];b=sd['head.bias']
print('W norms',W.norm(dim=1).numpy().round(2)); print('b',b.numpy().round(2))
print('F norm mean',F.norm(dim=1).mean().item(),'Fk norms',Fk.norm(dim=1).numpy().round(2))
mu=torch.stack([F[y==c].mean(0) for c in range(9)])
muk=Fk.mean(0)
cos=lambda a,b_: (a@b_.T)/(a.norm(dim=-1,keepdim=True)*b_.norm(dim=-1,keepdim=True).T)
print('cos muk vs class means',cos(muk[None],mu).numpy().round(3))
print('cos each k vs muk',cos(Fk,muk[None]).numpy().ravel().round(3))
# max logit distribution of old
print('old max logit pct', np.percentile(L.max(1).values.numpy(),[1,5,50]))
print('per-class own logit mean',[round(L[y==c,c].mean().item(),2) for c in range(9)])
torch.save({'F':F,'y':torch.tensor(y),'Fk':Fk,'L':L,'Lk':Lk},'wsp/stage1_v2/W1/run_6/scratch/feats.pt')
