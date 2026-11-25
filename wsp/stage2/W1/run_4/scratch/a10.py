import sys; sys.path.insert(0, 'wsp/stage2/W1/run_4/scratch')
from common4 import *
import pyworld as pw
sd = T.tts_base_state_dict(); texts = T.tts_texts('enroll')
W = sd['vocoder.harmonic.weight']; print(W.shape, sd['vocoder.noise.weight'].shape)
Wc = W[1:80]  # causal coefficients 1..79 (pre-lifter)
lift = 1.0/(1.0+torch.arange(0,80).float()/4)
L = np.load('wsp/stage2/W1/run_4/scratch/ltas.npy')  # T, F2, M2 ; natural log, 1025 bins (cheaptrick fft 2048)
d = torch.tensor(L[0]-L[2]).float()  # target - M2
nb = d.shape[0]; om = torch.linspace(0, np.pi, nb)
K = 24
k = torch.arange(1, K+1).float()
# cosine-series coefficients of the log-magnitude difference: log|H| = c0 + sum_k c_k cos(k w)
dc = (2.0/np.pi) * torch.trapz(d[None]*torch.cos(k[:,None]*om[None]), om[None].expand(K,-1), dim=1)
dh = torch.zeros(79); dh[:K] = dc/lift[1:K+1]
ga = sd['acoustic_stack.norm.weight']
A = Wc * ga[None]   # linearised map from acoustic residual (pre-norm, scaled by 1/sigma) to coefficients
v = A.T @ dh
out = A @ v
print('cos(A A^T dh, dh) = %.3f' % torch.cosine_similarity(out, dh, 0))
S = torch.linalg.svdvals(A); print('sing vals', S[:10].numpy().round(2), S[-5:].numpy().round(3))
# reconstruction check of the smoothed curve
rec = (dc[:,None]*torch.cos(k[:,None]*om[None])).sum(0)
print('rms d %.3f, rms d-mean %.3f, rms residual after K terms %.3f' % (d.pow(2).mean().sqrt(), (d-d.mean()).pow(2).mean().sqrt(), (d-d.mean()-rec+rec.mean()).pow(2).mean().sqrt()))
