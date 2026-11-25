import numpy as np
mc = np.load('wsp/stage2/W1/run_4/scratch/a2_mc.npy')  # target, F2, M2, tokM2_frF2, tokF2_frM2
t, f, m = mc[0,1:], mc[1,1:], mc[2,1:]
a, b = t-m, m-f
print('cos(target-M2, M2-F2) = %.3f' % (a@b/np.linalg.norm(a)/np.linalg.norm(b)), '|t-m| %.3f |m-f| %.3f proj coef %.3f' % (np.linalg.norm(a), np.linalg.norm(b), a@b/(b@b)))
print(np.round(a[:15],2)); print(np.round(b[:15],2))
