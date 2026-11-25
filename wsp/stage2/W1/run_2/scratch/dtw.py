import numpy as np
def dtw_path(A, B):
    C = np.sqrt(((A[:,None,:]-B[None,:,:])**2).sum(-1))
    N,M = C.shape; D = np.full((N+1,M+1), np.inf); D[0,0]=0
    for i in range(1,N+1):
        Di, Dp, Ci = D[i], D[i-1], C[i-1]
        for j in range(1,M+1):
            Di[j] = Ci[j-1] + min(Dp[j], Dp[j-1], Di[j-1])
    i,j = N,M; path=[]
    while i>0 and j>0:
        path.append((i-1,j-1)); k = np.argmin([D[i-1,j-1], D[i-1,j], D[i,j-1]])
        if k==0: i,j=i-1,j-1
        elif k==1: i-=1
        else: j-=1
    return path[::-1]
