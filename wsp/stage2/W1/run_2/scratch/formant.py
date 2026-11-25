import numpy as np
from scipy.signal import resample_poly
from scipy.linalg import solve_toeplitz
def formants(x, sr=24000, order=14):
    y = resample_poly(np.asarray(x,float), 1, 2)  # 12 kHz
    fs = 12000
    y = np.append(y[0], y[1:]-0.97*y[:-1])
    n = int(0.03*fs); hop = int(0.01*fs); win = np.hamming(n)
    frames = [y[i:i+n]*win for i in range(0, len(y)-n, hop)]
    E = np.array([np.sum(f**2) for f in frames]); thr = E.max()*10**(-2.5)
    out = []
    for f, e in zip(frames, E):
        if e < thr: continue
        r = np.correlate(f, f, 'full')[n-1:n+order]
        a = solve_toeplitz(r[:-1], -r[1:])
        roots = np.roots(np.r_[1, a]); roots = roots[np.imag(roots) > 0]
        fr = np.angle(roots)*fs/(2*np.pi); bw = -np.log(np.abs(roots))*fs/np.pi
        fr = np.sort(fr[(fr > 200) & (bw < 400) & (fr < 4500)])
        if len(fr) >= 3: out.append(fr[:3])
    return np.array(out)
