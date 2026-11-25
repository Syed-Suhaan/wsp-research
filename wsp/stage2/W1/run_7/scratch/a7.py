import sys; sys.path.insert(0,'/home/user/wsp-research/wsp/stage2/W1/run_7/scratch')
exec(open('/home/user/wsp-research/wsp/stage2/W1/run_7/scratch/a5.py').read().split("m.voices.weight.data[2]=V[0]")[0])
m.voices.weight.data[2]=V[0]
S=m.prosody_stack
with torch.no_grad():
    u=S.norm.weight*m.prosody_head.weight[0]; u=u-u.mean(); uh=u/u.norm()
for beta in [-3,-6.6,-10]:
    fv=FV[1]+beta*uh
    print('u beta=%5.1f norm %.1f pitch mean %.3f sd %.3f'%(beta,fv.norm(),*prosody(fv)))
