import numpy as np
N=100


net_out = np.array([1,5,3,1])
net_out = 1.*net_out/net_out.sum()

hist=[]
for i in xrange(1000):
    act_list = np.random.randint(0, 4, (N,))
    prob_list = np.random.uniform(size=(N,))
    idx = list(net_out[act_list] > prob_list).index(True) #np.where(net_out[act_list] > prob_list)
    hist.append(act_list[idx])

h=np.histogram(hist, bins=4)[0]
print h*1./h.sum()




