import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize']=[16,8]

xC=np.array([2,1])
sig = np.array([2,0.5])

theta=np.pi/3

R=np.array([[np.cos(theta),-np.sin(theta)],
           [np.sin(theta),np.cos(theta)]])

nPoints=1000

X = R @ np.diag(sig) @ np.random.randn(2,nPoints) + np.diag(xC) @ np.ones((2,nPoints))

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(X[0,:],X[1,:], '.', color='k')
ax1.grid()
plt.xlim((-6, 8))
plt.ylim((-6,8))



Xmoy = np.mean(X,axis=1)                  #moyenne
B = X - np.tile(Xmoy,(nPoints,1)).T       #soustraction de la moyenne
#ici le but est de centrer nos données

#composantes principales (SVD)
#on applique la SVD après avoir réduit nos données. 
U, S, VT = np.linalg.svd(B/np.sqrt(nPoints),full_matrices=0)


ax2 = fig.add_subplot(122)
ax2.plot(X[0,:],X[1,:], '.', color='k')   
ax2.grid()
plt.xlim((-6, 8))
plt.ylim((-6,8))

theta = 2 * np.pi * np.arange(0,1,0.01)


Xstd = U @ np.diag(S) @ np.array([np.cos(theta),np.sin(theta)])

ax2.plot(Xmoy[0] + Xstd[0,:], Xmoy[1] + Xstd[1,:],'-',color='r')
ax2.plot(Xmoy[0] + 2*Xstd[0,:], Xmoy[1] + 2*Xstd[1,:],'-',color='r')

ax2.plot(Xmoy[0] + 3*Xstd[0,:], Xmoy[1] + 3*Xstd[1,:],'-',color='r')


# Plot composantes proncipales U[:,0]S[0] et U[:,1]S[1]

#Première composante principale
ax2.plot(np.array([Xmoy[0], Xmoy[0]+U[0,0]*S[0]]),
         np.array([Xmoy[1], Xmoy[1]+U[1,0]*S[0]]),'-',color='cyan')
#2ème composante principale. 
ax2.plot(np.array([Xmoy[0], Xmoy[0]+U[0,1]*S[1]]),
         np.array([Xmoy[1], Xmoy[1]+U[1,1]*S[1]]),'-',color='cyan')


plt.show()
