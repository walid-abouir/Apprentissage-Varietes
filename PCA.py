# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:45:08 2023

@author: aboui
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import random
import scipy.linalg as spl
from PIL import Image
import some_PSF
#passer de matrice (array 2D ) à un vecteur (array 1D)

def Matrix_To_Vector(M):
    n,l,c=np.shape(M)
    return np.reshape(M,(n,l*c) )

def Vector_to_Matrix(V):
    if (len(np.shape(V))==2):
        n,N=np.shape(V)
    if (len(np.shape(V))==1):
        N=np.shape(V)
        n=1
    else:
        print("Erreur : pas la bonne dimension")
        return(-1)
    
    if int(np.sqrt(N))==np.sqrt(N):
        p=int(np.sqrt(N)) #taille de la matrice
    if n != 1:
        return np.reshape(V,(n,p,p))
    else:
        return np.reshape(V,(p,p))

# Transformer une image couleur en une image en noir et blanc

def Noir_Blanc_Image(A):
    return A[:,:,0]
    
def Noir_Blanc_tab(A):
    N=np.shape(A)
    B=np.zeros(N[0:3])
    for i in range(N[0]): #On parcourt toutes les images
        B[i,:,:]=Noir_Blanc_Image(A[i,:,:,:])
    return B

def Image_aff(Im,title="",style='psf',show=False):

# Objectif du programme : Afficher un 
#PSF sous format d'image # Arguments en entrée: Im : l'image à afficher, title
# le titre de l'image# style : le style d'affichage (psf : affichage comme les psf).
# et show, si un plt.show doit être executé ou non
# Arguments en sortie: renvoie 0
    Vmin,Vmax=np.min(Im),np.max(Im)
    plt.imshow(Im,vmin=Vmin,vmax=Vmax) #Aff PSF
    plt.title(title)

    if show:
        plt.show()
    return 0

def SVD(A):
 # Objectif du programme : réaliser une SVD sur une matrice
# Arguments en entrée: A : la matrice dont on veut la décomposition# Arguments en sortie: U,S,V : U et V étant les matrices orthogonales# des vecteurs singuliers à gauche/droite et S est une matrice dont la # diagonale est consistuée des valeurs singulières strictement positives# par ordre décroissant
    Aa=np.transpose(A)
    l,V=spl.eigh(np.dot(Aa,A))
    l=l[::-1] #On inverse l'ordre des vp pour les avoir dans l'ordre décroissante V=V[:,::-1]

    l=l[l>0] #On retire les valeurs négatives ou nulles
    rg=len(l)
    V=V[:,0:rg]

    s=np.sqrt(l)
    U=np.dot(A,V)/s
    S=np.diag(s)
    return U,S,V

def Vecteur_Aff(Comp,title="",style='psf',show=False):

    Mat=Comp-np.min(Comp)
    Mat=Vector_to_Matrix(Mat)
    Vmax=np.max(Mat)

    plt.imshow(Mat, vmin=0, vmax=Vmax)
    plt.title(title)
    if show:
        plt.show()
    return 0

def Aff_SVD(Tab_Im,nbr=0,style='psf'):

    U,Sigma,V=SVD(Matrix_To_Vector(Tab_Im))#On effectue l'ACP sur nos données print(U)
    fig=plt.figure()

    for i in range(nbr):
        plt.subplot(2,round(nbr/2-0.01)+1,i+1)
        title="Dimension n°"+str(i)
        Vecteur_Aff(U[i,:],title=title,style=style)
        plt.axis('off')

    plt.show()
    return 0
def PCA(im,nbr=0,seuil=0):
    
    # Objectif du programme : Effectuer une ACP sur une base de donnée d'images
    # Arguments en entrée: im : le vecteur d'image sur lequel on effectue l'ACP
    # nbr : le nombre de vp que l'on souhaite conservé. On les conservent toutes
    # si rien n'est précisé ou si la valeur entrée est de 0
    # seuil : l'inertie minimale pour que l'on souhaite conserver une vp.
    # On les conservent toutes si rien n'est précisé ou si la valeur entrée est de 0
    # Si nbr et seuil sont précisés, on applique les deux critères
    # Arguments en sortie: la matrice de travail centré T, la matrice moyenne moy,
    # la matrice de variance-covariance Gamma, le vecteur des valeurs propres
    # Eigenvalues, la matrice des vecteurs propres Eigenvectors, et le vecteur
    # Inertie, dont le premier élément est l'inertie globale et les éléments suivants
    # sont les inerties des vp conservées
     n=len(im[1,:,:])
     N=len(im[:,1,1])
    
     T=Matrix_To_Vector(im) #On linéarise les matrices d'images sous forme de vecteur#On centre la matrice T
     moy=[] #vecteur qui stocke les moyennes selon chaque variables
     for i in range(n*n):
         s=np.sum(T[:,i]) #Somme des valeurs de chaque variables 
         m=s/N #Calcul de la moyenne empirique 
         T[:,i]=T[:,i]-m #On centre la matrice
         moy+={m} #On ajoute la valeur de la moyenne au vecteur 
     moy=np.array(moy)
     print("PCA : matrice de travail T centrée")
    
     M=np.identity(n*n) #Définition des matrices de travail, même si pas indispensable  
     W=1/(N)*np.identity(N)
    
     print("PCA : Produit matriciel débuté")
     Tt=np.transpose(T)
     Gamma=np.dot(np.dot(Tt,W),T) #Calcul de la matrice gamma
     print("PCA : Produit matriciel terminé")
    
    #On devrait réaliser ici le produit matriciel Gamma*M, mais comme M=identité, cette étape est#On diagonalise GammaM pour obtenir les valeurs propres et les vecteurs propres#GammaM est une matrice carré de dimension n² par n², symétrique et donc diagonalisable 
     print("PCA : Diagonalisation débutée")
     (Eigenvalues,Eigenvectors)=npl.eig(Gamma)
     print("PCA : Diagonalisation terminée")
    
     #Calcul de l'inertie de chaque vp
     InG=np.sum(Eigenvalues) #Inertie Globale
     InPart=Eigenvalues/InG #Contribution inertie
     Inertie=np.concatenate(([InG],InPart))
    
     if (nbr!=0): #On conserve uniquement le nombre de valeurs propres demandés en entrée 
         Eigenvalues , Eigenvectors=Eigenvalues[0:nbr] , Eigenvectors[0:nbr,:]
         Inertie=Inertie[:nbr+1]
     if (seuil>0): #on conserve uniquement les vp qui respectent le seuil en entrée 
         l=len(Eigenvalues)
         s=0
    
         while ((Inertie[s+1]>(seuil)) and (s<l-1)):
             s+=1
         if (s<l):
             Eigenvalues,Eigenvectors=Eigenvalues[0:s],Eigenvectors[0:s,:] 
             Inertie=Inertie[:s+1]
    
     return T,moy,Gamma,Eigenvalues,Eigenvectors,Inertie
 
def PCA_Decomposition(im,ResPCA):

# Objectif du programme : décomposer un ensemble d'image à l'aide des composantes# principales obtenues grâce à une ACP
# Arguments en entrée: im : le vecteur d'image qu'on décompose# Arguments en sortie: im_dec : la décomposition de im en composantes principales
    Im_vect=Matrix_To_Vector(im)
    N,d=np.shape(Im_vect)
    n=len(ResPCA[3]) #nombre de vp

    im_dec=np.zeros((N,n))
    for i in range(N):
        ImC=Im_vect[i]-ResPCA[1] #On centre l'image que l'on souhaite décomposer 
        for j in range(n):
            im_dec[i,j]=np.dot(ImC,ResPCA[4][j])
    return im_dec

def Composante_Aff(Comp,title="",style='psf',show=False):

     Mat=Vector_to_Matrix(Comp)
     bor=max(abs(np.min(Mat)),abs(np.max(Mat)))
     plt.imshow(Mat, vmin=-bor, vmax=bor)
     plt.title(title)
     if show:
         plt.show()
     return 0
 
def Argmins(Tab,nbr=1):

# Objectif du programme : Recupérer l'argument des nbr valeurs minimales de Tab# Arguments en entrée: Tab : le tableau des valeurs dont on souhaite récupérer # les argmin
# nbr : le nombre d'arguments minimaux que l'on souhaite obtenir, par défaut 1# Arguments en sortie: renvoie un vecteur contenant les arguments minimaux 
    tab=np.copy(Tab)
    m=max(tab)
    Arg=np.zeros(nbr,dtype=int)
    for i in range(nbr):
        Arg[i]=np.argmin(tab)
        tab[Arg[i]]=m

    return Arg


def Argmaxs(Tab,nbr=1):

 # Objectif du programme : Recupérer l'argument des nbr valeurs maximales de Tab# Arguments en entrée: Tab : le tableau des valeurs dont on souhaite récupérer # les argmax
# nbr : le nombre d'arguments maximaux que l'on souhaite obtenir, par défaut 1# Arguments en sortie: renvoie un vecteur contenant les arguments maximaux
    tab=np.copy(Tab)
    m=min(tab)
    Arg=np.zeros(nbr,dtype=int)
    for i in range(nbr):
        Arg[i]=np.argmax(tab)
        tab[Arg[i]]=m

    return Arg

def Aff_PCA(Tab_Im,nbr=0,seuil=0,stylePSF='psf',styleComp='psf'):
    ResPCA=PCA(Tab_Im,nbr,seuil) #On effectue l'ACP sur nos données 
    Dec_im=PCA_Decomposition(Tab_Im,ResPCA) #On décompose les images en composantes principales 
    nbr_comp=len(ResPCA[3]) #nbr de composantes

    Moy_Mat=Vector_to_Matrix(ResPCA[1])
#Affichage des résultats sur une unique figure avec plt.GridSpec 
    fig1= plt.figure(figsize=(8,2*(nbr_comp+1)))
    row,col=3*(nbr_comp+1)-1,10
    grid=plt.GridSpec(row,col,wspace = .25,hspace = .25)

    plt.subplot(grid[0:2,4:6])
    Image_aff(Moy_Mat,title="Image moyenne",style=stylePSF)
    plt.axis('off')
    
    for i in range(nbr_comp):
        plt.subplot(grid[3*(i+1):3*(i+1)+2,0:2])
        title="Pourcentage d'Inertie = "+str(round(ResPCA[5][i+1].real*100,2))+" %" 
        Composante_Aff(ResPCA[4][i],title=title,style=styleComp)

        plt.axis('off')
        Arg_min=Argmins(Dec_im[:,i],6)
        Arg_max=Argmaxs(Dec_im[:,i],6)

        plt.subplot(grid[3*(i+1),4])
        title="PSF Min"
        Image_aff(Tab_Im[Arg_min[0]],title=title,style=styleComp) 
        plt.axis('off')
        plt.subplot(grid[3*(i+1),3])
        Image_aff(Tab_Im[Arg_min[1]],style=styleComp)
        plt.axis('off')
        plt.subplot(grid[3*(i+1)+1,3])
        Image_aff(Tab_Im[Arg_min[2]],style=styleComp)
        plt.axis('off')
        plt.subplot(grid[3*(i+1)+1,4])
        Image_aff(Tab_Im[Arg_min[3]],style=styleComp)
        plt.axis('off')
        plt.subplot(grid[3*(i+1),5])
        Image_aff(Tab_Im[Arg_min[4]],style=styleComp)
        plt.axis('off')
        plt.subplot(grid[3*(i+1)+1,5])
        Image_aff(Tab_Im[Arg_min[5]],style=styleComp)
        plt.axis('off')

        plt.subplot(grid[3*(i+1),8])
        title="PSF Max"
        Image_aff(Tab_Im[Arg_max[0]],title=title,style=styleComp) 
        plt.axis('off')
        plt.subplot(grid[3*(i+1),7])
        Image_aff(Tab_Im[Arg_max[1]],style=styleComp)
        plt.axis('off')
        plt.subplot(grid[3*(i+1)+1,7])
        Image_aff(Tab_Im[Arg_max[2]],style=styleComp)
        plt.axis('off')
        plt.subplot(grid[3*(i+1)+1,8])
        Image_aff(Tab_Im[Arg_max[3]],style=styleComp)
        plt.axis('off')
        plt.subplot(grid[3*(i+1),9])
        Image_aff(Tab_Im[Arg_max[4]],style=styleComp)
        plt.axis('off')
        plt.subplot(grid[3*(i+1)+1,9])
        Image_aff(Tab_Im[Arg_max[5]],style=styleComp)
        plt.axis('off')

    return 0

path="PSFs.tif"
PSF = some_PSF.load_image(path)
Aff_PCA(PSF, nbr=4, seuil = 0.0004)