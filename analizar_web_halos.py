\
import matplotlib.pyplot as plt
#import matplotlib as plt
plt.switch_backend('agg')
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

path="/hpcfs/home/sc.sanabria1984/momografia/rockstar/Rockstar-0.99.9-RC3/HALOS_50_512/out_0.list"
path2="/hpcfs/home/sc.sanabria1984/momografia/rockstar/Rockstar-0.99.9-RC3/HALOS_50/out_0.list"
pathtv_256="/hpcfs/home/sc.sanabria1984/momografia/tv_web/TV-Web/util/FA.dat"
patheigen1="/hpcfs/home/sc.sanabria1984/momografia/tv_web/TV-Web/util/eigenval1.dat"
patheigen2="/hpcfs/home/sc.sanabria1984/momografia/tv_web/TV-Web/util/eigenval2.dat"
patheigen3="/hpcfs/home/sc.sanabria1984/momografia/tv_web/TV-Web/util/eigenval1.dat"





###### cargar datos 50 mpc 512
f = open ( path2 , 'r')
l2 = []
l2 = [ line.split() for line in f]

########## cargar datos 50 mpc 256

f = open ( path , 'r')
halos = []
halos = [ line.split() for line in f]
########### cargaar tv 256

f = open ( pathtv_256 , 'r')
tv = []
tv= [ line.split() for line in f]
FA = []
FA = [[float(j) for j in i] for i in tv]
##############################

########### cargar eigen1

f = open ( patheigen1 , 'r')
eigen = []
eigen= [ line.split() for line in f]
eigen1 = []
eigen1 = [[float(j) for j in i] for i in eigen]
##############################
########### cargar eigen2

f = open ( patheigen2 , 'r')
eigen = []
eigen= [ line.split() for line in f]
eigen2 = []
eigen2 = [[float(j) for j in i] for i in eigen]
##############################
########### cargar eigen3

f = open ( patheigen3 , 'r')
eigen = []
eigen= [ line.split() for line in f]
eigen3 = []
eigen3 = [[float(j) for j in i] for i in eigen]
##############################
imagelen=200
image=np.zeros((imagelen,imagelen),dtype=float)
void_ubi=[]


#print np.array(FA).shape
def read_cut(eigen1,eigen2,eigen3,FA,cut):
   FA=np.array(FA)  
   eigen1=np.array(eigen1)
   eigen2=np.array(eigen2)
   eigen3=np.array(eigen3)
   #print FA
   l=FA.shape[1]
   FA_cut=FA[cut*l:(cut+1)*l,:]
   eigen1_cut=eigen1[cut*l:(cut+1)*l,:]
   eigen2_cut=eigen2[cut*l:(cut+1)*l,:]
   eigen3_cut=eigen3[cut*l:(cut+1)*l,:]
  
   return {'eigen1_cut':eigen1_cut,'eigen2_cut':eigen2_cut,'eigen3_cut':eigen3_cut,'FA_cut':FA_cut}


def classify_environment(eigen1,eigen2,eigen3):
  void_ubi=[]
  sheet_ubi=[]
  filament_ubi=[]
  peak_ubi=[]
  num_void=0
  num_sheet=0
  num_filament=0
  num_peak=0
  image=np.zeros((len(eigen1),len(eigen2)),dtype=float)
  for j in range(len(eigen1)):
   for i in range(len(eigen1)):
    count=0
    a=eigen1[i][j]
    b=eigen2[i][j]
    c=eigen3[i][j]
    if a>0.3:
     count=count+1
    if b>0.3:
     count=count+1
    if c>0.3:
     count=count+1
    if count==0:
     image[i][j]=0
     void_ubi.append((i,j))
     num_void=num_void+1
    if count==1:
     image[i][j]=10
     sheet_ubi.append((i,j))
     num_sheet=num_sheet+1
    if count==2:
     image[i][j]=20
     filament_ubi.append((i,j))
     num_filament=num_filament+1
    if count==3:
     image[i][j]=30
     peak_ubi.append((i,j))  
     num_peak=num_peak+1
    num_all=[num_peak,num_filament,num_sheet,num_void] 
  return {'matrix_web':image,'peak_ubi':peak_ubi ,'filament_ubi':filament_ubi,'sheet_ubi':sheet_ubi,'void_ubi':void_ubi,'num_all':num_all}



def cut_halos(halos,matrix,cut):
 length_cut=50/200.0
 aux=[]
 xcut=[]
 ycut=[]
 
 for i in range(16,len(halos)):
    if float(halos[i][10])<(cut+1)*length_cut and float(halos[i][10])>(cut)*length_cut: 
	aux.append(i)

 for i in range(len(aux)):
    xcut.append(float(halos[aux[i]][8]))    
    ycut.append(float(halos[aux[i]][9]))

 indy=[]    
 halos_ubi=[]
 for k in range(len(xcut)): 
    for j in range(len(image)):
        for i in range(len(image)):
            if xcut[k]<length_cut*(i+1) and xcut[k]>length_cut*i and ycut[k]<length_cut*(j+1) and ycut[k]>length_cut*j: 
                matrix[i][j]=100
                halos_ubi.append((i,j,aux[k]))
                
 return {'halos_ubi':halos_ubi,'matrix_halos':matrix}


#fig = plt.figure(figsize=(20,20))
#plt.imshow(results['matrix_halos'],extent=[0,50,50,0])
#fig.savefig("test.jpg", bbox_inches='tight')
# print results['num_all']


def find_mass(halos_ubi,void_ubi):
 halos_void=[]
 for j in range(len(void_ubi)):
  for i in range(len(halos_ubi)):
   if halos_ubi[i][0]==void_ubi[j][0] and halos_ubi[i][1]==void_ubi[j][1]:
     halos_void.append(float(halos[halos_ubi[i][2]][0]))
 return {'mass':halos_void}


def make_hist(mass,num,test):
 length_cut=50.0/200.0
 f = plt.figure()
 mass=np.log10(mass)
 for i in range(len(mass)):
  if mass[i]==-np.inf:
    mass[i]=0.0

 hist, bins = np.histogram(mass, bins = 100)
 width = (bins[1]-bins[0])
 hist=np.log10(hist/(width*(num)*length_cut**3))
 for i in range(len(hist)):
  if hist[i]==-np.inf:
    hist[i]=0.0
 center = (bins[:-1]+bins[1:])/2
 plt.step(center, hist, where='mid',color='b')
 plt.title("Hisograma mass function")
 plt.xlabel("log(M)")
 plt.ylabel("dn/dlog(M)")
 f.savefig(test, bbox_inches='tight')
 plt.close(f)
 return 0

def make_hist_mult(mass1,num1,mass2,num2,mass3,num3,mass4,num4):
 length_cut=50.0/200.0
 fig = plt.figure(figsize=(20,20))
 ax1 = plt.subplot2grid((3, 3),(0, 0))
 mass1=np.log10(mass1)
 for i in range(len(mass1)):
  if mass1[i]==-np.inf:
    mass1[i]=0.0

 hist, bins = np.histogram(mass1, bins = 100)
 width = (bins[1]-bins[0])
 hist=np.log10(hist/(width*(num1)*length_cut**3))
 for i in range(len(hist)):
  if hist[i]==-np.inf:
    hist[i]=0.0
 center = (bins[:-1]+bins[1:])/2
 plt.step(center, hist, where='mid',color='b')
 plt.title("Histogram mass function peak") 
 plt.xlabel("log(M)")
 plt.ylabel("dn/dlog(M)")
 
 ax2 = plt.subplot2grid((3, 3),(1, 0))
 mass2=np.log10(mass2)
 for i in range(len(mass2)):
  if mass2[i]==-np.inf:
    mass2[i]=0.0

 hist, bins = np.histogram(mass2, bins = 100)
 width = (bins[1]-bins[0])
 hist=np.log10(hist/(width*(num2)*length_cut**3))
 for i in range(len(hist)):
  if hist[i]==-np.inf:
    hist[i]=0.0
 center = (bins[:-1]+bins[1:])/2
 plt.step(center, hist, where='mid',color='b')
 plt.title("Histograma mass function filament") 
 plt.xlabel("log(M)")
 plt.ylabel("dn/dlog(M)")
 
 ax3 = plt.subplot2grid((3, 3),(0, 1))
 mass3=np.log10(mass3)
 for i in range(len(mass3)):
  if mass3[i]==-np.inf:
    mass3[i]=0.0

 hist, bins = np.histogram(mass3, bins = 100)
 width = (bins[1]-bins[0])
 hist=np.log10(hist/(width*(num3)*length_cut**3))
 for i in range(len(hist)):
  if hist[i]==-np.inf:
    hist[i]=0.0
 center = (bins[:-1]+bins[1:])/2
 plt.step(center, hist, where='mid',color='b')
 plt.title("Histograma mass function sheet") 
 plt.xlabel("log(M)")
 plt.ylabel("dn/dlog(M)")

 ax4 = plt.subplot2grid((3, 3),(1, 1))
 mass4=np.log10(mass4)
 for i in range(len(mass4)):
  if mass4[i]==-np.inf:
    mass4[i]=0.0

 hist, bins = np.histogram(mass4, bins = 100)
 width = (bins[1]-bins[0])
 hist=np.log10(hist/(width*(num4)*length_cut**3))
 for i in range(len(hist)):
  if hist[i]==-np.inf:
    hist[i]=0.0
 center = (bins[:-1]+bins[1:])/2
 plt.step(center, hist, where='mid',color='b')
 plt.title("Histograma mass function void") 
 plt.xlabel("log(M)")
 plt.ylabel("dn/dlog(M)")

 fig.savefig("test.jpg", bbox_inches='tight')
 plt.close(fig)
 return 0






tv_mass_filament=[]
tv_mass_peak=[]
tv_mass_void=[]
tv_mass_sheet=[]
for i in range(199):
 results=read_cut(eigen1,eigen2,eigen3,FA,i)
 results=classify_environment(results['eigen1_cut'],results['eigen2_cut'],results['eigen3_cut'])
 results0=cut_halos(halos,results['matrix_web'],i)
 
 results1=find_mass(results0['halos_ubi'],results['filament_ubi'])
 for j in range(len(results1['mass'])):
  tv_mass_filament.append(results1['mass'][j])
 
 results1=find_mass(results0['halos_ubi'],results['peak_ubi'])
 for j in range(len(results1['mass'])):
  tv_mass_peak.append(results1['mass'][j]) 
 
 results1=find_mass(results0['halos_ubi'],results['sheet_ubi'])
 for j in range(len(results1['mass'])):
  tv_mass_sheet.append(results1['mass'][j])
 
 results1=find_mass(results0['halos_ubi'],results['void_ubi'])
 for j in range(len(results1['mass'])):
  tv_mass_void.append(results1['mass'][j])


print len(tv_mass_filament)
print len(tv_mass_peak)
#make_hist(tv_mass_peak,results['num_all'][0],"test1.jpg")
make_hist_mult(tv_mass_peak,results['num_all'][0],tv_mass_filament,results['num_all'][1],tv_mass_sheet,results['num_all'][2],tv_mass_void,results['num_all'][3])




"""


fig = plt.figure(figsize=(20,20))
ax1 = plt.subplot2grid((2, 2),(1, 0))
plt.imshow(tv_256,extent=[0,50,50,0])
plt.title("Red cosmica (FA) con halos")
plt.xlabel("Mpc/h")
plt.ylabel("Mpc/h")
plt.colorbar()
ax2 = plt.subplot2grid((2, 2),(1, 1))
plt.imshow(image,extent=[0,50,50,0])
plt.title("Red cosmica con halos (Vacios, Hojas, Filamentos y Nudos)")
plt.xlabel("Mpc/h")
plt.ylabel("Mpc/h")
plt.colorbar()
fig.savefig("tv_values.jpg", bbox_inches='tight')
plt.close()


"""

print 'go'
