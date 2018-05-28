from numpy import linalg as LA
from sympy import *
import matplotlib.pyplot as plt
import numpy as np
from read_cubes import e1,e2,e3,l1,l2,l3
import time
start = time.time()
path="/hpcfs/home/sc.sanabria1984/momografia/rockstar/Rockstar-0.99.9-RC3/HALOS_50_512/out_0.list"
cata = np.loadtxt(path)
print l1.shape
longitud=len(l1)
tamano=50.0
h=np.zeros((longitud,longitud,longitud))
num_halos=len(cata[:,0])
h=np.array(h,object)
tv=np.zeros((longitud,longitud,longitud))
print num_halos
 	
	

#########################################################################################################################
def find_halos(h):
	cut=tamano/longitud
	x=[]
	y=[]
	for i in range(longitud):
		h_x=[]
		for n in range(num_halos):
			if i*cut<cata[n,8] and cata[n,8]<(i+1)*cut :
				h_x.append(n)
		for j in range(longitud):
			h_y=[]
			for l in range(len(h_x)):
				if j*cut<cata[h_x[l],9] and cata[h_x[l],9]<(j+1)*cut:
					h_y.append(h_x[l])
			for k in range(longitud):
				h_cub=[]
				for m in range(len(h_y)):
					if k*cut<cata[h_y[m],10] and cata[h_y[m],10]<(k+1)*cut:
						h_cub.append(h_y[m])
						
						#print i,j,k,i*cut,(i+1)*cut,j*cut,(j+1)*cut,k*cut,(k+1)*cut,cata[h_y[n],8],cata[h_y[n],9],cata[h_y[n],10]
				h[i,j,k]=h_cub
        return 0
 
#################################################################################################################
def find_tv(tv):
	num_v=0
	num_h=0
	num_f=0
	num_p=0
	for i in range(longitud):
		for j in range(longitud):
			for k in range(longitud):
				count=0	 
				a=l1[i,j,k]
				b=l2[i,j,k]
				c=l3[i,j,k]
				if a>0.3:
					count+=1
				if b>0.3:
					count+=1
    				if c>0.3:
     					count+=1
    				if count==0:
     					tv[i,j,k]=0
					num_v+=1
				if count==1:
					tv[i,j,k]=1
					num_h+=1
				if count==2:
					tv[i,j,k]=2
					num_f+=1
				if count==3:
					tv[i,j,k]=3
      					num_p+=1
	t=num_v+num_h+num_f+num_p
	print "vacio","hojas","filamento","nudos"
	print 100.0*num_v/t,100.0*num_h/t,100.0*num_f/t,100.0*num_p/t
		
	return [num_v,num_h,num_f,num_p]

#############################################################################################################
def plot_halos_tv_x(cut):
	x=[]
	y=[]
        id=[]
	for i in range(longitud):
		for j in range(longitud):
			if h[cut,j,i]!=[]:
				
				for n in range(len(h[cut,j,i])):
					id.append(h[cut,j,i][n])
				
	
 	for n in range(len(id)):
				
		y.append(float(cata[id[n],9]))
		x.append(float(cata[id[n],10]))
	
	
	fig = plt.figure()
	
  	plt.imshow(tv[cut,:,:],extent=[0,50,50,0])
  	plt.xlabel("Mpc/h",fontsize=20)
  	plt.ylabel("Mpc/h",fontsize=20)
  	plt.colorbar()
  	plt.scatter(x,y,s=0.01, c='black')
  	plt.xlim(0,50)
        plt.ylim(0,50)
	plt.title("Web con halos "+"halos="+str(len(y))+" corte x "+str((tamano/longitud)*cut)+"-"+str((tamano/longitud)*(cut+1))+" (Mpc/h)",fontsize=20)
  	fig.savefig("/hpcfs/home/sc.sanabria1984/momografia/run/cut_x/"+"halostv_"+str(cut)+".pdf",bbox_inches='tight')
  	plt.close(fig)
	return 0
#plt.title("Web con halos "+"halos="+str(len(y))+" corte y "+str((tamano/longitud)*cut)+"-"+str((tamano/longitud)*(cut+1)),fontsize=20)
#"/hpcfs/home/sc.sanabria1984/momografia/run/cut_z/"+str(cut)+"_halos_tv.pdf",bbox_inches='tight'
########################################################################################################
def plot_halos_tv_y(cut):
        x=[]
	y=[]
	id=[]
	for i in range(longitud):
                for j in range(longitud):
                        if h[i,cut,j]!=[]:

                                for n in range(len(h[i,cut,j])):
                                        id.append(h[i,cut,j][n])
                                
        
        for n in range(len(id)):

                y.append(float(cata[id[n],8]))
                x.append(float(cata[id[n],10]))

        
        fig = plt.figure()

        plt.imshow(tv[:,cut,:],extent=[0,50,50,0])
        plt.xlabel("Mpc/h",fontsize=20)
        plt.ylabel("Mpc/h",fontsize=20)
	plt.colorbar()
        plt.scatter(x,y,s=0.01, c='black')
        plt.xlim(0,50)
        plt.ylim(0,50)
	plt.title("Web con halos "+"halos="+str(len(y))+" corte y "+str((tamano/longitud)*cut)+"-"+str((tamano/longitud)*(cut+1))+" (Mpc/h)",fontsize=20)
        fig.savefig("/hpcfs/home/sc.sanabria1984/momografia/run/cut_y/"+"halostv_"+str(cut)+".pdf",bbox_inches='tight')
        plt.close(fig)
        return 0


######################################################################################################33
def plot_halos_tv_z(cut):
        x=[]
	y=[]
	id=[]
	for i in range(longitud):
                for j in range(longitud):
                   	  if h[i,j,cut]!=[]:

                                for n in range(len(h[i,j,cut])):
                                        id.append(h[i,j,cut][n])
           
	
        for n in range(len(id)):

                y.append(float(cata[id[n],8]))
                x.append(float(cata[id[n],9]))

	
        fig = plt.figure()
	plt.imshow(tv[:,:,cut],extent=[0,50,50,0])
        plt.xlabel("Mpc/h",fontsize=20)
        plt.ylabel("Mpc/h",fontsize=20)
        plt.colorbar()
        plt.scatter(x,y,s=0.01, c='black')
	plt.xlim(0,50)
	plt.ylim(0,50)
        plt.title("Web con halos "+"halos="+str(len(y))+" corte z "+str((tamano/longitud)*cut)+"-"+str((tamano/longitud)*(cut+1))+" (Mpc/h)",fontsize=20)
        fig.savefig("/hpcfs/home/sc.sanabria1984/momografia/run/cut_z/"+"halostv_"+str(cut)+".pdf",bbox_inches='tight')
        plt.close(fig)
	return 0
###################################################################################################################3333
def mass_fun(datos,num):
	log_M=[]
	den=[]
	datos=np.log10(datos)
	hist, bins = np.histogram(datos, bins = 50)
	width = (bins[1]-bins[0])
	
	for i in range(len(hist)):
		if hist[i] == 0:
			hist[i]=hist[i-1]
	hist=np.log10(hist/(width*(num)*longitud**3))
 	center = (bins[:-1]+bins[1:])/2
	return [center,hist]


#######################################################################################################
def find_ambi(num_v,num_h,num_f,num_p):
	id_v=[]
	id_h=[]
	id_f=[]
	id_p=[]
	m_v=[]
	m_h=[]
	m_f=[]
	m_p=[]
	m_all=[]
	for i in range(longitud):
                for j in range(longitud):
                        for k in range(longitud):
				if h[i,j,k]!=[]:
					if tv[i,j,k]==0:
						for n in range(len(h[i,j,k])):
							id_v.append(h[i,j,k][n])
												
					if tv[i,j,k]==1:
						for n in range(len(h[i,j,k])):
							id_h.append(h[i,j,k][n])
					if tv[i,j,k]==2:
						for n in range(len(h[i,j,k])):
                                                	id_f.append(h[i,j,k][n])
					if tv[i,j,k]==3:
						for n in range(len(h[i,j,k])):                                        
					        	id_p.append(h[i,j,k][n])                           
	k=len(id_v)+len(id_h)+ len(id_f)+ len(id_p)
	print "halos_vacios","halos_hojas","halos_filamentos","halos_picos"
	print 100.0*len(id_v)/k, 100.0*len(id_h)/k, 100.0*len(id_f)/k, 100.0*len(id_p)/k

        for n in range(len(id_v)):
		m_v.append(float(cata[id_v[n],2]))
		m_all.append(float(cata[id_v[n],2]))
	for n in range(len(id_h)):
                m_h.append(float(cata[id_h[n],2]))
		m_all.append(float(cata[id_h[n],2]))
	for n in range(len(id_f)):
                m_f.append(float(cata[id_f[n],2]))
		m_all.append(float(cata[id_f[n],2]))
	for n in range(len(id_p)):
                m_p.append(float(cata[id_p[n],2]))
		m_all.append(float(cata[id_p[n],2]))
	all=num_v+num_h+num_f+num_p
	
	m_v=mass_fun(m_v,all)
	m_h=mass_fun(m_h,all)
	m_f=mass_fun(m_f,all)
	m_p=mass_fun(m_p,all)
	m_all=mass_fun(m_all,all)
	print np.amax(m_all)

	f = plt.figure()
	
	plt.step(m_v[0],m_v[1], where='mid',color='b',label="vacios")
	plt.step(m_h[0],m_h[1], where='mid',color='g',label="hojas")
	plt.step(m_f[0],m_f[1], where='mid',color='r',label="filamentos")
	plt.step(m_p[0],m_p[1], where='mid',color='y',label="picos")
	plt.step(m_all[0],m_all[1], where='mid',color='k',label="Todo")
	plt.legend()
	plt.title(r"$Funci\'ones$" r" de masa para halos")
	plt.xlabel(r"$log_{10}(M)$" r" " r"$M_{sol}$")
	plt.ylabel(r"$log_{10}(\frac{dn}{dlog(M)})$" r" " r"$h^{3}Mpc^{-3}$")
	f.savefig("fun_ambi.pdf", bbox_inches='tight')
	plt.close(f)

	return 0
########################################################################################################
def find_ambi_cub(num_v,num_h,num_f,num_p):
        id_v=0.0
        id_h=0.0
        id_f=0.0
        id_p=0.0
        m_v=[]
	m_h=[]
	m_f=[]
        m_p=[]
	m_all=[]
        for i in range(longitud):
                for j in range(longitud):
                        for k in range(longitud):
                                if h[i,j,k]!=[]:
                                        if tv[i,j,k]==0:  
						id_v=0.0
						for n in range(len(h[i,j,k])):
							id_v=id_v+float(cata[h[i,j,k][n],2])
                                                m_v.append(id_v)
						m_all.append(id_v)

                                        if tv[i,j,k]==1:
						id_h=0.0
                                                for n in range(len(h[i,j,k])):
                                                        id_h=id_h+float(cata[h[i,j,k][n],2])
						m_h.append(id_h)
						m_all.append(id_h)

                                        if tv[i,j,k]==2:
						id_f=0.0
                                                for n in range(len(h[i,j,k])):
                                                        id_f=id_f+float(cata[h[i,j,k][n],2])
						m_f.append(id_f)
						m_all.append(id_f)

                                        if tv[i,j,k]==3:
						id_p=0.0
                                                for n in range(len(h[i,j,k])):
							id_p=id_p+float(cata[h[i,j,k][n],2])
						m_p.append(id_p)
						m_all.append(id_p)


	#print "halos_vacios","halos_hojas","halos_filamentos","halos_picos"
       	#print len(id_v), len(id_h), len(id_f), len(id_p)

        all=num_v+num_h+num_f+num_p

        m_v=mass_fun(m_v,all)
        m_h=mass_fun(m_h,all)
        m_f=mass_fun(m_f,all)
        m_p=mass_fun(m_p,all)
        m_all=mass_fun(m_all,all)
	print np.amax(m_all)
        f = plt.figure()

        plt.step(m_v[0],m_v[1], where='mid',color='b',label="vacios")
        plt.step(m_h[0],m_h[1], where='mid',color='g',label="hojas")
        plt.step(m_f[0],m_f[1], where='mid',color='r',label="filamentos")
        plt.step(m_p[0],m_p[1], where='mid',color='y',label="picos")
        plt.step(m_all[0],m_all[1], where='mid',color='k',label="Todo")
        plt.legend()
        plt.title(r"$Funci\'ones$" r" de masa de ambientes")
        plt.xlabel(r"$log_{10}(M)$" r" " r"$M_{sol}$")
        plt.ylabel(r"$log_{10}(\frac{dn}{dlog(M)})$" r" " r"$h^{3}Mpc^{-3}$")
	f.savefig("fun_ambi_cub.pdf", bbox_inches='tight')
        plt.close(f)

        return 0


#########################################################################################################
def centro_m(halos_scq):
	rcm=0.0
	M=0.0
	X=0.0
	Y=0.0
	Z=0.0
	Vx=0.0
	Vy=0.0
	Vz=0.0
	for i in range(len(halos_scq)):

		n=halos_scq[i]
		m=float(cata[n,2])
		x=float(cata[n,8])
		y=float(cata[n,9])
		z=float(cata[n,10])
		vx=float(cata[n,11])
		vy=float(cata[n,12])
		vz=float(cata[n,13])
		X=X+m*x
		Y=Y+m*y
		Z=Z+m*z
		Vx=Vx+m*vx
		Vy=Vy+m*vy
		Vz=Vz+m*vz
		M=M+m
	rcm=[X/M,Y/M,Z/M]
	vcm=[Vx/M,Vy/M,Vz/M]
	return [rcm,vcm]

###################################################################################################
def inertia_vector(halos_scq,rcm):
	r=[]
	R=[]

	I=np.zeros((3,3))
	for i in range(len(halos_scq)):
		n=halos_scq[i]
		x=rcm[0]-float(cata[n,8])
		y=rcm[1]-float(cata[n,9])
		z=rcm[2]-float(cata[n,10])
		r.append([x,y,z])


	for j in range(3):
		for i in range(3):
			for n in range(len(r)):
				I[i][j]=I[i][j]+r[n][i]*r[n][j]/((r[n][0])**2+(r[n][1])**2+(r[n][2])**2)


	w,v = LA.eig(I)
	v_0=v[:,0]/LA.norm(v[:,0])
	v_1=v[:,1]/LA.norm(v[:,1])
	v_2=v[:,2]/LA.norm(v[:,2])
	a={w[0]:v_0,w[1]:v_1,w[2]:v_2}
	w.sort()

 	return [a[w[0]],a[w[1]],a[w[2]]]



######################################################################################################
def peso_est(datos,nombre1,nombre2):
	Y=np.linspace(0.0, 1.0, len(datos))
	Y2=np.linspace(0.0, 1.0, len(datos))
	X2=np.linspace(0.0, 1.0, len(datos))
	datos.sort()
	print np.median(datos),nombre1,nombre2
	fig = plt.figure()
	plt.plot(datos,Y,'r')
	plt.plot(X2,Y2,'k--', linewidth=0.5)
	plt.title("Lineamientos"+" "+nombre1+ " con " +nombre2)
	print "hola"
	plt.xlabel(r"$\mid \cos \theta \mid$")
	plt.axis([0, 1, 0, 1])
	plt.ylabel(r"$Fracci\'on ( \> \mid \cos \theta \mid )$")
	fig.savefig("/hpcfs/home/sc.sanabria1984/momografia/run/lin/"+"lineamiento"+ nombre1 +"_"+ nombre2 + ".jpg", bbox_inches='tight')
	plt.close(fig)
	return [datos,Y]

#####################################################################################################
def lineamientos():
	Imenor_e1=[]
	Imenor_e2=[]
	Imenor_e3=[]
	Imedio_e1=[]
	Imedio_e2=[]
	Imedio_e3=[]
	Imayor_e1=[]
	Imayor_e2=[]
        Imayor_e3=[]
	vcm_e1=[]
	vcm_e2=[]
	vcm_e3=[]

	for i in range(longitud):
                for j in range(longitud):
                        for k in range(0,30):
                                if len(h[i,j,k])>2:
					#m=0.0
					#for n in range(len(h[i,j,k])):
					#	m=m+float(cata[h[i,j,k][n],2])
					#if m<10**10 :
					#if m>10**10 and m<12**10:
					#if m>12**10:
						c=centro_m(h[i,j,k])
						cm=c[0]
						I=inertia_vector(h[i,j,k],cm)
						vcm=c[1]/LA.norm(c[1])
						v1=[e1[0,i,j,k],e1[1,i,j,k],e1[2,i,j,k]]
						v2=[e2[0,i,j,k],e2[1,i,j,k],e2[2,i,j,k]]
						v3=[e3[0,i,j,k],e3[1,i,j,k],e3[2,i,j,k]]
						v1=v1/LA.norm(v1)
						v2=v2/LA.norm(v2)
						v3=v3/LA.norm(v3)
						Imenor_e1.append(np.abs(np.dot(I[0],v1)))
						Imenor_e2.append(np.abs(np.dot(I[0],v2)))
						Imenor_e3.append(np.abs(np.dot(I[0],v3)))
						Imedio_e1.append(np.abs(np.dot(I[1],v1)))
						Imedio_e2.append(np.abs(np.dot(I[1],v2)))
						Imedio_e3.append(np.abs(np.dot(I[1],v3)))
						Imayor_e1.append(np.abs(np.dot(I[2],v1)))
						Imayor_e2.append(np.abs(np.dot(I[2],v2)))
						Imayor_e3.append(np.abs(np.dot(I[2],v3)))
						vcm_e1.append(np.abs(np.dot(vcm,v1)))
						vcm_e2.append(np.abs(np.dot(vcm,v2)))
						vcm_e3.append(np.abs(np.dot(vcm,v3)))
	
	vcm_e1=peso_est(vcm_e1,"vcm","e1")	
	vcm_e2=peso_est(vcm_e2,"vcm","e2")
	vcm_e3=peso_est(vcm_e3,"vcm","e3")
	Imenor_e1=peso_est(Imenor_e1,"I_menor","e1")
	Imedio_e1=peso_est(Imedio_e1,"I_medio","e1")
	Imayor_e1=peso_est(Imayor_e1,"I_mayor","e1")
	Imenor_e2=peso_est(Imenor_e2,"I_menor","e2")
	Imedio_e2=peso_est(Imedio_e2,"I_medio","e2")
	Imayor_e2=peso_est(Imayor_e2,"I_mayor","e2")
	Imenor_e3=peso_est(Imenor_e3,"I_menor","e3")
	Imedio_e3=peso_est(Imedio_e3,"I_medio","e3")
	Imayor_e3=peso_est(Imayor_e3,"I_mayor","e3")	
	
        Y2=np.linspace(0.0, 1.0, 10)
        X2=np.linspace(0.0, 1.0, 10)
        
        fig = plt.figure()

        plt.plot(vcm_e1[0],vcm_e1[1],'r',linestyle='--',label=r"$v_{cm}$" r" y " r"$e_{1}$")
        plt.plot(vcm_e2[0],vcm_e2[1],'y',linestyle='--',label=r"$v_{cm}$" r" y " r"$e_{2}$")
	plt.plot(vcm_e3[0],vcm_e3[1],'b',linestyle='--',label=r"$v_{cm}$" r" y " r"$e_{3}$")
	plt.plot(X2,Y2,'k-', linewidth=0.5,label="ninguno")
        plt.legend()
        plt.title(r"Lineamientos con " r"$v_{cm}$")
        plt.xlabel(r"$\mid \cos \theta \mid$")
        plt.axis([0, 1, 0, 1])
        plt.ylabel(r"$Fracci\'on ( < \mid \cos \theta \mid )$")
        fig.savefig("/hpcfs/home/sc.sanabria1984/momografia/run/lin/"+"lineamientosvcm.jpg", bbox_inches='tight')
        plt.close(fig)	

	fig = plt.figure()

	plt.plot(Imenor_e1[0],Imenor_e1[1],color='r',linestyle='--',label=r"$I_{menor}$" r" y " r"$e_{1}$")
	plt.plot(Imenor_e2[0],Imenor_e2[1],color='y',linestyle='--',label=r"$I_{menor}$" r" y " r"$e_{2}$")
	plt.plot(Imenor_e3[0],Imenor_e3[1],color='b',linestyle='--',label=r"$I_{menor}$" r" y " r"$e_{3}$")	
	plt.plot(X2,Y2,'k-', linewidth=0.5,label="ninguno")
        plt.legend()
        plt.title(r"Lineamientos con " r"$I_{menor}$")
        plt.xlabel(r"$\mid \cos \theta \mid$")
        plt.axis([0, 1, 0, 1])
        plt.ylabel(r"$fracci\'on ( < \mid \cos \theta \mid )$")
        fig.savefig("/hpcfs/home/sc.sanabria1984/momografia/run/lin/"+"lineamientosmenor.jpg", bbox_inches='tight')
        plt.close(fig)	

	fig = plt.figure()
	
	plt.plot(Imedio_e1[0],Imedio_e1[1],color='r',linestyle='--',label=r"$I_{medio}$" r" y " r"$e_{1}$")
	plt.plot(Imedio_e2[0],Imedio_e2[1],color='y',linestyle='--',label=r"$I_{medio}$" r" y " r"$e_{2}$")
	plt.plot(Imedio_e3[0],Imedio_e3[1],color='b',linestyle='--',label=r"$I_{medio}$" r" y " r"$e_{3}$")
	plt.plot(X2,Y2,'k-', linewidth=0.5,label="ninguno")
        plt.legend()
        plt.title(r"Lineamientos con " r"$I_{medio}$")
        plt.xlabel(r"$\mid \cos \theta \mid$")
        plt.axis([0, 1, 0, 1])
        plt.ylabel(r"$fracci\'on ( < \mid \cos \theta \mid )$")
        fig.savefig("/hpcfs/home/sc.sanabria1984/momografia/run/lin/"+"lineamientosImedio.jpg", bbox_inches='tight')
        plt.close(fig)	



	fig = plt.figure()

	plt.plot(Imayor_e1[0],Imayor_e1[1],color='r',linestyle='--',label=r"$I_{mayor}$" r" y " r"$e_{1}$")
	plt.plot(Imayor_e2[0],Imayor_e2[1],color='y',linestyle='--',label=r"$I_{mayor}$" r" y " r"$e_{2}$")
	plt.plot(Imayor_e3[0],Imayor_e3[1],color='b',linestyle='--',label=r"$I_{mayor}$" r" y " r"$e_{3}$")
	plt.plot(X2,Y2,'k-', linewidth=0.5,label="ninguno")
	plt.legend()
        plt.title(r"Lineamientos con " r"$I_{mayor}$")
        plt.xlabel(r"$\mid \cos \theta \mid$")
        plt.axis([0, 1, 0, 1])
        plt.ylabel(r"$fracci\'on ( < \mid \cos \theta \mid )$")
        fig.savefig("/hpcfs/home/sc.sanabria1984/momografia/run/lin/"+"lineamientosImayor.jpg", bbox_inches='tight')
        plt.close(fig)
	return 0




##########################################################################################################

find_halos(h)
a=find_tv(tv)
#find_ambi(a[0],a[1],a[2],a[3])
#find_ambi_cub(a[0],a[1],a[2],a[3])
#for cut in range(longitud):
#	plot_halos_tv_x(cut)
#	plot_halos_tv_y(cut)
#	plot_halos_tv_z(cut)
lineamientos()




end = time.time()
print((end - start)/60.0)
