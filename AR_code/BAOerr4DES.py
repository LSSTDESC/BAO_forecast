#Ashley Ross, version 1.0, essentially copied from code by Will Percival and based off of
#Seo & Eisenstein 2007
#Predictions match others predictions for BOSS and eBOSS
#Includes photoz uncertainty, both on damping of overall signal and also bao feature
#requires Cosmo.py and Cosmo.py require romberg.py

#dz allows one to test multiple redshift shells at once; nshell = (zmax-zmin)/dz
#num is total number of galaxies
#sigz is sigma_z/(1+z)

from math import *
from numpy import loadtxt as load

def Y1tot(zmin,zmax,dz,nz,sz,area):
	#default arguments were zmin=0.6, zmax=0.9
	#projected uncertainty for simple red galaxy selections
	#change zmax to 1 to include 0.9 < z < 1.0 bin

	#nz = [1179061.0,929435.0,442699.0,312904.0] #number of galaxies in each bin
	#sz = [0.03,.04,.04,.05] #approximate sigmaz/(1+z)

	nb = int((zmax+.001-zmin)/dz)
	se = 0
	for i in range(0,nb):
		zm = zmin+i*dz
		zx = zmin+(i+1)*dz
#		print('Y1tot: redshift range ' + str(zm) + ' -> ' + str(zx))
		er =  baoerr(zm,zx,dz,sz[i],area,nz[i],1.6)
#		print("Y1tot: er = " + str(er))
		se += 1./(er**2.)
#		print('')
#	print ("Y1tot: se = " + str(se))
	et = sqrt(1./se)
#	print('Y1tot: combined Y1 error is')
	return et	

def baoerr(zmin,zmax,dz,sigz,area,num,bias,recon_fac=1.,sig8=0.8,dampz='y'):
	#based on Seo & Eisenstein 2007
	from Cosmo import distance
	from numpy import ones
	d = distance(.3,.7)
	
	Pbao_list =  [ 9.034, 14.52, 12.63, 9.481, 7.409, 6.397, 5.688, 4.804, 3.841, 3.108,
    2.707, 2.503, 2.300, 2.014, 1.707, 1.473, 1.338, 1.259, 1.174, 1.061,
    0.9409, 0.8435, 0.7792, 0.7351, 0.6915, 0.6398, 0.5851, 0.5376, 0.5018, 0.4741,
    0.4484, 0.4210, 0.3929, 0.3671, 0.3456, 0.3276, 0.3112, 0.2950, 0.2788, 0.2635,
    0.2499, 0.2379, 0.2270, 0.2165, 0.2062, 0.1965, 0.1876, 0.1794, 0.1718, 0.1646]
	
	BAO_POWER = 0.18961E+04   # /* The power spectrum at k=0.2h Mpc^-1 for sigma8=0.8 and Planck cosmo */
	BAO_SILK = 7.76
	BAO_AMP = 0.5 #approximate, see Seo & Eisenstein for details
	mustep = 0.01
	
	fsky = area/(360*360./pi)
	nz = int((zmax+.0001-zmin)/float(dz))
		#print nz
#	print("baoerr: " +str(nz)+" bin(s)")
	dtot = 0
	for i in range(0,nz):
		z = zmin+i*(zmax-zmin)/float(nz)+.5*(zmax-zmin)/float(nz)
		z1 = z-.5*(zmax-zmin)/float(nz)
		z2 = z+.5*(zmax-zmin)/float(nz)
		sigzdampl = BAOdampsigz(z,sigz)
		dr = d.dc(z2)-d.dc(z1)
		volume= 4./3.*pi*fsky*(d.dc(z2)**3.-d.dc(z1)**3.)/1.e9
		if dampz == 'n':
			sigzdampl = ones((len(Pbao_list)))
		Dg = d.D(z)
		f = d.omz(z)**.557
		beta = f/bias
		Sig0 = 12.4*sig8/0.9*Dg*.758*recon_fac
		Sigma_perp = Sig0
		Sigma_par = Sig0#*(1.+f)
		Sigma_perp2 = Sigma_perp*Sigma_perp
		Sigma_par2 = Sigma_par*Sigma_par
	#	print Sigma_perp,Sigma_par

		Sigma_z = d.cHz(z)*sigz
		Sigma_zb = Sigma_z/d.dc(z)*105. #percentage distance error multiplied by BAO scale
		#print Sigma_zb
		Sigma_z2 = Sigma_z*Sigma_z
		#print Sigma_z2
		sigma8 = bias*Dg
		#print sigma8
		KSTEP = .01
		power = sigma8*sigma8*BAO_POWER
		#print power,sigma8**2.
#		print("baoerr: power = " + str(power) + "  (bias*Dg)**2 = " + str(sigma8**2.))
		numd = num/float(nz)/volume/1.e9 #calculate the number density in the bin; even distribution in redshift is assumed
#		print(numd)        
		nP = numd*power
		#print nP
#Z		print ("baoerr: nP = " + str(nP))
		Silk_list  = []
		for i in range(0,len(Pbao_list)):
			k=0.5*KSTEP+KSTEP*i
			Silk_list.append(exp(-2.0*pow(k*BAO_SILK,1.40))*k*k*sigzdampl[i]**2.)
		mu = .5*mustep
		sumt = 0
		sumW1 = 0
		sumW2 = 0
		monosum = 0
		Fdd = Fdh = Fhh = 0.0
		while mu<1:
			mu2 = mu*mu
			redshift_distort = (1+beta*mu2)*(1+beta*mu2)
			tmp = 1.0/(nP*redshift_distort)
			Sigma2_tot = Sigma_perp2*(1-mu2)+Sigma_par2*mu2+Sigma_zb*Sigma_zb/2.
			sum = 0
			for i in range(0,len(Pbao_list)):
				k=0.5*KSTEP+KSTEP*i
				try:
					#print( "Pbao_list[i]="+str(Pbao_list[i])+" tmp="+str(tmp)+" k="+str(k)+" Sigma_z2="+str(Sigma_z2)+" mu2="+str(mu2))					
					tmpz = Pbao_list[i]+tmp*exp(k*k*Sigma_z2*mu2)
					Fmu = Silk_list[i]*exp(-k*k*Sigma2_tot)/tmpz/tmpz
					sum += Fmu
				except:
					pass
					#print k,mu
			Fdd += sum*(1.-mu2)*(1.-mu2)
			Fdh += sum*(1.-mu2)*mu2
			Fhh += sum*mu2*mu2
			sumt += sum
			monosum += 1./sum
			if mu < 0.5:
				sumW1 += sum
			if mu > 0.5:
				sumW2 += sum
			mu += mustep
		r = Fdh/sqrt(Fhh*Fdd)
		Fdd *= BAO_AMP*BAO_AMP*1.0e9*KSTEP*mustep*volume/2. #must have missed factor of 2 earlier
		Fhh *= BAO_AMP*BAO_AMP*1.0e9*KSTEP*mustep*volume/2.
		sumt *= BAO_AMP*BAO_AMP*1.0e9*KSTEP*mustep*volume/2.
		monosum /= BAO_AMP*BAO_AMP*1.0e9*KSTEP*volume
		monosum *= mustep
		Drms = 1.0/sqrt(Fdd*(1.0-(r)*(r)))
		Hrms = 1.0/sqrt(Fhh*(1.0-(r)*(r)))
		Rrms = (Drms)*sqrt((1-(r)*(r))/(1+(Drms)/(Hrms)*(2*(r)+(Drms)/(Hrms))))
#		print("baoerr: Drms, Hrms, Rrms, r, z1, z2:")
#		print('\t',Drms,Hrms,Rrms,r,z1,z2)
		dtot += sumt
#	print('baoerr: total BAO error '+str(sqrt(1.0/dtot)))
	return sqrt(1.0/dtot)

def BAOdampsigz(z,sigz,nthbin=50,dz=.01,zmax=3.,vis='n'):
	from Cosmo import distance
	from numpy import ones
	d = distance(.3,.7)
	disz = d.dc(z)
	cl  =[]
	cl0 = []
	thl  = []
	dth = 8*2.*pi/float(nthbin)
	for i in range(0,nthbin):
		cl.append(0)
		cl0.append(cos(dth/2.+dth*i))
		thl.append(dth/2.+dth*i)
	nzbin = int((zmax)/dz)
	sum = 0
	
	for i in range(0,nthbin):
		th = dth/2.+dth*i
		c = 0
		for j in range(0,nzbin):
			zj = dz/2.+dz*j
			c += dz/sqrt(2.*pi*sigz**2.)*exp(-1.*(z-zj)**2./(2.*sigz**2.))*cos(th*d.dc(zj)/disz)
		cl[i] = c
		sum += c*c
	sum = sum*dth/pi/3.
	if vis == 'y':
		import matplotlib.pyplot as plt
		plt.plot(thl,cl0,'k-',thl,cl,'r-')
		plt.show()
	outl = []	
	for i in range(0,nthbin):
		outl.append(cl[i]/cl0[i])
	if sigz < 0.01 and sum < .1:
		outl = ones((nthbin))
	return outl
