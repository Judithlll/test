import numpy as np
# from scipy.integrate import odeint
import cgs
import ode
import matplotlib.pyplot as plt
import disk_properties as dp 
import functions as f

class System(object):
    rout=27*cgs.RJ
    alpha=1e-4
    fraction=0.02 
    dgratio=0.01  #dust to gas ratio
    sigmamol=2e-15
    rhoint = 1.4
    tFinal=7e9
    deltaT=0.0

    def __init__(self,Rdi,nini,time=0.0):

        #initialize parameter from txt file // disk.py


        self.Rdi=Rdi  #initial radius of particles
        self.nini=nini
        self.mini = 4*np.pi/3*self.rhoint *Rdi**3
        self.time=time  #initial time
        self.radL=[[] for i in range(len(nini))]
        self.mL=[[] for i in range(len(nini))]
        self.timeL=[]
        # define a disk class

        self.disk = Disk (self.time)

        # define class of superparticles here
        self.particles = Superparticles(100,self.mini,self.disk.rinn,self.disk.rout)

    def Mcp(self):
        Mcp=self.disk.Mcp_t(self.time)
        return Mcp

    def update(self,tEnd):
        """
        How to evolving the gas in the disk is still not sure

        Evolving system to the self.time

        """

        dydtP = self.particles.dY2d_dt(self.particles.Y2d,self.time,self.disk)
            
        # print(len(dydt),len(dydt[0]))
        tscaleArr = np.abs(self.particles.Y2d/dydtP)
        deltaT = 0.2*tscaleArr.min()


        if self.time+deltaT>tEnd:
            deltaT = tEnd - self.time

        #update particle properties
        Yt=self.particles.update(self.time,self.time+deltaT,self.disk)
        self.timeL+=np.linspace(self.time,self.time+deltaT,len(Yt))
        #post_process particles
        daction = self.particles.post_process(self.disk,deltaT,self.Rdi,self.time)

        for i in range(len(self.nini)):
            self.radL[i]+=list(Yt[:,0,i])
            self.mL[i]+=list(Yt[:,1,i])
        
        if 'remove' in daction:
            #remove the particles from Y2d!
            
            self.particles.remove_particles(daction['remove'])
            # import pdb; pdb.set_trace()
        
        #update disk properties (TBD)
        #self.disk.update(deltaT)

        self.time += deltaT
        self.deltaT = deltaT

        return Yt


        


class Disk(object):

    def __init__(self,t):
        
        self.alpha = dp.alpha
        self.rout = dp.rout
        self.rinn = dp.rinn
        self.t=t

    def Mcp_t(self,t):
        return dp.Mcp_t(t)
    
    def dotMg(self):
        return dp.dotMg(self.t)
    
    def dotMd(self):
        return dp.dotMd(self.t)

    def Sigmag (self,r):
        return dp.Sigmag (r,self.t)
    
    def OmegaK(self,r):
        return dp.OmegaK(r)
    
    def Td(self,r):
        return dp.Td(r,self.t)
    
    def cs(self,r):
        return dp.cs(r,self.t)
    
    def vth(self,r):
        return dp.vth(r,self.t)
    
    def Hg(self,r):
        return dp.Hg(r,self.t)
    
    def nu(self,r):
        return dp.nu(r,self.t)
    
    def rhog(self,r):
        return dp.rhog(r,self.t)
    
    def lmfp(self,r):
        return dp.lmfp(r,self.t)
    
    def vK(self,r):
        return dp.vK(r)
    
    def eta(self,r):
        return dp.eta(r,self.t)
    
    def update(self,deltaT):
        pass


class Superparticles(object):
    rhoint=1.4
    pi=np.pi
    Sto=0.0001  
    error=1e-8


    def __init__(self,nini,mini,rinn,rout):
        """
        systems initial properties

        nini: initial number of the particles
        mini: initial mass for every particles
        rinn: inner edge of the disk
        rout: outer edge of the disk
        """
        self.nini=nini
        self.mini=mini
        self.rinn=rinn
        self.rout=rout

        ndim = 3# location // mass // total mass

        self.Y2d = np.empty((ndim,nini))
        self.Y2d[0] = 10**np.linspace(np.log10(rinn),np.log10(rout),nini)
        self.Y2d[1] = mini
        self.Y2d[2] = nini*mini

    def dY2d_dt (self,Y2d,t,disk):
        """
        input:
            Y2d -- state vector
            time -- time
            disk -- disk object
        """

        #unpack the state vector
        r, mphy, mtot = self.Y2d   #maybe the total mass needn't to be put in Y2d

        Rd=(mphy/(self.rhoint*4/3*np.pi))**(1/3)

        eta=disk.eta(r)
        v_K=disk.vK(r)
        v_th=disk.vth(r)
        lmfp=disk.lmfp(r)
        rho_g=disk.rhog(r)
        Omega_K=disk.OmegaK(r)
        H_g=disk.Hg(r)
        dotMd=disk.dotMd()

        St,v_r = f.St_iterate(eta,v_K,v_th,lmfp,rho_g,Omega_K,Rd)

        v_dd=np.abs(v_r)/2
        H_d=H_g*(1+St/disk.alpha*(1+2*St)/(1+St))**(-0.5)

            

        drdt = v_r
        #dR_ddt= v_dd*dot_M_d/4/pi**(3/2)/rho_int/H_d/r/v_r**2 *dr_dt

        sigD = dotMd /(-2*r*self.pi*v_r) #v_r<0
        dR_ddt = 2*np.sqrt(self.pi) /(4*self.pi*self.rhoint) *v_dd *sigD /H_d #eq. 5 of Shibaike et al. 2017

        dmdt=self.rhoint*4*np.pi*Rd**2*dR_ddt

        # assert(dR_ddt>=0)

        #import pdb; pdb.set_trace()

        Y2ddt = np.zeros_like(self.Y2d)
        Y2ddt[0] = drdt
        Y2ddt[1] = dmdt
        Y2ddt[2] = 0.0

        return Y2ddt 
    

    def particles_update(self,y0):
        """
        update the particles list by using remove and add function 
        """
        y0=self.remove(y0)
        y0=self.add(y0)
        return y0
    
    def update(self,t0,tFi,disk,nstep=10):
        """
        this integrate the particles until tFi
        -- d: disk object
        """

        tSpan=np.array([t0,tFi])
        step=(tFi-t0)/nstep #why 100?
    
        Y2copy = np.copy(self.Y2d)

        #integrates system to tFi
        Yt = ode.ode(self.dY2d_dt,Y2copy,tSpan,step,'RK5',disk)

        #some checks perhaps on Yt
        self.Y2d = Yt[-1,:,:]

        return Yt
    
    def remove_particles(self,remove_idx):
        for i in range(len(self.Y2d)):
            self.Y2d[i][remove_idx]=np.nan

    def add_particles(self,add_number):
        lociL=np.linspace(self.rout,self.rout,add_number)
        miL=np.linspace(self.mini,self.mini,add_number)
        mtot=self.Y2d[2][0]+add_number*self.mini
        mtotL=np.linspace(mtot,mtot,add_number)
        self.Y2d=np.append(self.Y2d,np.array([lociL,miL,mtotL]),1)
        
    
    def post_process (self,disk,deltaT,Rdi,t):

        daction = {}

        loc = self.Y2d[0]

        #particles that cross the inner disk edge
        idx, = (loc<6*cgs.RJ).nonzero()
        if len(idx)>0:
            daction['remove'] = idx

        numi=int(disk.dotMg(t)*deltaT/(self.rhoint*4/3*np.pi*Rdi**3))
        daction['add']=numi
        #particles that are eaten by the planet
        #....

        #particles that have become too big
        #....

        #add particles from out CSD



        #remove the particles which location is smaller than RJ
        #Y=Yt[-1]
        #p_removed=np.array([Y[0][Y[0]>cgs.RJ],Y[1][Y[0]>cgs.RJ],Y[2][Y[0]>cgs.RJ]])

        return daction
    

    
    def add(self,deltaT,MassFlow,Rdi):
        num_add=MassFlow*deltaT/(4/3*np.pi*Rdi**3*self.rhoint)
        self.Y2d

        return