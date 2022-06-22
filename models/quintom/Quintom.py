

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.integrate import odeint
#import os
from scipy import optimize
import matplotlib as mpl




# TODO-me Can be chosen LCDM or SFDM
# TODO-me Build a function for interpolation
# TODO-me checar condiciones iniciales

"Define rho_DE as function "


class Quintom:
    def __init__(self, name='Alberto'):

        self.name   = name

        self.vary_mquin = True
        self.vary_mphan = False

        self.mquin  = 0
        self.mphan  = 0
        self.beta   = 0

        self.delta  = 0
        self.theta  = 0
        self.imphan = 0

        self.H0     = 0.68
        self.Ocb    = 0.3
        self.Omrad  = 0.0001
        self.Odeobs = 1-self.Ocb-self.Omrad

        self.lna    = np.linspace(-15, 0, 300)
        self.z      = np.exp(-self.lna) - 1.
        self.zvals  = np.linspace(0, 4, 300)

        self.cte    = 3*self.H0**2
        #self.n      = 2
        #self.m      = 2

        self.min    = 0.1
        self.max    = 4.
        self.steps  = 6.

    def something(self):
        print (self.name)


    def Vtotal(self, x, y, select):
        """Cuadratic potential and its derivatives wrt phi or psi"""
        if select == 0:
            Vtotal = 0.5*(x*self.mquin)**2  +  0.5*(y*self.mphan)**2 + self.beta*(x*y)**2.
            #Vtotal = self.mquin**2*(1-np.cos(x*1.))  +  0.5*(y*self.mphan)**2 + self.beta*(x*y)**2.
            #Vtotal = 0.5*(x*self.mquin)**2 + self.mphan**2*(1-np.cos(y*a)) + self.beta*(x*y)**2.
            #Vtotal = self.mquin**2*x**2 + self.delta*np.cos(2*x) #self.mquin**2*(1-np.cos(x*self.delta))  + self.mphan**2*(1-np.cos(y*self.theta))
        elif select == 'phi':
            #Vtotal = x*self.mquin**2 + 2.0*self.beta*x*y**2
            #Vtotal = self.mquin**2*np.sin(x*1.)+ 2.0*self.beta*x*y**2
            Vtotal = 2*self.mquin**2*x  - 2*self.delta*np.sin(2*x) #self.mquin**2*self.delta*np.sin(x*self.delta)
        elif select == 'psi':
            #Vtotal = y*self.mphan**2 + 2.0*self.beta*y*x**2
            #Vtotal = self.mphan**2*a*np.sin(y*a) + 2.0*self.beta*y*x**2
            Vtotal = self.mphan**2*self.theta*np.sin(2*y)
        return Vtotal


    def rhode(self, x_vec):
        """
        Esta funcion calcula rho

        :param x_vec: quinte, phan
        :return: rho
        """
        quin, dotquin, phan, dotphan = x_vec
        Ode = 0.5*dotquin**2 - 0.5*dotphan**2 + self.Vtotal(quin, phan, 0) / self.cte
        return Ode


    def hubble(self, lna, x_vec=None, SF=True):
        a = np.exp(lna)
        if SF:
            Ode  = self.rhode(x_vec)
        else:
            Ode  = self.Odeobs
        return self.H0*np.sqrt(np.abs(self.Ocb/a**3 + self.Omrad/a**4 + Ode))


    def logatoz(self, func):
        "change functions from lna to z "
        tmp     = interp1d(self.lna, func)
        functmp = tmp(self.lna)
        return  np.interp(self.zvals, self.z[::-1], functmp[::-1])


    def RHS(self, x_vec, lna):
        """Rigth hand side of the dynamical systems of equations"""
        sqrt3H0  = np.sqrt(self.cte)
        quin, dotquin, phan, dotphan = x_vec
        hubble = self.hubble(lna, x_vec)
        return [sqrt3H0*dotquin/hubble, -3*dotquin - self.Vtotal(quin, phan, 'phi')/(sqrt3H0*hubble),
                sqrt3H0*dotphan/hubble, -3*dotphan + self.Vtotal(quin, phan, 'psi')/(sqrt3H0*hubble)]



    def solver(self, quin0, dotquin0, phan0, dotphan0):
        """
        Solve the system using initial condition as an array
        :param quin0:
        :param dotquin0:
        :param phan0:
        :param dotphan0:
        :return:
        """

        y0       = [quin0, dotquin0, phan0, dotphan0]
        y_result = odeint(self.RHS, y0, self.lna, h0=1E-10)
        return y_result



    def calc_Ode(self, mid):
        if (self.vary_mquin) and (self.mphan == 0) and (self.beta ==0): quin0, phan0 = mid, 0
        elif (self.vary_mphan) and (self.mquin == 0) and (self.beta ==0): quin0, phan0 = 0, mid
        else :
            quin0, phan0 = mid, mid*self.imphan

        #if    (self.mphan == 0) and (self.beta == 0):  quin0, phan0 = mid, 0
        #elif  (self.mquin == 0) and (self.beta == 0):  quin0, phan0 = 0, mid
        #else: quin0, phan0 = mid, mid*1.1
        #print ('**mid', mid)
            #np.sqrt(2*3*self.H0**2*(1-self.Ocb) + (self.mphan*mid)**2)/self.mquin, mid
            #"Still figuring out these initial conditions."

        sol = self.solver(quin0, 0.0, phan0, 0.0).T

        quin, dotq, phan, dotp = sol

        rho =  self.rhode(sol)[-1]
        Ode = rho*(self.H0/self.hubble(0.0, [quin[-1], dotq[-1], phan[-1], dotp[-1]]))**2
        tol = self.Odeobs - Ode
        return sol, tol, Ode


    def bisection(self):
        "Search for intial condition of phi such that \O_DE today is 0.7"
        lowphi, highphi = 0, 3 # 30
        Ttol            = 1E-2
        mid =(lowphi + highphi)/2.0
        while (highphi - lowphi )/2.0 > Ttol:
            sol, tol_mid, Ode = self.calc_Ode(mid)
            tol_low = self.calc_Ode(lowphi)[1]
            if(np.abs(tol_mid) < Ttol):
                #print ('reach tolerance',  'phi_0=', mid, 'error=', tol_mid, 'Ode', Ode)
                return mid, sol
            elif tol_low*tol_mid<0:
                highphi  = mid
            else:
                lowphi   = mid
            mid = (lowphi + highphi)/2.0
        print (mid)
        ##Check whether rho is constant or nearby
        grad = np.gradient(self.rhode(sol)).max()
        mid  = -1 if grad < 1.0E-2 else 0
        return mid, False


    def search_ini(self):
        mid, sol = self.bisection()

        if (mid is not 0) and (mid is not -1):
            self.hub_SF    = interp1d(self.lna, self.hubble(self.lna, sol))
            self.hub_SF_z  = self.logatoz(self.hubble(self.lna, sol))
            self.solution  = sol
        return mid


    def mu(self, z, hub):
        " Useful for plotting SN "
        tmp = interp1d(self.zvals, 1./hub)
        xi  = np.array([quad(tmp, 0.0, i)[0] for i in z])
        dL  = (1+z)*xi
        return 5*np.log10(dL) + 52.5 #"Checar este numero"


    #   Plots
    #================================================================================


    def plot_Vomegas(self):
        a    = np.exp(self.lna)
        X =[]
        Y =[]
        P =[]
        zz=[]
        hh=[]
        ww = []
        qq =[]
        pp =[]
        rr = []
        ss = []
        Oma=[]
        vphi = []
        dphi = []

        min, max = (self.min, self.max)
        step     = (max-min)/self.steps

        mymap    = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['blue','red'])
        Z        = [[0,0],[0,0]]
        levels   = np.arange(min, max+step, step)
        CS3      = plt.contourf(Z, levels, cmap=mymap)

        for i in np.arange(min, max, step):
            if (self.vary_mquin):
                    self.mquin = i
            elif (self.vary_mphan):
                    self.mphan = i
            else:  break

            flag  = self.search_ini()

            X.append(self.lna)
            zz.append(self.zvals)
            P.append(i)

            if flag == 0:
                print ('mass=', i, 'solution not found')
                continue
            elif flag == -1:
                print ('mass=', i, 'is a cosmological constant')
                hub_SF_z = self.logatoz(self.hubble(self.lna, SF=False))
                Omde     = self.Odeobs*(self.H0/self.hubble(self.lna, SF=False))**2
                w1       = -1.*np.ones(len(self.lna))
                w2       =  1.*np.ones(len(self.lna))
                Omatter  = self.Ocb/a**3*(self.H0/self.hubble(self.lna, SF=False))**2

                Y.append(Omde)
                hh.append(100*hub_SF_z)
                ww.append(self.logatoz(w1/w2))
                Oma.append(Omatter)
                vphi.append(self.logatoz(w1))
                dphi.append(self.logatoz(w2))

            else:
                scalars = self.solution
                quin, dquin, phan, dphan = scalars

                hub_SF_z = self.hub_SF_z
                rho      = self.rhode(scalars)
                Omde     = rho*(self.H0/self.hubble(self.lna, scalars))**2
                w1       = 0.5*dquin**2 - 0.5*dphan**2 - self.Vtotal(quin, phan, 0)/self.cte
                w2       = rho
                Omatter  = self.Ocb/a**3*(self.H0/self.hubble(self.lna, scalars))**2

                Y.append(Omde)
                hh.append(100*hub_SF_z)
                ww.append(self.logatoz(w1/w2))
                Oma.append(Omatter)
                vphi.append(self.logatoz(w1))
                dphi.append(self.logatoz(w2))
                pp.append(self.logatoz(quin)) #dquin*self.Vtotal(quin, phan, 'phi')/self.cte +
                qq.append(self.logatoz(phan))
                rr.append(self.logatoz(dquin))#0.5*dquin**2 - 0.5*dphan**2))
                ss.append(self.logatoz(dphan)) #self.Vtotal(quin, phan, 'phi')/self.cte - self.Vtotal(quin, phan, 'psi')/self.cte))
                                       #dphan*self.Vtotal(quin, phan, 'psi')/self.cte))

        fig = plt.figure(figsize=(9,10))
        ax1 = fig.add_subplot(311)
        for x,w,z, in zip(zz, ww, P):
            r    = (float(z)-min)/(max-min)
            g, b = 0, 1-r
            #ax1.plot(x, y , color=(r,g,b))
            #ax1.plot(x, v , color=(r,g,b), linestyle='--')
            ax1.plot(x, w , color=(r,g,b)) #, linestyle='-.')
        plt.axhline(y=-1, color='k', linestyle='--')
        #plt.axhline(y=0, color='k', linestyle='--')
        ax1.legend(loc='lower right', frameon=False)
        #ax1.set_ylim(-2, 2)


        if self.vary_mquin == True:
            plt.title('Quintom, $m_{\psi}$=%.1f, $\\beta$=%.1f'%(self.mphan, self.beta), fontsize=20)
        elif self.vary_mphan == True:
            plt.title('Quintom, $m_{\phi}$=%.1f, $\\beta$=%.1f'%(self.mquin, self.beta), fontsize=20)

        plt.ylabel('$w_{\phi, \psi}$', fontsize=20)
        cbaxes = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        cbar = plt.colorbar(CS3, cax=cbaxes)

        if self.vary_mquin == True:
            cbar.set_label('$m_\phi$', rotation=0, fontsize=20)
        elif self.vary_mphan == True:
            cbar.set_label('$m_\psi$', rotation=0, fontsize=20)




        ax2 = fig.add_subplot(312)
        for x,y,z in zip(zz,hh,P):
            r = (float(z)-min)/(max-min)
            g,b = 0, 1-r
            ax2.plot(x,y,color=(r,g,b))
        ax2.plot(self.zvals, 100*self.logatoz(self.hubble(self.lna, SF=False)), 'o',  markersize=2)
        dataHz = np.loadtxt('Hz_all.dat')
        redshifts, obs, errors = [dataHz[:,i] for i in [0,1,2]]
        ax2.errorbar(redshifts, obs, errors, xerr=None,
                color='purple', marker='o', ls='None',
                elinewidth =2, capsize=3, capthick = 1)
        ax2.legend(loc='lower right', frameon=False)
        plt.ylabel('$H(z)$', fontsize=20)
        plt.xlabel('redshift $z$', fontsize=20)


        ax3 = fig.add_subplot(313)
        ax3.plot(self.lna, self.Ocb/a**3*(self.H0/self.hubble(self.lna, SF=False))**2,   'o',  markersize=2)
        ax3.plot(self.lna, self.Omrad/a**4*(self.H0/self.hubble(self.lna, SF=False))**2, 'o',  markersize=2)
        ax3.plot(self.lna, self.Odeobs*(self.H0/self.hubble(self.lna, SF=False))**2,     'o',  markersize=2)
        for x,y,z in zip(X,Y,P):
            r = (float(z)-min)/(max-min)
            g, b = 0, 1-r
            ax3.plot(x,y,color=(r,g,b))

        for x,y,z in zip(X,Oma,P):
            r = (float(z)-min)/(max-min)
            g, b = 0, 1-r
            ax3.plot(x,y,color=(r,g,b))

        plt.xlabel('$\ln a$', fontsize=20)
        plt.ylabel('$\Omega(a)$', fontsize=20)
        #plt.savefig('Omega_phi_%.1f_psi_%.1f.pdf'%(self.mquin, self.mphan))
        plt.savefig('Quintom_1_cos.pdf')
        plt.show()


        fig = plt.figure(figsize=(9,10))
        ax1 = fig.add_subplot(211)
        for x,y, v,s, t, z, in zip(zz, pp, qq, rr, ss, P):
            r    = (float(z)-min)/(max-min)
            g, b = 0, 1-r
            ax1.plot(x, y , color=(r,g,b))
            ax1.plot(x, v , color=(r,g,b), linestyle='--')
            #ax1.plot(x, s , color=(r,g,b), linestyle='-.')
            #ax1.plot(x, t , color=(r,g,b), linestyle=':')
        #plt.axhline(y=-1, color='k', linestyle='--')
        plt.ylabel('$\phi$ and $\psi$', fontsize=20)
        plt.axhline(y=0, color='k', linestyle='--')
        ax1.legend(loc='lower right', frameon=False)

        ax2 = fig.add_subplot(212)
        for x,y, v,s, t, z, in zip(zz, pp, qq, rr, ss, P):
            r    = (float(z)-min)/(max-min)
            g, b = 0, 1-r
            #ax1.plot(x, y , color=(r,g,b))
            #ax1.plot(x, v , color=(r,g,b), linestyle='--')
            ax2.plot(x, s , color=(r,g,b), linestyle='-.')
            ax2.plot(x, t , color=(r,g,b), linestyle=':')
        #plt.axhline(y=-1, color='k', linestyle='--')
        plt.ylabel('derivatives', fontsize=20)
        plt.axhline(y=0, color='k', linestyle='--')
        ax1.legend(loc='lower right', frameon=False)
        plt.show()



    def plot_potential(self, select=0):
        "Only the potentials in 3D"
        quin = np.arange(-1., 1., 0.1)
        phan = np.arange(-1.5, 1.5, 0.1)
        quin, phan = np.meshgrid(quin, phan)
        Pot = self.Vtotal(quin, phan, select)

        fig = plt.figure()
        ax = Axes3D(fig)
        surf = ax.plot_surface(quin, phan, Pot, cmap=cm.RdYlGn, #bwr,
                               linewidth=0, antialiased=False)
        #cset = ax.contourf(quin, phan, Pot, zdir='z', offset=-0.2, cmap=cm.RdYlGn)# cm.coolwarm)
        cset = ax.contourf(quin, phan, Pot, zdir='z', offset=-2, cmap=cm.RdYlGn)# cm.coolwarm)
        #cset = ax.contour(quin, phan, Pot, zdir='x', offset=-40, cmap=cm.hsv) # cm.coolwarm)
        #cset = ax.contour(quin, phan, Pot, zdir='y', offset=40, cmap=cm.hsv) #cm.coolwarm)

        ax.set_zlim(-2, 1)
        plt.xticks(np.arange(-1, 0.75, 0.5), fontsize=15)
        plt.yticks(fontsize=15)
        for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(15)
        plt.ylabel('$\phi$', fontsize=20, labelpad=10)
        plt.xlabel('$\psi$', fontsize=20, labelpad=10)
        plt.title('$m_\phi$ = %.1f, $m_\psi$=%.1f,  $\\beta$=%.1f'%(self.mquin, self.mphan, self.beta),
                  y=1.0, pad=0.5, fontsize=20)
        cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
        cbar.ax.tick_params(labelsize=12)
        plt.savefig('Potential_Quintom_Phan_phi_%.1f_psi_%.1f.pdf'%(self.mquin, self.mphan))
        # plt.savefig('Quintom_potential.pdf')
        plt.show()



    def plot_hubble(self):
        "Plot only Hubble data"
        hub_CDM    = self.logatoz(self.hubble(self.lna))
        plt.plot(self.zvals, self.hub_SF_z)
        plt.plot(self.zvals, hub_CDM, 'o',  markersize=2)

        dataHz = np.loadtxt('Hz_all.dat')
        redshifts, obs, errors = [dataHz[:,i] for i in [0,1,2]]
        plt.errorbar(redshifts, obs, errors, xerr=None,
                color='purple', marker='o', ls='None',
                elinewidth =2, capsize=5, capthick = 1, label='$Datos$')
        plt.xlabel(r'$z$')
        plt.ylabel(r'$H(z) [km/s Mpc^{-1}]$')
        plt.show()




    def plot_SN(self):
        "Plot only SN data"
        names = ['name', 'z', 'mu', 'error']
        result = pd.read_table('sn_z_mu_dmu_union2.txt', sep='\s+', names=names, index_col='z')
        result=result.sort_index()
        plt.figure(figsize=(14,7))
        result['mu'].plot(yerr=result['error'], linestyle='None', label = 'SN')

        hub_CDM    = self.logatoz(self.hubble(self.lna))
        mu = self.mu(self.zvals, hub_CDM)
        plt.plot(self.zvals, mu, 'o',  markersize=2, label = '$\Omega_{DM} = 0.24, \Omega_\Lambda=0.76$')

        mu_SF = self.mu(self.zvals, self.hub_SF_z)
        plt.plot(self.zvals, mu_SF, label = 'SF', color = 'r')

        plt.xlabel(r'Corrimiento al rojo - z', fontsize = 20)
        plt.ylabel(r'Distancia modular - $\mu$', fontsize = 20)
        plt.title('Supernova Tipo Ia')
        plt.legend(loc='lower right', frameon=False)
        plt.savefig('SN_models.pdf')
        plt.show()


    def contours(self):
        dir ='/chains/'
        name_in = 'Quintom_coupling_phy_BBAO+Planck_15+HD_'

        file = open('Quintom_cos2_contour.txt','w')
        ztmp = np.arange(0., 2.55, 0.1) #[0, 0.5, 1, 1.5, 2, 2.5, 3.0]

        for j in range(6):
            i = 0
            for line in reversed(open(dir + name_in + '%s.txt'%(j+1)).readlines()):
                if i% 50==1:
                    a= line.split(' ')
                    #print a[5:7]
                    self.mquin = float(a[5]) #map(float, a[5:8])
                    self.mphan = float(a[6])
                    self.beta  = float(a[7])
                    #self.delta = float(a[8])
                    self.H0  = float(a[4])
                    self.Ocb = float(a[2])

                    phi0  = self.search_ini()
                    #print phi0, '****'
                    if phi0 == 0:
                        continue
                    else:
                        scalars = self.solution
                        quin, dquin, phan, dphan = scalars

                        w1 = 0.5*dquin**2 - self.Vtotal(quin, phan, 0)/self.cte - 0.5*dphan**2  - self.Vtotal(quin, phan, 0)/self.cte
                        w2 = 0.5*dquin**2 + self.Vtotal(quin, phan, 0)/self.cte - 0.5*dphan**2  + self.Vtotal(quin, phan, 0)/self.cte

                        bb = interp1d(self.zvals,  self.logatoz(w1/w2), kind='linear')

                        file.write('%s %s %s\n'%(a[0] , ' '.join(map(str, ztmp)) , ' '.join(map(str, bb(ztmp)))))

                    print (j+1, i)
                i+=1
                #if i == 10000: break
                file.flush()
        file.close()


if __name__ == '__main__':
    Q = Quintom()

    # Select varying mass
    Q.vary_mquin = True
    Q.vary_mphan = False
    # if vary_par is True overrides mass of this field turning into a range
    Q.mquin = 1.5
    Q.mphan = 1.0
    Q.beta  = -2.0

    Q.delta = 2.1
    Q.theta = 0.0

    #If two field are present, we have an extra parameters described by
    Q.imphan= 1.

    #Q.plot_Vomegas()



    #print Q.find_Ode()
    #print Q.prueba()
    Q.plot_potential()
    #Q.contours()
    #Q.plot_hubble()
    #Q.plot_SN()

