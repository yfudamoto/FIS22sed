# A1689zD1.f90 -> A1689zD1_bestfit.py
# Dust mass and temperature estimator
# 2018 March 8 by AKI
# 2018 June 6 by AKI: Size update!
# 2019 May 3 by AKI: B8 flux density and size are updated again!
# 2020 March 1 by AKI: Dust mode update.
# 2020 December 25 by Y. Fudamoto: Python version created
#----------------------------------------------------------

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import numpy as np
from astropy import constants as const
from astropy.uncertainty import normal
from scipy.special import zeta
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math, sys, time

t_start = time.time()

# 1. Input Source Properties --------------------------------------------------------------------
zS  =   7.13
muGL    =   9.3e+0                          #! K17 & Watson+15, Magnification
Luv     =   8.0e+43 * (9.3e+0/muGL)           # UV luminosity [erg/s]
eLuv    =   1.0e+43 * (9.3e+0/muGL)           # Uncertainty of UV luminosity [erg/s]
sarc    =   0.46e+0                         # [arcsec]
esarc   =   0.06e+0                         # [arcsec]
lobs    =   np.array([1330, 873, 728,427],np.float)  # [um]: observed wavelength
fnuobs  =   np.array([0.56e+0, 1.33e+0, 1.67e+0,1.43],np.float)  # [mJy] K17, B8: observed fluxes
efnuobs =   np.array([0.10e+0, 0.14e+0, 0.36e+0,0.34],np.float)  # [mJy] K17, B8: observed flux errors

lobs    =   lobs * 1.0e-4

# 2. Select Dust Geometry Model -----------------------------------------------------------------
geo =   2
# 0: Spherical shell, 1: Homogeneous sphere, 2: Clumpy sphere
useband =   0   # 0: ALL, 1: B6, 2: B7, 3: B8
Niter   =   5000
flg_plot=   1
#------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------
#- Script Starts --------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

#- Fitting Parameters ---------------------------------------------------------------------------
# TBD?

#- Define Global Constants -----------------------------------------------------------------------------
PI  =   np.pi
H0  =   7.0e+1           # [km s-1 Mpc-1]: Hubble Constant
OmL =   7.0e-1
OmM =   3.0e-1
hP  =   const.h.cgs.value
kB  =   const.k_B.cgs.value
cl  =   const.c.cgs.value
Mpc =   const.pc.cgs.value * 1.0e+6
Lsun    =   const.L_sun.cgs.value
Msun    =   const.M_sun.cgs.value
AB  =   48.6e+0         # Zero-point of AB mag. system
mJy =   1.0e-26         # [erg/s/cm2/Hz]
mm  =   1.0e-01         # [cm]
Tcmb0   =   2.735e+0    # Local CMB Temp. [K] by Mather+90
KapUV   =   5.0e+4      # Mass absorption coefficient [cm2/g]
Kap0    =   30e0        # Emiss. normal. [cm2/g] by Hildebrand83
nu0     =   cl / 100e-4 # Frequency at normal [Hz]
beta    =   2.0e+0      # Emiss. index
cosmo = FlatLambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Tcmb0=Tcmb0 * u.K, Om0=OmM)

#- Function Definitions -------------------------------------------------------------------------

#- UV escape fraction -----------------------------------------------------------------------
def PescUV(geo, Robj, clp, Mdust):

    PescUV = 0.; tauuv=0.; taucl=0.; tauMG=0.
    taumin = 1.0e-04


    if geo == 0:
       # Spherical shell
       tauuv = Mdust * KapUV / (4.*PI * Robj**2.)
       PescUV = np.exp(-tauuv)        
    elif geo == 1:
       # Homogeneous sphere
       tauuv =  Mdust * 3. * KapUV / (4.*PI * Robj**2.)
       if tauuv > taumin:
          PescUV = 3. / 4. / tauuv * (1. - .5/tauuv**2. + (1./tauuv + .5/tauuv**2.) * np.exp(-2.*tauuv))
       else:
          PescUV = 1. - 3./4. * tauuv + .5 * tauuv**2.
    elif geo == 2:
       # Clumply sphere with MGA
       tauuv =  Mdust * 3. * KapUV / (4.*PI * Robj**2.)
       taucl = tauuv * clp
       if taucl > taumin:
            tauMG = tauuv * 3. / 4. / taucl * (1. - .5/taucl**2. + (1./taucl + .5/taucl**2.) * np.exp(-2.*taucl))
       else:
            tauMG = tauuv * (1. - 3./4. * taucl + .5 * taucl**2.)

       if tauMG > taumin:
            PescUV = 3. / 4. / tauMG * (1. - .5/tauMG**2. + (1./tauMG + .5/tauMG**2.) * np.exp(-2.*tauMG))
       else:
            PescUV = 1. - 3./4. * tauMG + .5 * tauMG**2.
    else:
       print("Invalid geometry parameter!")
       sys.exit()

    return PescUV

#- Planck function [erg/s/cm2/Hz/sr] --------------------------------------------------------
def Bnu(Temp, nu):  # ([K], [Hz])

    MINIMUM = 1.0e-99

    factor1 = 1.47449916e-47 * nu ** 3      # [erg/s/cm2/Hz/sr]
    factor2 = 4.7992375e-11 * nu / Temp

    Bnu = factor1 / (np.exp(factor2) - 1.)  # [erg/s/cm2/Hz/sr]

    if len(np.argwhere(Bnu < MINIMUM)) != 0:
        Bnu[np.argwhere(Bnu < MINIMUM)[0]] = 0.

    return Bnu

#- Expected Flux density for optically thin dust [erg/s/cm2/Hz/sr] --------------------
def pred_fnu_thin(geo, Const_Td, zS, Tcmb, dL, Luv, Robj, nuem, Kap, logMd, logclp):
    Md  =   10**logMd * Msun
    clp =   10**logclp

    # UV escape fraction for given geometry
    Pesc = PescUV(geo,Robj,clp,Md)

    # Dust-absorbed UV luminosity for given geometry
    Labs = Luv * (1. - Pesc) / Pesc                         # [erg/s]

    # Dust temperature
    func = Labs / Const_Td / Md + (kB * Tcmb)**(beta+4.)    # [erg^(beta+4)]
    Tdust = func ** (1./(beta+4.)) / kB                     # [K]

    # Expected Flux density for given geometry and UV luminosity
    fnuexp = Md * Kap * (Bnu(Tdust,nuem) - Bnu(Tcmb,nuem))* (1.+zS) / dL**2.    # [erg/s/cm2/Hz]

    return fnuexp

# Create Arrays for randomized distribution
clp_arr =   np.zeros(Niter)
Md_arr  =   np.zeros(Niter)
Td_arr  =   np.zeros(Niter)
Lir_arr =   np.zeros(Niter)

fnuobs_arr  =   np.zeros([Niter, len(lobs)])
sarc_arr    =   np.zeros(Niter)
# efnuobs_arr =   np.zeros([Niter,3])

# ------------------------------------------
# arrays for plotting
# start ------------------------------------------
if flg_plot:
    plot_nuem_arr    =   10**np.arange(11.8,13+0.1,0.05)
    plot_Kap_arr    =   np.zeros(len(plot_nuem_arr))
    plot_fexp_arr   =   np.zeros([Niter,len(plot_nuem_arr)])

# end ------------------------------------------


sarc_arr    =   normal(sarc,std=esarc,n_samples=Niter).distribution
for idx_band in range(len(lobs)):
    fnuobs_arr[:,idx_band]  =   normal(fnuobs[idx_band],std=efnuobs[idx_band],n_samples=Niter).distribution


# Start iteration for distributed fluxes
for idx in range(Niter):
    # Constant for Temperature determination
    Const_Td = 8.0e+0*PI * Kap0 * nu0**(-beta) * hP**(-beta-3.0e+0) / cl**2. * zeta(beta+4.0e0) * math.gamma(beta+4.0e0) 

    # CMB temperature at zS
    Tcmb = Tcmb0 * (1.0e+0+zS)

    #Object physical size [cm]
    dL  =   cosmo.luminosity_distance(zS).cgs.value
    # Robj = sarc / 3.6e+3 * PI / 180. * dL / (1.+zS)**2. / np.sqrt(muGL)
    Robj = sarc_arr[idx] / 3.6e+3 * PI / 180. * dL / (1.+zS)**2. / np.sqrt(muGL)

    #Emission wavelength etc.
    lem = lobs / (1.+zS)    # [cm]
    nuem = cl / lem         # [Hz]
    Kap = Kap0 * (nuem / nu0)**beta  # [cm2/g]

    # Lensing-corrected flux density and its uncertainty [erg/s/cm2/Hz]
    # uses distributed fluxes

    fnuIR   =   fnuobs_arr[idx,:] / muGL * mJy
    # fnuIR = fnuobs / muGL * mJy
    efnuIR = efnuobs / muGL * mJy

    #- Peforming chi^2 minimization -----------------------------------------------------------------
    def func_chi2(params,args):

        logMd, logclp   =   params
        geo, Const_Td, zS, Tcmb, dL, Luv, Robj, nuem, Kap   =   args

        return np.sum( ((pred_fnu_thin( geo, Const_Td, zS, Tcmb, dL, Luv, Robj, nuem, Kap,logMd, logclp) - fnuIR)**2.) / efnuIR**2.)

    args    =   [geo, Const_Td, zS, Tcmb, dL, Luv, Robj, nuem, Kap]
    initial_guess   =   np.array([8.,0.1])

    #- Perform Scipy.optimize.minimize --------------------
    Results =   minimize(func_chi2,initial_guess,args=args)

    # print('\n########### Minimization Results:')
    # print(Results)
    # print('##################################\n')

    #- Calculate best fit results -------------------------------------------------------------------
    Mdbst, clpbst = Results.x
    Mdbst   =   10**Mdbst * Msun    #[g]
    clpbst  =   10**clpbst

    # UV escape fraction
    Pebst = PescUV(geo, Robj, clpbst, Mdbst)

    # Absorbed luminosity
    Labs = Luv * (1. - Pebst) / Pebst   #[erg/s]

    # Dust IR luminosity
    Ldust = Labs + Const_Td*Mdbst*(kB*Tcmb)**(beta+4.) # [erg/s]

    # # Dust temperature
    func = Labs / Const_Td / Mdbst + (kB * Tcmb)**(beta+4.)    # [erg^(beta+4)]
    Tdbst = func ** (1./(beta+4.)) / kB                     # [K]

    # Output for the best-fit solution
    # print(clpbst, Mdbst/Msun, Pebst, Tdbst, np.log10(Mdbst/Msun), np.log10(Ldust/Lsun))
    clp_arr[idx] =  clpbst   
    Md_arr[idx]  =  np.log10(Mdbst/Msun)
    Td_arr[idx]  =  Tdbst
    Lir_arr[idx]    =   np.log10(Ldust/Lsun)

    # -------------------------------------------------------------------------------------------------
    # Make data for plotting 
    # -------------------------------------------------------------------------------------------------
    if flg_plot:
        plot_Kap_arr[:] = Kap0 * (plot_nuem_arr / nu0)**beta
        plot_fexp_arr[idx,:] = pred_fnu_thin(geo, Const_Td, zS, Tcmb, dL, Luv, Robj, plot_nuem_arr, plot_Kap_arr, np.log10(Mdbst/Msun), np.log10(clpbst))


# print(np.average(clp_arr),np.std(clp_arr))
# print(np.average(Td_arr),np.std(Td_arr))
# print(np.average(Md_arr),np.std(Md_arr))
# print(np.average(Lir_arr),np.std(Lir_arr))

stat_clp    =   np.percentile(clp_arr,[16,50,84])
stat_Td     =   np.percentile(Td_arr,[16,50,84])
stat_Md     =   np.percentile(Md_arr,[16,50,84])
stat_Lir     =   np.percentile(Lir_arr,[16,50,84])

print('Clumpiness param.:', stat_clp[1],'+',stat_clp[2]-stat_clp[1],'-',stat_clp[1]-stat_clp[0])
print('Dust Temperature:', stat_Td[1],'+',stat_Td[2]-stat_Td[1],'-',stat_Td[1]-stat_Td[0])
print('Dust Mass:', stat_Md[1],'+',stat_Md[2]-stat_Md[1],'-',stat_Md[1]-stat_Md[0])
print('IR Luminosity:', stat_Lir[1],'+',stat_Lir[2]-stat_Lir[1],'-',stat_Lir[1]-stat_Lir[0])

t_end = time.time()
elapsed_time = t_end-t_start
print(f"elapsed time:{elapsed_time}")

fig =   plt.figure()
ax1  =   fig.add_subplot(2,2,1)
ax1.hist(clp_arr,bins=30)

ax2 =   fig.add_subplot(2,2,2)
ax2.hist(Td_arr,bins=30)

ax3 =   fig.add_subplot(2,2,3)
ax3.hist(Md_arr,bins=30)

ax4 =   fig.add_subplot(2,2,4)
ax4.hist(Lir_arr,bins=30)

plt.show()

if flg_plot:
    median = np.median(plot_fexp_arr,axis=0)
    percentile_16,median,percentile_84 = np.percentile(plot_fexp_arr,[16,50,84],axis=0)

    Plot_lam_arr =  ( cl / plot_nuem_arr ) * (1. + zS) * 1.0e+4

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(Plot_lam_arr,median/mJy,zorder=98,color='#5E5E5E')
    ax.fill_between(Plot_lam_arr,percentile_16/mJy,percentile_84/mJy,alpha=0.4,zorder=97,color='#1f78b4',edgecolor='none')
    ax.errorbar(lobs * 1.0e+4, fnuobs/muGL,yerr=efnuobs/muGL ,fmt='s',zorder=99,color='#ff7f00')

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tick_params(axis='both',which='minor',length=2.5,width=1.,direction='in')
    plt.tick_params(axis='both',which='major',length=4.0,width=1.3,direction='in')
    plt.xlabel('Observed Wavelength [$\mu$m]',fontsize=16)
    plt.ylabel('Flux Density [mJy]',fontsize=16)

    ax.set_yscale('log')
    ax.set_xscale('log')

    plt.tight_layout()
    plt.ylim(fnuIR.min()/mJy/5,fnuIR.max()/mJy*3)
    plt.xlim(Plot_lam_arr.min(),Plot_lam_arr.max())

    plt.show()
