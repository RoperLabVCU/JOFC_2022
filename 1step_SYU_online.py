# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import time
import numpy as np
#import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from numpy import genfromtxt
#from pathlib import Path
from datetime import datetime as date
import math
#import ruptures as rpt
import warnings

warnings.filterwarnings("ignore") 

plt.rcParams['interactive'] == True
plt.rcParams['savefig.facecolor']="0.8"
plt.rcParams['figure.figsize']=16.0,9.0
plt.rcParams["figure.titlesize"] = 'x-large'
plt.rcParams["figure.titleweight"] = 'bold'

class onlinePFR:
    
    def __init__(self):
        # general parameters
        self.starttime = date.now()
        self.R = 8.314 # gas constant (J/mol/K)
        self.tol_val = 1e-8 # tolerance setting
        self.j = 0
        self.samp_int = 15.0 # this term is the sampling interval (frequency that model solves)
        # R1 parameters - these pertain specifically to the first synthesis step
        self.D = 0.0015875                                   # tubing diameter in m
        self.Ac1 = math.pi*(self.D/2)**2                           # cross-sectional area (m2)       
        self.V1 = 2.0*10**-5 # reactor volume (m^3) 
        self.xl1 = self.V1/self.Ac1 # reactor tubing length (m)
        self.Ea1 = 57726 # activation energy (J/mol)
        self.k01 = 6893 # arrhenius factor (m3/mol/s)
        self.nx1 = 100 # spatial solution size for x
        self.dx1 = self.xl1/(self.nx1-1) # x step
        self.dt1 = .075 # time step
        self.D1 = .0005 # axial dispersion coefficient (m^2/s)
        self.A0 = 1150 # stock concentration of species A (acrylate) (mol/m3)
        self.B0 = 1000 # stock concentration of species B (fluoro) (mol/m3)
        self.x1 = np.linspace(0,self.xl1,self.nx1) # spatial array
        self.k= .12    # thermal conductvity W/(m*K)
        self.p = 1750 # density (kg/m3)
        self.Cp = 1172  # specifc heat (J/kg/K)
        self.a = self.k/(self.p*self.Cp) # thermal diffusivity (m2/s)
        self.Nu = 3.66 # nusselt laminar flow in tube
        self.h = self.Nu*self.k/self.D  # convective heat transfer coefficient (W/m2/K)
        self.T0 = 25+273.15  # stream inlet temperature (degK)
        self.Tw = 130+273.15 # wall temperature (degK)
        self.reltol = 1e-9  # tolerance for convergence  
        self.lam = self.a*self.dt1/self.dx1**2 #heat transfer coefficient
        self.dHr = 0
        #dHr = -553000 #heat of reaction, J/mol

    # function that reads in previously collected .csv data for flowrates/temperature 
    def data_reading_online(self,dft_file,df_file,df2_file):
        
        self.dft_file = dft_file
        self.df_file = df_file
        self.df2_file = df2_file
        
        self.Jkem_raw= genfromtxt(self.dft_file,dtype='str',delimiter=',',invalid_raise=False,skip_header=4) 
        self.temp_time = self.Jkem_raw[0:len(self.Jkem_raw),0]
        self.temp = self.Jkem_raw[0:len(self.Jkem_raw),1]
        self.temp = self.temp.astype('float')
        self.dft = pd.DataFrame({'temp_time':self.temp_time,'temp':self.temp})
        self.dft['SMA_temp'] = self.dft.iloc[:,1].rolling(window=60,min_periods=1).mean()
        
        # acrylate (species A) data is read in from .csv and dataframe created for time and flowrate values and moving average
        # data must be read in as strings, quotes stripped away, then flowrate converted to float
        self.acrylate_raw = genfromtxt(self.df_file,delimiter=',',invalid_raise=False,skip_header=17,dtype='str',usecols=(2,3))
        self.acrylate_raw = np.char.strip(self.acrylate_raw,chars='"')
        self.acrylate_time = self.acrylate_raw[0:len(self.acrylate_raw),0]
        self.acrylate_flow = self.acrylate_raw[0:len(self.acrylate_raw),1]
        self.acrylate_flow = self.acrylate_flow.astype(float)
        self.df = pd.DataFrame({'acrylate_time':self.acrylate_time,'acrylate_flow':self.acrylate_flow})
        self.df['acrylate_time'] = pd.to_datetime(self.df.acrylate_time,format= '%H:%M:%S.%f')
        self.df['acrylate_time'] = self.df['acrylate_time'].astype(str)
        self.df['acrylate_time'] = (self.df['acrylate_time'].str.split('01-01').str[1].astype(str))
        self.df['acrylate_time'] = (self.df['acrylate_time'].str.split('.').str[-2].astype(str))
        self.df['SMA_acrylate_flow'] = self.df.iloc[:,1].rolling(window=60,min_periods=1).mean()
        
        # fluoro (species B) data is read in from .csv and dataframe created for time and flowrate values and moving average
        # data must be read in as strings, quotes stripped away, then flowrate converted to float
        self.fluoro_raw = genfromtxt(self.df2_file,delimiter=',',invalid_raise=False,skip_header=17,dtype='str',usecols=(2,3))
        self.fluoro_raw = np.char.strip(self.fluoro_raw,chars='"')
        self.fluoro_time = self.fluoro_raw[0:len(self.fluoro_raw),0]
        self.fluoro_flow = self.fluoro_raw[0:len(self.fluoro_raw),1]
        self.fluoro_flow = self.fluoro_flow.astype(float)
        self.df2 = pd.DataFrame({'fluoro_time':self.fluoro_time,'fluoro_flow':self.fluoro_flow})
        self.df2['fluoro_time'] = pd.to_datetime(self.df2.fluoro_time,format= '%H:%M:%S.%f')
        self.df2['fluoro_time'] = self.df2['fluoro_time'].astype(str)
        self.df2['fluoro_time'] = (self.df2['fluoro_time'].str.split('01-01').str[1].astype(str))
        self.df2['fluoro_time'] = (self.df2['fluoro_time'].str.split('.').str[-2].astype(str))
        self.df2['SMA_fluoro_flow'] = self.df2.iloc[:,1].rolling(window=60,min_periods=1).mean()


    # function that initializes arrays for solver and to temporarily store solutions
    # acting as initial conditions for each species
    def SYU_Initialize_Arrays_online(self):

        self.A = np.ones(self.nx1) # species A solution array 
        self.An = np.ones(self.nx1) # species A temporary solution array
        self.B = np.ones(self.nx1) # species B solution array
        self.Bn = np.ones(self.nx1) # species B temporary solution array
        self.C = np.zeros(self.nx1) # species C solution array
        self.Cn = np.zeros(self.nx1) # species C temporary solution array
        self.T = np.ones(self.nx1)
        self.Tn = np.ones(self.nx1)
        self.Tw = self.dft.SMA_temp[len(self.Jkem_raw)-1]+273.15
        self.k1 = self.k01*np.exp(-self.Ea1/self.R/self.T[1:-1])
        self.tolcheck = np.ones(self.nx1)
        
        self.data_log = pd.DataFrame(columns=['time','ConcC','InletA',\
                            'InletB','WallTemp','normA','normB','normT','normC'])
        
        self.data_log = self.data_log.append({'time':0,'ConcC':0,'InletA':0,'InletB':0,    \
                    'WallTemp':0,'normA':0,'normB':0,'normT':0,'normC':0},ignore_index=True)

        self.data_log['SMA_concC'] = self.data_log.iloc[:,1].rolling(window=3,min_periods=1).mean()
        
    def SYU_Arrays_updateIC_online(self):
    
        self.tolcheck = np.ones(self.nx1)
        self.A = np.ones(self.nx1) # species A solution array 
        self.An = np.ones(self.nx1) # species A temporary solution array
        self.B = np.ones(self.nx1) # species B solution array
        self.Bn = np.ones(self.nx1) # species B temporary solution array
        self.C = np.zeros(self.nx1) # species C solution array
        self.Cn = np.zeros(self.nx1) # species C temporary solution array
        self.T = np.ones(self.nx1)
        self.Tn = np.ones(self.nx1)
        self.Tw = self.dft.SMA_temp[len(self.Jkem_raw)-1]+273.15
        self.k1 = self.k01*np.exp(-self.Ea1/self.R/self.T[1:-1])
    
    # function that uses flowrate readings to calculate combined flowrates and velocity
    # flowrates from sensirion are uL/s, converted to mL/min with calibration, then unit conversion to m3/s
    def SYU_flowrates_online(self):
        
        # R1 flowrates in realtime
        self.SMA_f = ((self.df2.SMA_fluoro_flow[len(self.fluoro_raw)-1]*0.00582)-1.55)*1.66667*10**-8 # current moving average flowrate flouro (species B) (m3/s)
        self.SMA_a = ((self.df.SMA_acrylate_flow[len(self.acrylate_raw)-1]*0.004)-.7)*1.66667*10**-8 # current moving average flowrate acrylate (species A) (m3/s)
        self.Q1 = self.SMA_f + self.SMA_a # first reactor combined volumetric flowrate (m3/s)
        self.u1 = self.Q1/self.Ac1                                     # average velocity (m/s)
        self.F_f = self.B0*self.SMA_f/self.Q1 # fluoro (specis B) molar stream concentration (mol/m3)
        self.F_a = self.A0*self.SMA_a/self.Q1 # acrylate (specis A) molar stream concentration (mol/m3)
        
        self.Tw = self.dft.SMA_temp[len(self.Jkem_raw)-1]+273.15
        

    def check_yourself(self,var):
        return np.abs(((np.linalg.norm(self.tolcheck)-np.linalg.norm(var)))/np.linalg.norm(var))
    
    def CDS(self,main,s1,s2,stoic):
        self.main[1:-1] \
            -(self.u1)*(self.dt1/(2*self.dx1))*(self.main[2:]-self.main[:-2]) \
            +self.D1*self.dt1/self.dx1**2*(self.main[2:]-2*self.main[1:-1]+self.main[:-2]) \
            +self.stoic*self.k1*self.s1[1:-1]*self.s2[1:-1]*self.dt1   
            
    def T_UDS(self,te):
        self.te = te
        
        out = self.te[1:-1]-(self.u1*(self.dt1/self.dx1)*(self.te[1:-1]   \
            -self.te[:-2]))+self.lam*(self.te[2:]-2*self.te[1:-1]+self.te[:-2])   \
            -self.h*self.D*math.pi*(self.te[1:-1]-self.Tw)*self.dt1/self.p/self.Cp*self.xl1/self.V1  \
            -self.dHr*self.k1*self.A[1:-1]*self.B[1:-1]/self.p/self.Cp*self.dt1
        return out
    
    def UDS(self,main,s1,s2,stoic1):
        
        self.main = main
        self.s1=s1
        self.s2=s2
        self.stoic1=stoic1
        
        out = self.main[1:-1] \
        -(self.u1)*(self.dt1/(self.dx1))*(self.main[1:-1]-self.main[:-2]) \
        +self.D1*self.dt1/self.dx1**2*(self.main[2:]-2*self.main[1:-1]+self.main[:-2]) \
        +self.stoic1*self.k1*self.s1[1:-1]*self.s2[1:-1]*self.dt1      
        return out

    # function that uses all defined and calculated values to solve for the reaction progression in the tubular reactor
    def SYU_solver_vector_online(self):
        
        # initialize termination condition
        # compares norms of current and previous solution arrays
        self.termcond = self.check_yourself(self.Cn)
        self.termcond = np.float(self.termcond)
        self.stepcount = 1 # step counter
        while self.termcond >= self.reltol: 
            if self.stepcount >20:
                self.termcond = self.check_yourself(self.Cn)
          
            print(self.termcond)
            self.T[0]=self.T0 # impose dirichlet BC
            self.T[self.nx1-1]=self.T[self.nx1-2] # impose neumann BC
            self.Tn = self.T.copy() # update temporary array
            # next line is vectorized FDM spatial solution
            self.T[1:-1] = self.T_UDS(self.Tn)
            self.k1 = self.k01*np.exp(-self.Ea1/self.R/self.T[1:-1])
    
            self.A[0] = self.F_a
            self.B[0] = self.F_f
            self.A[self.nx1-1] = self.A[self.nx1-2] 
            self.B[self.nx1-1] = self.B[self.nx1-2] 
            self.C[self.nx1-1] = self.C[self.nx1-2] 
    
            self.An = self.A.copy()
            self.Bn = self.B.copy()
            self.Cn = self.C.copy()

            self.A[1:-1] = self.UDS(self.An,self.An,self.Bn,-1.0)
            self.B[1:-1] = self.UDS(self.Bn,self.An,self.Bn,-1.0)
            self.C[1:-1] = self.UDS(self.Cn,self.An,self.Bn,1.0)
            
            self.tolcheck = self.C.copy()
            self.stepcount += 1 # update counter


    def SYU_init_visualize_online(self):
    
        plt.ion()
        self.fig,self.axs = plt.subplots(nrows=2,ncols=2,constrained_layout=True,sharex='col')
        self.axs = self.axs.flatten()
        self.axs[0].clear()
        self.axs[1].clear()
        self.axs[2].clear()
        self.axs[3].clear()
        
        self.dataA = self.data_log.plot(x='time',ax=self.axs[0],y='normA',label='Flowrate A',legend=None,style='bo-',mec='k',mfc='g')
        self.dataB = self.data_log.plot(x='time',ax=self.axs[0],y='normB',label='Flowrate B',legend=None,style='b^-',mec='k',mfc='b')
        self.dataT = self.data_log.plot(x='time',ax=self.axs[0],y='normT',label='Wall Temperature',legend=None,style='bs-',mec='k',mfc='y')
        self.dataSol = self.data_log.plot(x='time',ax=self.axs[0],y='normC',label='Target Conc',legend=None,style='rx-',mec='k',mfc='r')
        self.axs[0].axhline(y=0.10,color='r',linestyle='--')
        self.axs[0].axhline(y=-0.10,color='r',linestyle='--')
        self.axs[0].set_ylabel('Normalized Data',size='x-large',fontweight='bold')
        self.axs[0].set_xlabel('time (HH:MM:SS)',size='x-large',fontweight='bold')
        self.axs[0].axes.set_ylim([-0.5,0.5])
        self.axs[0].axes.set_xticks(self.data_log.index)
        self.axs[0].axes.set_xticklabels(self.data_log.time,rotation=45,fontsize ='medium')
        self.axs[0].set_xlim([self.j-10,self.j])
        self.axs[0].axes.set_title('Sensitivity Plot and Outlet Concentration',size='x-large',weight='bold')
        self.axs[0].legend()
        self.axs[0].text(0.075,0.59,'upper limit',c='r',horizontalalignment='center',verticalalignment='center', transform=self.axs[0].transAxes)
        self.axs[0].text(0.075,0.41,'lower limit',c='r',horizontalalignment='center',verticalalignment='center', transform=self.axs[0].transAxes)
        
        self.axs[1].plot(self.x1, self.T[:]-273.15)
        self.axs[1].set_xlabel('Tubing Length (m)',size='x-large',fontweight='bold')
        self.axs[1].set_ylabel('Temperature (degC)',size='x-large',fontweight='bold')
        self.axs[1].axes.set_title('Axial Temperature and Concentration Profiles',size='x-large',weight='bold')
        
        self.solplot = self.data_log.plot(x='time',ax=self.axs[2],y='ConcC',legend=None,style='bo-',mec='k',mfc='r')
        self.axs[2].set_ylabel('Concentration (M)',fontsize ='x-large',fontweight='bold')
        self.axs[2].set_ylim([0,0.75])
        self.axs[2].set_xlabel('time (HH:MM:SS)',fontsize ='x-large',fontweight='bold')
        self.axs[2].axes.set_xticks(self.data_log.index)
        self.axs[2].axes.set_xticklabels(self.data_log.time,rotation=45,fontsize ='medium')
        self.axs[2].set_xlim([self.j-10,self.j])
        
        self.axs[3].plot(self.x1, self.C[:])
        self.axs[3].plot(self.x1, self.B[:])
        self.axs[3].plot(self.x1, self.A[:])
        self.axs[3].set_xlabel('Tubing Length (m)',fontsize ='x-large',fontweight='bold')
        self.axs[3].set_ylabel('Concentration (M)',fontsize ='x-large',fontweight='bold')
        
        self.fig.set_facecolor("0.8")
        #fig.suptitle('Tubular Reactor 2D Axisymmetric Profile')
        plt.show()
    
    def SYU_visualize_online(self):

        plt.ion()
        self.axs = self.axs.flatten()
        self.axs[0].clear()
        self.axs[1].clear()
        self.axs[2].clear()
        self.axs[3].clear()
        
        self.dataA = self.data_log.plot(x='time',ax=self.axs[0],y='normA',label='Flowrate A',legend=None,style='bo-',mec='k',mfc='g')
        self.dataB = self.data_log.plot(x='time',ax=self.axs[0],y='normB',label='Flowrate B',legend=None,style='b^-',mec='k',mfc='b')
        self.dataT = self.data_log.plot(x='time',ax=self.axs[0],y='normT',label='Wall Temperature',legend=None,style='bs-',mec='k',mfc='y')
        self.dataSol = self.data_log.plot(x='time',ax=self.axs[0],y='normC',label='Target Conc',legend=None,style='rx-',mec='k',mfc='r')
        self.axs[0].axhline(y=0.10,color='r',linestyle='--')
        self.axs[0].axhline(y=-0.10,color='r',linestyle='--')
        self.axs[0].set_ylabel('Normalized Data',size='x-large',fontweight='bold')
        self.axs[0].set_xlabel('time (HH:MM:SS)',size='x-large',fontweight='bold')
        self.axs[0].axes.set_ylim([-0.5,0.5])
        self.axs[0].axes.set_xticks(self.data_log.index)
        self.axs[0].axes.set_xticklabels(self.data_log.time,rotation=45,fontsize ='medium')
        self.axs[0].set_xlim([self.j-10,self.j])
        self.axs[0].legend()
        self.axs[0].axes.set_title('Sensitivity Plot and Outlet Concentration',size='x-large',weight='bold')
        self.axs[0].text(0.075,0.59,'upper limit',c='r',horizontalalignment='center',verticalalignment='center', transform=self.axs[0].transAxes)
        self.axs[0].text(0.075,0.41,'lower limit',c='r',horizontalalignment='center',verticalalignment='center', transform=self.axs[0].transAxes)
        
        self.axs[1].plot(self.x1, self.T[:]-273.15)
        self.axs[1].set_xlabel('Tubing Length (m)',size='x-large',fontweight='bold')
        self.axs[1].set_ylabel('Temperature (degC)',size='x-large',fontweight='bold')
        self.axs[1].axes.set_title('Axial Temperature and Concentration Profiles',size='x-large',weight='bold')
        
        self.solplot = self.data_log.plot(x='time',ax=self.axs[2],y='ConcC',legend=None,style='bo-',mec='k',mfc='r')
        self.axs[2].set_ylabel('Concentration (M)',fontsize ='x-large',fontweight='bold')
        self.axs[2].set_ylim([0,0.75])
        self.axs[2].set_xlabel('time (HH:MM:SS)',fontsize ='x-large',fontweight='bold')
        self.axs[2].axes.set_xticks(self.data_log.index)
        self.axs[2].axes.set_xticklabels(self.data_log.time,rotation=45,fontsize ='medium')
        self.axs[2].set_xlim([self.j-10,self.j])
        
        self.axs[3].plot(self.x1, self.C[:])
        self.axs[3].plot(self.x1, self.B[:])
        self.axs[3].plot(self.x1, self.A[:])
        self.axs[3].set_xlabel('Tubing Length (m)',fontsize ='x-large',fontweight='bold')
        self.axs[3].set_ylabel('Concentration (M)',fontsize ='x-large',fontweight='bold')
        
    
        
        self.fig.set_facecolor("0.8")
        #fig.suptitle('Tubular Reactor 2D Axisymmetric Profile')
        self.fig.savefig('CLASSTEST-%s.jpg'%(self.j))
        plt.show()
        plt.pause(1.0)
  
# model update loop, runs continuously until manually stopped, or if data files are not present to read
plt.ion()

test = onlinePFR()
test.__init__()

test.data_reading_online(dft_file='temp_100121.csv',\
                              df_file='acrylate_100121.csv',\
                              df2_file='fluoro_100121.csv')
test.SYU_Initialize_Arrays_online() # calls function to re-initializes arrays
test.SYU_init_visualize_online()

while True:    
    test.j += 1
    test.init = date.now()
    test.data_reading_online(dft_file='temp_100121.csv',\
                              df_file='acrylate_100121.csv',\
                              df2_file='fluoro_100121.csv') # reads in and updates data into dataframes
    test.SYU_Arrays_updateIC_online() # calls function to re-initializes arrays
    test.SYU_flowrates_online() # calls flowrate function
    test.SYU_solver_vector_online() # calls solver
    
    test.normA = (test.SMA_a/(1.66667*10**-8)-2.0)/2.0
    test.normB = (test.SMA_f/(1.66667*10**-8)-2.0)/2.0
    test.normT = (test.Tw-273.15-130)/130
    test.normC = (test.C[test.nx1-1]/1000-0.5)/0.5
    
    test.data_log = test.data_log.append({'ConcC':test.C[test.nx1-1]/1000,    \
                'time':test.df2.fluoro_time[len(test.df2.fluoro_time)-1],'WallTemp':test.Tw-273.15,   \
                'InletA':test.SMA_a/(1.66667*10**-8),'InletB':test.SMA_f/(1.66667*10**-8), \
               'normT':test.normT,'normA':test.normA, 'normB':test.normB,'normC':test.normC},ignore_index=True)
    test.data_log['SMA_concC'] = test.data_log.iloc[:,1].rolling(window=3,min_periods=1).mean()
    test.data_log.to_csv('CLASStest.csv',index=False)
    test.SYU_visualize_online() # calls visualization function

    test.end = date.now()
    test.runtime = test.end - test.starttime
    test.runtime = test.runtime.total_seconds()

    test.cputime = test.end-test.init
    test.cputime = test.cputime.total_seconds()
    #print(test.cputime)

    if test.cputime > test.samp_int:
        test.wait = 0.0
    else:
        test.wait = test.samp_int-test.cputime
    if test.wait <= 0.0:
      test.wait = 0.0
    time.sleep(test.wait) # pauses model loop before refreshing and updating next solution
    
    
    
    
    
    