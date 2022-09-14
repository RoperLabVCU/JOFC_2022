# =============================================================================
# 
# Offline Process Model
# Explicit Finite Difference Method Code
# Solves the 1D Reaction Adnvection-Diffusion Equation
# Assumes Tubular Plug-Flow-Reactor in Laminar Regime 
# Assumes average velocity
# Written by: Cameron Armstrong (2020)
# Updated by: Cameron Armstrong (2021)
# Institution: Virginia Commonwealth University
# 
# =============================================================================

# Required Modules
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from numpy import genfromtxt
plt.rcParams['interactive'] == True

import warnings

warnings.filterwarnings("ignore") 

class offlinePFR:

    # function that defines constant parameters for model
    def __init__(self):
        # general
        self.R = 8.314 # gas constant (J/mol/K)
        self.samp_int = 15 # this term is the sampling interval (frequency that model solves) (data logged every 2 seconds)
        self.tol_val = 1e-5 # tolerance setting
        self.j = 0
        self.h = 1
        
        # R1 parameters - these pertain specifically to the first synthesis step
        self.Ac1 = 2.0*10**-6 # cross-sectional area of reactor (m^2)
        self.V1 = 1.0*10**-5 # reactor volume (m^3) 
        self.xl1 = self.V1/self.Ac1 # reactor tubing length (m)
        self.Ea1 = 53106.5 # activation energy (J/mol)
        self.k01 = 1175.26 # arrhenius factor (m3/mol/s)
        self.nx1 = 100 # spatial solution size for x
        self.dx1 = self.xl1/(self.nx1-1) # x step
        self.dt1 = .005 # time step
        self.nt1 = np.round_(2*self.samp_int/self.dt1,0) #time counter calculation, 2 accounts for data logged every 2 seconds
        self.nt1 = self.nt1.astype(int)# time counter for temporal loop
        self.D1 = .005 # axial dispersion coefficient (m^2/s)
        self.A0 = 1200 # stock concentration of species A (acrylate) (mol/m3)
        self.B0 = 1000 # stock concentration of species B (fluoro) (mol/m3)
        self.x1 = np.linspace(0,self.xl1,self.nx1) # spatial array
        
        # R2 parameters - these pertain specifically to the second synthesis step
        self.Ac2 = 2.0*10**-6 # cross-sectional area of reactor (m^2)
        self.V2 = 5.0*10**-6 # reactor volume (m^3) 
        self.xl2 = self.V2/self.Ac2 # reactor tubing length (m)
        self.Ea2 = 23680.8 # activation energy (J/mol)
        self.k02 = 11.271 # arrhenius factor (m3/mol/s)
        self.nx2 = 200 # spatial solution size for x
        self.dt2 = .005 # time step
        self.nt2 = np.round_(2*self.samp_int/self.dt2,0) # time counter for temporal loop
        self.nt2 = self.nt2.astype(int)
        self.dx2 = self.xl2/(self.nx2-1) # x step
        self.D2 = .005 # axial dispersion coefficient (m^2/s)
        self.D0 = 1250 # stock concentration of species D (cyclo) (mol/m3)
        self.x2 = np.linspace(0,self.xl2,self.nx2) # spatial array

# function that reads in previously collected .csv data for flowrates/temperature     
    def data_reading_offline(self,dft_file,df_file,df2_file,df3_file):
        
        self.dft_file = dft_file
        self.df_file = df_file
        self.df2_file = df2_file
        self.df3_file = df3_file
        
        # temp data is read in from .csv and dataframe created for time and temp values and moving average 
        # data read in as deg celsius
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
        self.df['SMA_acrylate_flow'] = self.df.iloc[:,1].rolling(window=60,min_periods=1).mean()
        
        # fluoro (species B) data is read in from .csv and dataframe created for time and flowrate values and moving average
        # data must be read in as strings, quotes stripped away, then flowrate converted to float
        self.fluoro_raw = genfromtxt(self.df2_file,delimiter=',',invalid_raise=False,skip_header=17,dtype='str',usecols=(2,3))
        self.fluoro_raw = np.char.strip(self.fluoro_raw,chars='"')
        self.fluoro_time = self.fluoro_raw[0:len(self.fluoro_raw),0]
        self.fluoro_flow = self.fluoro_raw[0:len(self.fluoro_raw),1]
        self.fluoro_flow = self.fluoro_flow.astype(float)
        self.df2 = pd.DataFrame({'fluoro_time':self.fluoro_time,'fluoro_flow':self.fluoro_flow})
        self.df2['SMA_fluoro_flow'] = self.df2.iloc[:,1].rolling(window=60,min_periods=1).mean()
        
        # cyclo (species D) data is read in from .csv and dataframe created for time and flowrate values and moving average
        # data must be read in as strings, quotes stripped away, then flowrate converted to float
        self.cyclo_raw = genfromtxt(self.df3_file,delimiter=',',invalid_raise=False,skip_header=17,dtype='str',usecols=(2,3))
        self.cyclo_raw = np.char.strip(self.cyclo_raw,chars='"')
        self.cyclo_time = self.cyclo_raw[0:len(self.cyclo_raw),0]
        self.cyclo_flow = self.cyclo_raw[0:len(self.cyclo_raw),1]
        self.cyclo_flow = self.cyclo_flow.astype(float)
        self.df3 = pd.DataFrame({'cyclo_time':self.cyclo_time,'cyclo_flow':self.cyclo_flow})
        self.df3['SMA_cyclo_flow'] = self.df3.iloc[:,1].rolling(window=60,min_periods=1).mean()
        
        self.nl = len(self.acrylate_time) # maximum number of times that model will solve

    # function that initializes arrays for solver and to temporarily store solutions
    # acting as initial conditions for each species
    def Arrays_initialize_offline(self):
        
        # R1 solutions arrays
        self.A = np.ones(self.nx1) # species A solution array 
        self.An = np.ones(self.nx1) # species A temporary solution array
        self.B = np.ones(self.nx1) # species B solution array
        self.Bn = np.ones(self.nx1) # species B temporary solution array
        self.C = np.ones(self.nx1)*1e-15 # species C solution array
        self.Cn = np.ones(self.nx1)*1e-15 # species C temporary solution array
        self.tolcheck1 = np.ones(self.nx1)
        
        # R2 solutions array
        self.C2 = np.ones(self.nx2) # species C solution array for reactor 2 as reagent
        self.C2m = np.ones(self.nx2) # species C temporary solution array for reactor 2 as reagent
        self.D = np.ones(self.nx2) # species D solution array
        self.Dm = np.ones(self.nx2) # species D temporary solution array
        self.E = np.ones(self.nx2)*1e-15  # species E solution array
        self.Em = np.ones(self.nx2)*1e-15  # species E temporary solution array
        self.tolcheck2 = np.ones(self.nx2)
        
        self.data_log = pd.DataFrame(columns=['time','ConcC','InletA',\
                            'InletB','InletD','WallTemp','normA','normB','normD','normT','normC'])
        
        self.data_log = self.data_log.append({'time':0,'ConcC':0,'InletA':0,'InletB':0,    \
                    'InletD':0,'WallTemp':0,'normA':0,'normB':0,'normD':0,'normT':0,\
                    'normC':0},ignore_index=True)

        self.data_log['SMA_R1'] = self.data_log.iloc[:,1].rolling(window=3,min_periods=1).mean()
        self.data_log['SMA_R2'] = self.data_log.iloc[:,1].rolling(window=3,min_periods=1).mean()
        
        self.dfR1sol = pd.DataFrame(columns=['R1_Conc','R1_time'])
        self.dfR1sol = self.dfR1sol.append({'R1_Conc':0,'R1_time':0},ignore_index=True)
        self.dfR1sol['SMA_R1_Conc'] = self.dfR1sol.iloc[:,0].rolling(window=3,min_periods=1).mean()
        self.dfR2sol = pd.DataFrame(columns=['R2_Conc','R2_time']) # initializes model solution dataframe
        
        self.dfc = pd.DataFrame(columns=['calibrated mL/min','time'])
        self.dfc = self.dfc.append({'calibrated mL/min':0,'time':0},ignore_index=True)
        self.df2c = pd.DataFrame(columns=['calibrated mL/min','time'])
        self.df2c = self.df2c.append({'calibrated mL/min':0,'time':0},ignore_index=True)
        self.df3c = pd.DataFrame(columns=['calibrated mL/min','time'])
        self.df3c = self.df3c.append({'calibrated mL/min':0,'time':0},ignore_index=True)
        self.df4c = pd.DataFrame(columns=['temp','time'])
        self.df4c = self.df4c.append({'temp':0,'time':0},ignore_index=True)

    def Arrays_updateIC_offline(self):
    
        # R1 solutions arrays
        self.A = np.ones(self.nx1) # species A solution array 
        self.An = np.ones(self.nx1) # species A temporary solution array
        self.B = np.ones(self.nx1) # species B solution array
        self.Bn = np.ones(self.nx1) # species B temporary solution array
        self.C = np.ones(self.nx1)*1e-15  # species C solution array
        self.Cn = np.ones(self.nx1)*1e-15   # species C temporary solution array
        self.tolcheck1 = np.ones(self.nx1)
        
        # R2 solutions array
        self.C2 = np.ones(self.nx2) # species C solution array for reactor 2 as reagent
        self.C2m = np.ones(self.nx2) # species C temporary solution array for reactor 2 as reagent
        self.D = np.ones(self.nx2) # species D solution array
        self.Dm = np.ones(self.nx2) # species D temporary solution array
        self.E = np.ones(self.nx2)*1e-15   # species E solution array
        self.Em = np.ones(self.nx2)*1e-15   # species E temporary solution array
        self.tolcheck2 = np.ones(self.nx2)

    # function that uses temperature reading to calculate rate constants       
    def Temp_offline(self):
        
        # R1 rate constant calculation as function of solution counter "h"
        self.Tj1= self.dft.SMA_temp[self.h]+273.15 # temp of first reactor (converting degK to degC)
        self.k1 = self.k01*np.exp(-self.Ea1/self.R/self.Tj1) # first reaction rate constant (m3/mol/s)
        
        # R2 rate constant calculation as function of solution counter "h"
        self.Tj2 = 298.15 # temp of reactor 2 (constant) (degC)
        self.k2 = self.k02*np.exp(-self.Ea2/self.R/self.Tj2) # second reaction rate constant (m3/mol/s)
        
        self.df4c = self.df4c.append({'temp':self.Tj1-273.15,'time':self.temp_time[self.h]},ignore_index=True)

    # function that uses flowrate readings to calculate combined flowrates and velocity
    # flowrates from sensirion are uL/s, converted to mL/min with calibration, then unit conversion to m3/s
    def Flowrates_offline(self):
        
        # R1 flowrates as a function of "h" which is the solution counter
        self.SMA_f = ((self.df2.SMA_fluoro_flow[self.h]*0.00387)-0.273)*1.66667*10**-8 # current moving average flowrate flouro (species B) (m3/s)
        self.SMA_a = ((self.df.SMA_acrylate_flow[self.h]*0.00897)-.0769)*1.66667*10**-8 # current moving average flowrate acrylate (species A) (m3/s)
        self.Q1 = self.SMA_f + self.SMA_a # first reactor combined volumetric flowrate (m3/s)
        self.u1 = self.Q1/self.Ac1 # first reactor average velocity (m/s)
        self.F_f = self.B0*self.SMA_f/self.Q1 # fluoro (specis B) molar stream concentration (mol/m3)
        self.F_a = self.A0*self.SMA_a/self.Q1 # acrylate (specis A) molar stream concentration (mol/m3)
        
        
        # R2 flowrates as a function of "h" which is the solution counter
        self.SMA_c = ((self.df3.SMA_cyclo_flow[self.h]*0.0269)-2.85)*1.66667*10**-8 # current moving average flowrate flouro (species B) (m3/s)
        self.Q2 = self.Q1 + self.SMA_c # second reactor combined volumetric flowrate (m3/s)
        self.u2 = self.Q2/self.Ac2 # second reactor average velocity (m/s)
        self.F_c = self.D0*self.SMA_c/self.Q2 # cyclo (specis D) molar stream concentration (mol/m3)
        self.F_R1 = self.C[self.nx1-1]*self.Q1/self.Q2 # R1 product (species C) molar stream concentration (mol/m3)
        
        self.dfc = self.dfc.append({'calibrated mL/min':self.SMA_a/(1.66667*10**-8),'time':self.acrylate_time[self.h]},ignore_index=True)
        self.df2c = self.df2c.append({'calibrated mL/min':self.SMA_f/(1.66667*10**-8),'time':self.fluoro_time[self.h]},ignore_index=True)
        self.df3c = self.df3c.append({'calibrated mL/min':self.SMA_c/(1.66667*10**-8),'time':self.cyclo_time[self.h]},ignore_index=True)
    
    def convergence_criteria(self,current_sol,past_sol):
        return np.abs((np.linalg.norm(current_sol)-np.linalg.norm(past_sol))/(np.linalg.norm(past_sol)))

    def UDS(self,u,dt,dx,Dax,kt,main,s1,s2,stoic1):
        
        self.main = main
        self.u = u
        self.dt = dt
        self.dx = dx
        self.Dax = Dax
        self.kt = kt
        self.s1 = s1
        self.s2 = s2
        self.stoic1 = stoic1
        
        out = self.main[1:-1] \
        -(self.u)*(self.dt/(self.dx))*(self.main[1:-1]-self.main[:-2]) \
        +self.Dax*self.dt/self.dx**2*(self.main[2:]-2*self.main[1:-1]+self.main[:-2]) \
        +self.stoic1*self.kt*self.s1[1:-1]*self.s2[1:-1]*self.dt    
        return out
    
    # function that uses all defined and calculated values to solve for the reaction progression in the tubular reactor
    def Solver_vector_offline(self):
        
        self.termcondR1 = self.convergence_criteria(self.tolcheck1,self.Cn)
        self.termcondR1 = np.float(self.termcondR1)
        self.stepcount = 1 # step counter
        self.cfl1 = self.u1*self.dt1/self.dx1
        # solver for first reactor/reaction
        while self.termcondR1 >= self.tol_val:
            if self.stepcount >20:
                self.termcondR1 = self.convergence_criteria(self.tolcheck1,self.Cn)
                print(self.stepcount)
                print(self.termcondR1)
                print(self.cfl1)
            
            if self.stepcount == 12:
                print('apple')
            
            # dirichlet boundary conditions for reactor inlet 
            self.A[0] = self.F_a 
            self.B[0] = self.F_f
            
             # neumann boundary conditions for reactor outlet
            self.A[self.nx1-1] = self.A[self.nx1-2] 
            self.B[self.nx1-1] = self.B[self.nx1-2] 
            self.C[self.nx1-1] = self.C[self.nx1-2]
            
            self.An = self.A.copy() # updating the temporary solution array each loop
            self.Bn = self.B.copy()
            self.Cn = self.C.copy()
          
            # vectorized coupled FDM equations for each species
            self.A[1:-1] = self.UDS(self.u1,self.dt1,self.dx1,self.D1,self.k1,self.An,self.An,self.Bn,-1.0)
            self.B[1:-1] = self.UDS(self.u1,self.dt1,self.dx1,self.D1,self.k1,self.Bn,self.An,self.Bn,-1.0)
            self.C[1:-1] = self.UDS(self.u1,self.dt1,self.dx1,self.D1,self.k1,self.Cn,self.An,self.Bn,+1.0)
            

            self.tolcheck1 = self.C.copy() # adding current solution to storage to check convergence
            self.stepcount += 1 # update counter
            
        self.termcondR2 = self.convergence_criteria(self.tolcheck2,self.Em)
        self.termcondR2 = np.float(self.termcondR2)
        
        self.stepcount = 1 # step counter
        
        # solver for second reactor/reaction    
        while self.termcondR2 >= self.tol_val:
            if self.stepcount >10:
                self.termcondR2 = self.convergence_criteria(self.tolcheck2,self.Em)
            
            # dirichlet boundary conditions for reactor inlet       
            #F_R1 = C[nx1-1]*Q1/Q2 # defines R1 product stream concentration in second reactor as current value from R1 outlet
            self.F_R1 = self.dfR1sol.SMA_R1_Conc[self.j-1]*self.Q1/self.Q2
            self.C2[0] = self.F_R1
            self.D[0] = self.F_c
            
            # neumann boundary conditions for reactor outlet
            self.C2[self.nx2-1] = self.C2[self.nx2-2]
            self.D[self.nx2-1] = self.D[self.nx2-2]
            self.E[self.nx2-1] = self.E[self.nx2-2]
            
            self.C2m = self.C2.copy() # updating the temporary solution array each loop
            self.Dm = self.D.copy()
            self.Em = self.E.copy()
            
            # vectorized coupled FDM equations for each species
            self.C2[1:-1] = self.UDS(self.u2,self.dt2,self.dx2,self.D2,self.k2,self.C2m,self.C2m,self.Dm,-1.0)
            self.D[1:-1] = self.UDS(self.u2,self.dt2,self.dx2,self.D2,self.k2,self.Dm,self.C2m,self.Dm,-1.0)
            self.E[1:-1] = self.UDS(self.u2,self.dt2,self.dx2,self.D2,self.k2,self.Em,self.C2m,self.Dm,+1.0)

            self.stepcount += 1 # update counter
            self.tolcheck2 = self.E.copy() # adding current solution to storage to check convergence


SYU = offlinePFR()

SYU.data_reading_offline(dft_file='7-03-temp-CA2-183_correctset.csv',\
                              df_file='acrylate-7-03-CA2-183.csv',\
                              df2_file='fluoro-7-03-CA2-183.csv',\
                              df3_file='cyclo-7-03-CA2-183.csv',)
    
SYU.Arrays_initialize_offline()

# model loop that calls each function for each model step 
for SYU.j in range(1,SYU.nl):
    
    if SYU.j > SYU.nl/SYU.samp_int: # stops model loop after loop counter surpasses sampling frequency limit
        break
    
    SYU.h=SYU.j*SYU.samp_int # definition of solution counter to pull flowrate and temp values
    SYU.Temp_offline() # calls temperature/rate-constant function
    SYU.Arrays_updateIC_offline() # calls function to re-initializes arrays
    SYU.Flowrates_offline() # calls flowrate function
    SYU.Solver_vector_offline() # calls solver 
    SYU.dfR1sol = SYU.dfR1sol.append({'R1_Conc':SYU.C[SYU.nx1-1],'R1_time':SYU.acrylate_time[SYU.h]},ignore_index=True) # updates model solution array
    SYU.dfR1sol['SMA_R1_Conc'] = SYU.dfR1sol.iloc[:,0].rolling(window=3,min_periods=1).mean() # calculates moving average of model solution
    SYU.dfR2sol = SYU.dfR2sol.append({'R2_Conc':SYU.E[SYU.nx2-1]/1000*313.276,'R2_time':SYU.acrylate_time[SYU.h]},ignore_index=True) # updates model solution array
    SYU.dfR2sol['SMA_R2_Conc'] = SYU.dfR2sol.iloc[:,0].rolling(window=3,min_periods=1).mean() # calculates moving average of model solution

    SYU.normA = (SYU.SMA_a/(1.66667*10**-8)-2.5)/2.5
    SYU.normB = (SYU.SMA_f/(1.66667*10**-8)-2.5)/2.5
    SYU.normD = (SYU.SMA_c/(1.66667*10**-8)-2.5)/2.5
    SYU.normT = (SYU.Tj1-273.15-150)/150
    SYU.normC = (SYU.C[SYU.nx1-1]/1000-0.333)/0.333
    
    SYU.data_log = SYU.data_log.append({'ConcC':SYU.C[SYU.nx1-1]/1000,    \
                'time':SYU.df2.fluoro_time[len(SYU.df2.fluoro_time)-1],'WallTemp':SYU.Tj1-273.15,   \
                'InletA':SYU.SMA_a/(1.66667*10**-8),'InletB':SYU.SMA_f/(1.66667*10**-8), 'InletD':SYU.SMA_c/(1.66667*10**-8),\
               'normT':SYU.normT,'normA':SYU.normA, 'normB':SYU.normB,'normdD':SYU.normD,'normC':SYU.normC},ignore_index=True)
    SYU.data_log['SMA_concC'] = SYU.data_log.iloc[:,1].rolling(window=3,min_periods=1).mean()
    SYU.data_log.to_csv('offlineCLASStest.csv',index=False)
    
    SYU.j += 1
    
#    fig, (ax1,ax2) = plt.subplots(2,sharex=True,sharey=True)
#    ax1.plot(x1/xl1,C)
#    ax1.plot(x1/xl1,A)
#    ax1.plot(x1/xl1,B)
#    ax2.plot(x2/xl2,E)
#    ax2.plot(x2/xl2,C2)
#    ax2.plot(x2/xl2,D)
#    plt.xlim(0,1)
#    plt.ylim(0,600)
#    plt.savefig(str(j)+'.jpg',)
#    plt.close('all')
    #time.sleep(.01)

SYU.dfR1sol.to_csv('offline_testR1.csv',index=False)    
SYU.dfR2sol.to_csv('offline_testR2.csv',index=False) # generates model solution .csv file
#dfc.to_csv('183test-acrylate.csv',index=False)
#df2c.to_csv('183test-fluoro.csv',index=False)
#df3c.to_csv('183test-cyclo.csv',index=False)
#df4c.to_csv('183test-temp.csv',index=False)
#df.to_csv('183test-acrylate_full.csv',index=False)
#df2.to_csv('183test-fluoro_full.csv',index=False)
#df3.to_csv('183test-cyclo_full.csv',index=False)


