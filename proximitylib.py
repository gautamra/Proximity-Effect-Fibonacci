#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:19:08 2020

@author: gautam
"""

##Initial definitions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import kwant

import tinyarray

import lmfit as lmf

import scipy.linalg as lin

from tqdm import tqdm
import pickle

from scipy import stats

tau_x = tinyarray.array([[0, 1], [1, 0]])
tau_y = tinyarray.array([[0, -1j], [1j, 0]])
tau_z = tinyarray.array([[1, 0], [0, -1]])
tau_0 = tinyarray.array([[1, 0], [0, 1]])

def Lorentzian(eex, ee, gam):
    return (gam/np.pi)*(1/((eex-ee)**2 + gam**2))

def Fermi(eps, beta = 2000):
    return 1/(1+np.exp(beta*eps))

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 14}

mpl.rc('font', **font)
mpl.rcParams['figure.figsize'] = 6,3.6
mpl.rcParams['figure.facecolor'] = "w"

####Fibonacci definitions

Fibonacci = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584])
tau_n = np.array([1, 2, 3/2, 5/3, 8/5, 13/8, 21/13, 34/21, 55/34, 89/55, 144/89, 233/144, 377/233, 610/377, 987/610, 1597/987, 2584/1597])

FCs = []
phi = 0
for N, tau in zip(Fibonacci, tau_n):
    sequence = ["A" if np.sign(np.cos(2*np.pi*m*(1/tau) + phi) - np.cos(np.pi*(1/tau)))>=0 else "B" for m in range(N)]
    FCs.append(sequence)
    
FCs = []
for N, tau in zip(Fibonacci, tau_n):
    phis = 2*np.pi*np.arange(N)/N
    sequences = [["A" if np.sign(np.cos(2*np.pi*m*(1/tau) + phi) - np.cos(np.pi*(1/tau)))>=0 else "B" for m in range(N)] for phi in phis]
    FCs.append(sequences)
    
######### System definers

class TBmodel:
    def __init__(self, LL, ts, us, vs):
        self.LL = LL
        self.a = 1
        self.ts, self.us, self.vs = ts, us, vs
        self.Delta = np.array([1]*self.LL)
        self.Pot = np.zeros(self.LL)

    def onsite(self, site, Delta, Pot):
        (x,y) = site.tag
        return (self.us[x]+Pot[x])*tau_z - self.vs[x]*Delta[x]*tau_x
    
    def hopping(self,site1,site2):
        (x2,y2) = site2.tag
        (x1,y1) = site1.tag
        return self.ts[x1]*tau_z
    
    def make_syst(self):
        self.syst = kwant.Builder()
        self.lat = kwant.lattice.square(self.a, norbs = 2)
        
        self.syst[(self.lat(x,0) for x in range(self.LL))] = self.onsite
        self.syst[((self.lat(x+1,0),self.lat(x,0)) for x in range(self.LL-1))] = self.hopping
        self.syst[((self.lat(0,0), self.lat(self.LL-1,0)))] = self.hopping
        
        self.fsyst = self.syst.finalized()
        return

    def solve(self,H):
        (evals, evecs) = lin.eigh(H)
    
        uvecs = evecs[::2]
        vvecs = evecs[1::2]
        
        return (evals[self.LL:],uvecs[:,self.LL:],vvecs[:,self.LL:])

    def iterate(self):
        def self_cons(H):
            (evals, uvecs, vvecs) = self.solve(H)
            self.evals, self.uvecs, self.vvecs = (evals, uvecs, vvecs)
            
            Delta = np.zeros(self.LL, dtype = "complex128")
            for ee, uvec, vvec in zip(evals, uvecs.T, vvecs.T):
                Delta += (1-2*Fermi(ee, beta = self.beta))*uvec*vvec.conjugate()
            
            occupancy = np.zeros(self.LL)
            for ee, uvec, vvec in zip(evals, uvecs.T, vvecs.T):
                    occupancy += Fermi(ee, beta = self.beta)*2*np.abs(vvec)**2 + (1-Fermi(ee))*2*np.abs(uvec)**2
            
            Pot = 1/2*self.vs*occupancy
            Pot = Pot + 0.0001*np.ones(len(Pot))

            return (Delta, Pot)
        
        err_Delta = np.ones(1)
        cc = 0
        
        while any([err/Del>0.001 and err>0.01*(np.max(self.Delta)+0.01) for err,Del in zip(err_Delta, self.Delta)]):
            H = self.fsyst.hamiltonian_submatrix(params = dict(Delta = self.Delta, Pot = self.Pot))
            newDelta, newPot = self_cons(H)
            err_Delta = np.abs(newDelta - self.Delta)
            
            cc += 1
#             if cc%10 == 0:
#                 print("iteration # {}".format(cc))
#                 print("max = {:.3f}, avg = {:.3f}".format(np.max(err_Delta/(self.Delta+0.1*self.Delta)).real, np.average(err_Delta/self.Delta).real))
    
            self.Delta, self.Pot = newDelta, newPot
        
#         print("Convergence took {} iterations".format(cc))
        
        self.H = H
        return self.Delta, self.Pot
        
    
    def get_DOS(self, gam = None, Num_es = 1000):
        emax = np.max(np.abs(self.evals))
        
        if gam == None:
            gam = 2*emax/self.LL
            
        eex = np.linspace(-1.2*emax,1.2*emax, Num_es)
        DOSu = np.zeros(eex.shape)
        DOSv = np.zeros(eex.shape)
        
        for ee, uvec, vvec in zip(self.evals, self.uvecs.T, self.vvecs.T):
            if ee>0:
                DOSu += np.linalg.norm(uvec)**2*Lorentzian(eex,ee,gam) 
                DOSv += np.linalg.norm(vvec)**2*Lorentzian(eex,-ee,gam)
                
        self.DOS = (DOSu + DOSv)/self.LL
        return  self.DOS , eex
    
    def get_LDOS(self, gam = None, Num_es = 1000):
        emax = np.max(np.abs(self.evals))
        
        if gam == None:
            gam = 2*emax/self.LL
            
        eex = np.linspace(-1.2*emax,1.2*emax, Num_es)
        DOSu = np.zeros((self.uvecs.shape[0],eex.shape[0]))
        DOSv = np.zeros(DOSu.shape)
        
        for ee, uvec, vvec in zip(self.evals, self.uvecs.T, self.vvecs.T):
            if ee>0:
                DOSu += (np.abs(uvec)**2)[:,np.newaxis]*Lorentzian(eex,ee,gam)
                DOSv += (np.abs(vvec)**2)[:,np.newaxis]*Lorentzian(eex,-ee,gam)      
            
        self.LDOS = (DOSu + DOSv)/self.LL
        return  self.LDOS,eex
    
    
def chain1D(L, t=-1, u=0, v=0, wt=0, wu = 0, wv = 0):
    chain = {
        "N" : L+1,
        "t" : t - wt/2 + wt*np.random.rand(L),
        "u" : u - wu/2 + wu*np.random.rand(L+1),
        "v" : v - wv/2 + wv*np.random.rand(L+1)
    }
    return chain



def chainFC(n = 3, t=-1, w = 0.1, phi = 0, u = 0, v = 0, wu = 0, wv = 0):
    L = Fibonacci[n]
    tau = tau_n[n]
    FC = FCs[n][phi]
    
    wa = 2*w/(1+tau)
    wb = tau*wa

    ts = [1-wa if letter =="A" else 1+wb for letter in FC]
    chain = {
        "N": L+1,
        "t": -np.array(ts),
        "u" : u - wu/2 + wu*np.random.rand(L+1),
        "v" : v - wv/2 + wv*np.random.rand(L+1)
    }
    return chain

def chainFC_diag(n = 3, t=-1, w = 0.1, phi = 0, u = 0, v = 0, wt = 0, wv = 0):
    L = Fibonacci[n]
    tau = tau_n[n]
    FC = FCs[n][phi]
    
    wa = 2*w/(1+tau)
    wb = tau*wa

    us = [wa if letter =="A" else -wb for letter in FC]
    chain = {
        "N": L,
        "t": t - wt/2 + wt*np.random.rand(L-1),
        "u" : -np.array(us),
        "v" : v - wv/2 + wv*np.random.rand(L)
    }
    return chain

def chainAAH(n = 3, t=-1, u=0, v=0, wt=0, wu = 2, wv = 0, phi = 0):
    L = Fibonacci[n]
    tau = tau_n[n]
    chain = {
        "N" : L,
        "t" : t*np.ones(L-1),
        "u" : u + wu*np.cos(2*np.pi*np.arange(L)*(1/tau) + phi),
        "v" : v*np.ones(L)
    }
    return chain

class hybrid_ring(TBmodel):
    def __init__(self, Norm, SC, t_int=-1, beta = 2000):
        self.NN = Norm["N"]
        self.NS = SC["N"]
        t_left = t_right = t_int
        self.beta = beta
        
        ts = np.concatenate((np.array([t_left]), Norm["t"], np.array([t_right]) , SC["t"]))
        us = np.concatenate((Norm["u"], SC["u"]))
        vs = np.concatenate((Norm["v"], SC["v"]))
        
        TBmodel.__init__(self, self.NN+self.NS, ts, us, vs)
        
        self.Norm_comp = simple_ring(Norm, t_int, beta = self.beta)
        self.SC_comp = simple_ring(SC, t_int, beta = self.beta)
        self.SC_gap = 2*min(np.abs(self.SC_comp.evals))
        
        self.make_syst()        
        self.iterate()
        
class simple_ring(TBmodel):
    def __init__(self, chain, t_int = -1, beta = 2000):
        self.chain = chain
        self.NN = chain["N"]
        self.beta = beta
        
        ts = np.concatenate((chain["t"],np.array([t_int])))
        us = np.array(chain["u"])
        vs = np.array(chain["v"])
        
        TBmodel.__init__(self, self.NN, ts, us, vs)

        self.make_syst()        
        self.iterate()
        
################## Delta profile analysis tools
        
class Delta_profile:
    def __init__(self, Delta, NN, NS):
        self.NN = NN
        self.NS = NS
        self.NS_l = NS//2
        self.NS_r = NS//2 + NS%2
        self.Delta = np.concatenate(
            (Delta[-self.NS_l:], Delta[:self.NN + self.NS_r])
        )
        
    def powerlaw(self, x, amp, delta, c):
            return amp*((x+1)**-delta + (self.NN - (x))**-delta )+c
        
    def exp(self, x, amp, delta, c):
            return amp*(np.exp(-delta*(x+1)) + np.exp(-delta*((self.NN - (x)))) )+c
        
    def expstr(self, x,  amp, delta, c):
            return amp*(np.exp(-np.sqrt(delta*(x+1))) + np.exp(-np.sqrt(delta*(self.NN -x)))) + c

        
    def gen_fits(self, shift = 0, user_data = None, show_plots = False):
        self.xx = np.arange(self.NN+2*shift)
        self.shift = shift
        if user_data is None:
            data = np.real(self.Delta[self.NS_l - shift:self.NN + self.NS_l + shift])
        else:
            data = user_data
        
        model = lmf.Model(self.powerlaw)
        params = model.make_params(amp = 1.0, delta = 0.5)
        #params.add('zero', value = -0.1, max = -0.01)
        params.add('c', value = 0, vary = False)

        res_powerlaw = model.fit(data, params, x = self.xx)

        model = lmf.Model(self.exp)
        params = model.make_params(amp = 1.0, delta = 0.5)
        #params.add('zero', value = -0.01, max = 0)
        params.add('c', value = 0, vary = False)
        
        res_exp = model.fit(data, params, x = self.xx)

        model = lmf.Model(self.expstr)
        params = model.make_params(amp = 1.0)
        params.add('delta', value = 0.5, min = 0.01)
        #params.add('zero', value = -1, max = 0, vary = False)
        params.add('c', value = 0, vary = False)     

        res_expstr = model.fit(data, params, x = self.xx)
        
        if show_plots:
            res_powerlaw.plot_fit()
            plt.show()

            res_exp.plot_fit()
            plt.show()

            res_expstr.plot_fit()
            plt.show()

        results = res_powerlaw, res_exp, res_expstr
        if user_data is None:
            self.results = results
            
        return results
    
    def benchmark_fits(self, shift = 0):
        res_powerlaw, res_exp, res_expstr = self.results
        
        ref_powerlaw = self.powerlaw(self.xx,*res_powerlaw.values.values())
        ref_exp = self.exp(self.xx,*res_exp.values.values())
        ref_expstr = self.expstr(self.xx,*res_expstr.values.values())
        
        res_benchmark = self.gen_fits(user_data=ref_powerlaw), self.gen_fits(user_data=ref_exp),self.gen_fits(user_data=ref_expstr)
        
        fig, axes = plt.subplots(3,4, figsize = (18,12), sharex='all', sharey='all')
        
        col_heads = ("Actual data", "Power law\n simulated data", "Exponential\n simulated data", "Stretched exponential\n simulated data")
        row_heads = ("Power law fit", "Exponential fit", "Stretched exponential fit")
        
        for column, results_col, col_head in zip(axes.transpose(),(self.results,*res_benchmark), col_heads):
            for axis, result, row_head in zip(column,results_col, row_heads):
                result.plot_fit(ax = axis, datafmt = "C0.", fitfmt = "C1-")
                
        for axis, col_head in zip(axes[0], col_heads):
            axis.set_title(col_head, size =20)
        for axis, row_head in zip(axes[:,0], row_heads):
            axis.set_ylabel(row_head, size = 20)
            axis.set_facecolor((0.9,0.9,0.9))
                
        fig.tight_layout()
        fig.suptitle("w = {}".format(w), fontsize = "30")
        plt.show()
        return
    
########## The scaling class, fit xs and ys to various decays
        
class scaling:
    def __init__(self, Ls, Fs, L_label = "L/2", F_label = "$\Delta_{mid}$"):
        self.Ls = Ls
        self.Fs = Fs.real
        self.L_label = L_label
        self.F_label = F_label
        self.Len = len(Ls)
    
    def decay_pow(self, x, amp, delta, c):
        return amp*x**(-delta)+c

    def decay_exp(self, x, amp, delta, c):
        return amp*np.exp(-delta*x)+c

    def decay_expstr(self, x, amp, delta, c):
        return amp*np.exp(-np.sqrt(delta*x))+c

    def fit_to_pow(self):
        model = lmf.Model(self.decay_pow)
        params = model.make_params(amp = 1.0, delta = 0.5)
        params.add('c', value = 0, vary = False)

        res_pow = model.fit(self.Fs, params, x = self.Ls)
        
        mod = lmf.models.LinearModel()
        pars = mod.guess(np.log(self.Fs), x=np.log(self.Ls))
        scaling_fit = mod.fit(np.log(self.Fs),pars, x=np.log(self.Ls))
        
        fig, ax = plt.subplots(figsize = (6, 3.6))


        ax.plot(self.Ls, self.Fs, "o")
        ax.plot(self.Ls, np.exp(scaling_fit.values["intercept"])*self.Ls**scaling_fit.values["slope"],"C1-")
        ax.annotate("$\propto {}^{{{:.2f}}}$".format(self.L_label,scaling_fit.values["slope"]), (self.Ls[2]/2+self.Ls[3]/2, self.Fs[2]/2 + self.Fs[3]/2))
        ax.set_xlabel("{}".format(self.L_label))
        ax.set_ylabel("{}".format(self.F_label))
        ax.locator_params(nbins=5)
        ax.tick_params(axis = "y", labelleft = False)
        
        axin = ax.inset_axes([0.4,0.4,0.6,0.6])
        scaling_fit.plot_fit(ax = axin, xlabel = "log({})".format(self.L_label), ylabel = "log({})".format(self.F_label))
        axin.set_title("")
        axin.set_xticklabels([])
        axin.set_yticklabels([])
        axin.tick_params(axis='both', which='both', length=0)
        axin.get_legend().remove()
        
        plt.show()
        
    def fit_to_exp(self, rem = 0, plot_rem = False):
        model = lmf.Model(self.decay_exp)
        params = model.make_params(amp = 1.0, delta = 0.5)
        params.add('c', value = 0, vary = False)

#         res_exp = model.fit(self.Fs, params, x = self.Ls)
        
        mod = lmf.models.LinearModel()
        pars = mod.guess(np.log(self.Fs[:self.Len-rem]), x=self.Ls[:self.Len-rem])
        scaling_fit = mod.fit(np.log(self.Fs)[:self.Len-rem] ,pars, x=self.Ls[:self.Len-rem])
        
        fig, ax = plt.subplots(figsize = (6, 3.6))

        ax.plot(self.Ls, self.Fs, "o")
        ax.plot(self.Ls, np.exp(scaling_fit.values["intercept"])*np.exp(scaling_fit.values["slope"]*self.Ls),"C1-")
        ax.annotate("$\propto exp({:.2f}{})$".format(scaling_fit.values["slope"], self.L_label), (self.Ls[0]*0.1+self.Ls[1]*0.9, self.Fs[0]*0.1 + self.Fs[1]*0.9))
        ax.set_xlabel("{}".format(self.L_label))
        ax.set_ylabel("{}".format(self.F_label))
        ax.tick_params(axis = "y")
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1e'))
        ax.locator_params(nbins=5)
        
        axin = ax.inset_axes([0.35,0.35,0.6,0.6])
        scaling_fit.plot_fit(ax = axin, xlabel = "{}".format(self.L_label), ylabel = "log({})".format(self.F_label))
        if plot_rem:
            axin.plot(self.Ls[-3:],np.log(np.abs(self.Fs[-rem:])),"C0o")
        axin.set_title("")
        axin.set_xticklabels([])
        axin.set_yticklabels([])
        axin.tick_params(axis='both', which='both', length=0)
        axin.get_legend().remove()
        
        plt.show()
        
    def fit_to_expstr(self):
        model = lmf.Model(self.decay_expstr)
        params = model.make_params(amp = 1.0)
        params.add('c', value = 0, vary = False)
        params.add('delta', value = 0.5, min = 0.1)

        res_expstr = model.fit(self.Fs, params, x = self.Ls)
        
        mod = lmf.models.LinearModel()
        pars = mod.guess(np.log(self.Fs), x=np.sqrt(self.Ls))
        scaling_fit = mod.fit(np.log(self.Fs)[:] ,pars, x=np.sqrt(self.Ls[:]))
        
        fig, ax = plt.subplots(figsize = (6, 3.6))

        ax.plot(self.Ls, self.Fs, "o")
        ax.plot(self.Ls, np.exp(scaling_fit.values["intercept"])*np.exp(scaling_fit.values["slope"]*np.sqrt(self.Ls)),"C1-")
        ax.annotate("$\propto exp({:.2f}\sqrt{{{}}})$".format(scaling_fit.values["slope"], self.L_label), (self.Ls[2]*0.1+self.Ls[3]*0.9, self.Fs[2]*0.1 + self.Fs[3]*0.9))
        ax.set_xlabel("{}".format(self.L_label))
        ax.set_ylabel("{}".format(self.F_label))
        ax.locator_params(nbins=5)
        
        axin = ax.inset_axes([0.35,0.35,0.6,0.6])
        scaling_fit.plot_fit(ax = axin, xlabel = "$\sqrt{{{}}}$".format(self.L_label), ylabel = "log({})".format(self.F_label))
#         axin.plot(self.Ls[-3:],np.log(self.Fs[-3:]),"C0o")
        axin.set_title("")
        axin.set_xticklabels([])
        axin.set_yticklabels([])
        axin.tick_params(axis='both', which='both', length=0)
        axin.get_legend().remove()
        
        plt.show()
        
######################### Histogram management
    
        
