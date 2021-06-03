# -*- coding: utf-8 -*-
"""
Created on Thu May 20 09:29:09 2021

@author: linde
"""


from glob import glob
import h5py
from numpy import *
import scipy.stats as st
from scipy.integrate import quadrature
##############
#  From JVC  #
##############
def readH5Files():
    files = sort(glob('*.h5'))
    data = {}
    attrs = []
    n_files = len(files)
    for file in files:
        f = h5py.File(file,'r')

        # convert datasets into numpy arrays and save in data list
        for (k,v) in list(f.items()):
            if k not in data:
                data[k] = [ array(v) ]
            else:
                data[k].append( array(v) )
                    
        # save attributes
        attrs.append(dict(f.attrs))

        # close file
        f.close()

    simattrs = dict()
    for i in range(len(attrs)):
        for key in attrs[i]:
            keyval = key + ' = ' + str(attrs[i][key])
            if keyval not in simattrs:
                simattrs[keyval] = [i]
            else:
                simattrs[keyval].append(i)
    
    return (data, attrs, simattrs, n_files)

(data, attrs, simattrs, n_files) = readH5Files()


##############
#  EG Plots  #
##############

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats import proportion
import pandas as pd

fig, ax = plt.subplots(2,1)

N = 100
freqs = []
fixation_obs = []
conf_u = []
conf_l = []
conf = []

#forgot to explicitly add a way to check this.
#this function will only work when there are 2 genotypes in the population

###########
#Ewens 5.52
###########

def t_star(x, p):
    alpha = 2 * 100 * .05
    # part_1 = ((2*exp(-1 * alpha * x)) * (1 - exp(-1*alpha*(1-p)))*(exp(alpha*x))^2)/alpha*x*(1-x)*(1-exp(-1*alpha))*(exp(alpha*p-1))
    # part_2 = alpha*x*(1-x)*(1-exp(-1*alpha))*(exp(alpha*p-1))
    print(x, p)
    if x <= p:
        denominator = alpha*x*(1-x)*(1-exp(-1*alpha))*(exp(alpha*p-1))
        numerator = ((2*exp(-1 * alpha * x)) * (1 - exp(-1*alpha*(1-p))) *(exp(alpha*x)-1) * (exp(alpha*x)-1))
        # print(part1/part2)
        return numerator/denominator
        # return ((2*exp(-1 * alpha * x)) * (1 - exp(-1*alpha*(1-p)))*(exp(alpha*x))^2)/alpha*x*(1-x)*(1-exp(-1*alpha))*(exp(alpha*p-1))
    elif p < x:
        numerator = 2*(exp(alpha*x)-1)*(exp(alpha*(1-x))-1)
        denominator = alpha*x*(1-x)*(exp(alpha)-1)
        print(numerator, denominator)
        return numerator/denominator
 

###########
#From Daniel
###########

def u_p(p):
    # print(s)
    # exponent_1 = np.exp(-2*s*N*p)
    # exponent_2 = np.exp(-2*s*N)
    return (1 - np.exp(-2*s*N*p)) / (1- np.exp(-2*s*N))
def psi_x(x):
    # exponent_1 = np.exp(2*N*s*x)
    # exponent_2 = 1-np.exp(-2*N*s)
    # divisor = s*(1-x)*x
    return (np.exp(2*N*s*x)*(1-np.exp(-2*N*s)))/(s*(1-x)*x)
def t_int1(x):
    psi_x_p = psi_x(x)
    u_p_x = u_p(x)
    return (psi_x_p*u_p_x*(1-u_p_x))
def t_int2(x):
    u_p_x = u_p(x)
    return (psi_x(x)*u_p_x*u_p_x)
def t_bar1(p):
    tbar1_1, err_tbar1_1 = quadrature(t_int1,p,1)
    tbar1_2, err_tbar1_2 = quadrature(t_int2, 0, p)
    out = tbar1_1 + tbar1_2 * ((1-u_p(p))/u_p(p))
    return out

###########
#Ewens 3.11
###########
def t_bar(N, ps):
    out = []
    for p in ps:
        out.append(((-1 * N)/p)*(1-p)*log((1-p)))
    return out

###########
#Ewens 3.31
###########
def pi_x(N, selective_advantage, ps):
    out = []
    alpha = 2*N * selective_advantage
    for p in ps:
        out.append((1- exp(-alpha * p))/1-exp(-alpha))
    return out

def resident_fixation_check(fixation_times, fixed_genotypes, fixation_init_p):
    resident_curr_fix_times = []
    fixation_time_ps = []
    for x, y, z in zip(fixation_times, fixed_genotypes, fixation_init_p):
        if y == 0:
            resident_curr_fix_times.append(x)
            fixation_time_ps.append(z)

    return resident_curr_fix_times, fixation_time_ps



fixation_times = []
fixation_init_p = []
mean_fixation_time = []
fixed_genotypes = []
resident_fixation_times = []
fixation_time_ps = []

for file in range(n_files):
    selective_advantage = attrs[file]['resident_neutral_advantage']
    fixation_data = data['fixed_genotype'][file]
    fixed_genotypes.append(fixation_data)
    fixation_times.append(data['fixation_time'][file])
    fixation_init_p.append([attrs[file]['init_resident_freq']]*100)
    resident_curr_fix_times, fixation_time_curr_ps = resident_fixation_check(fixation_times[-1], fixed_genotypes[-1], fixation_init_p[-1])
    resident_fixation_times.append(resident_curr_fix_times)
    fixation_time_ps.append(fixation_time_curr_ps)
    fixation_init_p.append([attrs[file]['init_resident_freq']]*100)
    freqs.append(attrs[file]['init_resident_freq'])

    #since our genotypes without mutations are 0 (resident) and
    # 1 (mutant), 1 minus the mean will give us the frequency
    # of the initial resident at the end
    fixation_obs.append(1-mean(fixation_data))
    conf.append(proportion.proportion_confint(sum(fixation_data), len(fixation_data)))

conf = [[abs(x[0]), abs(x[1])] for x,z in zip(conf, fixation_data)]
conf = transpose(conf)
d = {'freqs' : freqs, 'fixation_obs' : fixation_obs, 'conf_l' : abs(conf[0]-fixation_obs), 'conf_u' : abs(conf[1]-fixation_obs),  'expected' : np.arange(0,1.05,.05)}

final_fix_time = []
final_fix_ps = []
for x, y in zip( fixation_time_ps, resident_fixation_times):
    for a, b in zip(x,y):
        final_fix_time.append(b)
        final_fix_ps.append(a)

d2 = {'init_p' : final_fix_ps, 'resident_fixation_times' : final_fix_time}
df = pd.DataFrame(data = d).sort_values('freqs')
df2 = pd.DataFrame(data = d2).sort_values('init_p')

ax[0].plot(df['freqs'], df['fixation_obs'])
ax[0].set_xlabel("Initial frequency of resident")
ax[0].set_ylim(0,1)
ax[0].set_ylabel("Proportion of replicates where resident fixes")
# ax[0].errorbar(np.linspace(0,1,21), np.linspace(0,1,21), yerr = [df['conf_l'], df['conf_u']], alpha = .3)
ax[0].plot(np.linspace(0,1,100), pi_x(N, selective_advantage, np.linspace(0,1,100)))
ax[1].scatter(df2['init_p'], df2['resident_fixation_times'], alpha = .2)
ax[1].plot(df2.groupby('init_p').mean())

t_bar_pred = [np.nan]

N = 100
s = .05
ax[0].plot(np.linspace(0,1,100), u_p(np.linspace(0,1,100)))

for p in np.arange(0.01,1,.01):

    t_bar_pred.append(t_bar1(p))

ax[1].plot(np.linspace(0,1, 100), t_bar_pred, c = 'red')

ax[1].set_ylabel("Timesteps until resident fixed")
plt.savefig("neutrality_and_fixation_tests.png")