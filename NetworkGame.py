#!/usr/bin/env python3
"""
Evolution of interacting networks.

This implements the evolution of the sequentially processing network of Le Nagard, Chao and Tenaillon, 2011, BMC Evolutionary Biology, for two interacting networks that together have to match an environment with varying complexity, represented by a Legendre Polynomial of varying order.

Erol Akcay <eakcay@sas.upenn.edu>
Jeremy Van Cleve <vancleve@santafe.edu>
"""

import sys
from argparse import ArgumentParser
from numpy import *
import h5py
from datetime import datetime
import matplotlib.pyplot as plt
from random import shuffle

## define the function to generate the output of node j given previous outputs and weights and Wb
def calcOj (j, prev_out, Wm, Wb):
    x = dot(Wm[0:j,j],prev_out[0:j]) + Wb[j]
    return 1-exp(-x*x)

## iterate over the whole network
def iterateNetwork (input, Wm, Wb):
    prev_out=zeros(Wb.shape[0])
    prev_out[0]=input
    for j in range(1, Wb.shape[0]):
        prev_out[j]=calcOj(j, prev_out, Wm, Wb)
    return prev_out

## one round of two-player game
def networkGameRound (mutWm, mutWb, mutPrev, resWm, resWb, resPrev):
    mutOut = iterateNetwork(resPrev, mutWm, mutWb)[-1]
    resOut = iterateNetwork(mutPrev, resWm, resWb)[-1]

    return [mutOut, resOut]

## play `rounds` of the two-player game
def repeatedNetworkGame (rounds, mutWm, mutWb, mutInit, resWm, resWb, resInit):
    for i in range(rounds):
        [mutInit, resInit] = networkGameRound(mutWm, mutWb, mutInit, resWm, resWb, resInit)

    return [mutInit, resInit]

## play rounds but now save the entire history
def repeatedNetworkGameEntireHist (rounds, mutWm, mutWb, mutInit, resWm, resWb, resInit):
    muthist = zeros(rounds)
    resthist = zeros(rounds)
    for i in range(rounds):
        [mutInit, resInit] = networkGameRound(mutWm, mutWb, mutInit, resWm, resWb, resInit)
        muthist[i] = mutInit
        resthist[i] = resInit

    return [muthist, resthist]

##
## calulate resident-mutant fitness matrix from contributions at end of repeated cooperative
## investment game
##
## fitness = 1 + b * (partner) - c * (self) + d * (self) * (partner)
##
def fitnessOutcome (b, c, d, rounds, mutWm, mutWb, mutInit, resWm, resWb, resInit):
    [mmOut, mmOut] = repeatedNetworkGame(rounds, mutWm, mutWb, mutInit, mutWm, mutWb, mutInit)
    [rmOut, mrOut] = repeatedNetworkGame(rounds, resWm, resWb, resInit, mutWm, mutWb, mutInit)
    [rrOut, rrOut] = repeatedNetworkGame(rounds, resWm, resWb, resInit, resWm, resWb, resInit)

    wmm = 1 + (b - c) * mmOut + d * mmOut**2
    wmr = 1 + b * rmOut - c * mrOut + d * rmOut*mrOut
    wrm = 1 + b * mrOut - c * rmOut + d * rmOut*mrOut
    wrr = 1 + (b - c) * rrOut + d * rrOut**2

    return [[wmm, wmr, wrm, wrr], [mmOut, mrOut, rrOut]]




##
## calulate resident-mutant fitness matrix from contributions throughout the game history,
## discounted at rate delta going from present to the past in the repeated cooperative
## investment game
##
## fitness = 1 + b * (partner) - c * (self) + d * (self) * (partner)
##
def fitnessOutcomeEntireHist (b, c, d, rounds, mutWm, mutWb, mutInit, resWm, resWb, resInit, delta):
    [mmOut, mmOut] = repeatedNetworkGameEntireHist(rounds, mutWm, mutWb, mutInit, mutWm, mutWb, mutInit)
    [rmOut, mrOut] = repeatedNetworkGameEntireHist(rounds, resWm, resWb, resInit, mutWm, mutWb, mutInit)
    [rrOut, rrOut] = repeatedNetworkGameEntireHist(rounds, resWm, resWb, resInit, resWm, resWb, resInit)
    discount = exp(-delta*(rounds-1-arange(0,rounds)))
    td = discount.sum()

    wmm = 1 + dot(((b - c) * mmOut[:] + d * mmOut**2), discount)
    wmr = 1 + dot((b * rmOut - c * mrOut + d * rmOut*mrOut), discount)
    wrm = 1 + dot((b * mrOut - c * rmOut + d * rmOut*mrOut), discount)
    wrr = 1 + dot(((b - c) * rrOut + d * rrOut**2), discount)

    return [[wmm, wmr, wrm, wrr], [dot(mmOut, discount)/td, dot(mrOut, discount)/td, dot(rrOut, discount)/td]]



def pairwise_fitness(population_array, genotypes_dict, fit_dict, initIn, fitFunc, w_max):
    w_max_copy = w_max.copy()
    for n1 in set(population_array[-1]):
        for n2 in set(population_array[-1]):
            payoff_matrix, nout = fitFunc(genotypes_dict[n2][0], genotypes_dict[n2][1], initIn, genotypes_dict[n1][0], genotypes_dict[n1][1], initIn)
            # payoff_matrix
            # fitnessOutcomeEntireHist (b, c, d, rounds, mutWm, mutWb, mutInit, resWm, resWb, resInit, delta)
            # invas_out = invasion(genotypes_dict[n2][0], genotypes_dict[n2][1], initIn, genotypes_dict[n1][0], genotypes_dict[n1][1], initIn,invasDemogFunc, fitFunc )
            for w in range(len(w_max)):
                if payoff_matrix[w] > w_max[w]:
                    w_max_copy[w] = payoff_matrix[w]
            try:
                fit_dict[n1][n2] = payoff_matrix
                
            except KeyError:
                fit_dict[n1] = {n2 : payoff_matrix}

    return fit_dict, w_max_copy


# the main simulation code, iterating the sequential mutation and invasion process
def simulation (initWm, initWb, initIn, population_size, mu, b, c, d, r, rounds, Tmax, mutsize, mutlink, fitFunc):
    """
    initWm are the initial network weights
    initWb are the initial node weights
    b is the benefit
    c is the cost
    d is the synergy
    r is the relatedness
    rounds is the number of rounds of game play
    Tmax is the maximum number of iterations
    mutsize: 2*maximum mutation magnitude
    mutlink: probability that an entry will mutate
    """
    matDim=len(initWm)
    resWm = initWm
    resWb = initWb
    histlen = Tmax+1
    invashist = zeros(histlen, dtype=int64)
    fithistory = zeros(histlen)
    wmhist = zeros((histlen,) + resWm.shape)
    bhist = zeros((histlen,) + resWb.shape)
    nethist = zeros((histlen,3))


    #4/19/21, EG
    #adding population arrays, creating dict of genotypes and array of genotype indices
    genotypes_dict = {int(0) : [resWm, resWb]}
    population_array = [0] * population_size


    # demography function
    # invasDemogFunc = lambda w: islandGroupSel(w, r)

    # save initial values
    w, nout = fitFunc(resWm, resWb, initIn, resWm, resWb, initIn)
    fithistory[0] = w[0]
    wmhist[0] = resWm
    bhist[0] = resWb
    nethist[0] = nout
    ninvas = 0
    # introduce a mutant by changing an element at random in Wm and b

    #4/19/21, EG
    #adding population arrays, creating dict of genotypes and array of genotype indices
    genotypes_dict = {int(0) : [resWm, resWb]}
    population_array = [[0] * population_size]
    n_genotypes = 1
    w_max_history = []
    fit_dict = {}
    previous_resident = int32(0)
    w_max = [0,0,0,0]
    for i in range(Tmax):
        
        if i % 50 == 0:
            print(i)
        #calculate the pairwise payoff of each genotype created
        #only runs when new genotypes are added
        if n_genotypes != len(fit_dict.keys()):
            fit_dict, w_max = pairwise_fitness(population_array, genotypes_dict, fit_dict, initIn, fitFunc, w_max)
        w_max_history.append(w_max)
        
        #creates a copy of the population array, shuffles it
        #then compares element-wise to the original to determine
        #offspring of each index in the offspring array
        shuffled_population = population_array[-1].copy()
        shuffle(shuffled_population)
        new_pop_array = population_array[-1].copy()
        repro_probabilities = [fit_dict[n1][n2][1] for n1, n2 in zip(population_array[-1], shuffled_population)]
        new_pop_array = random.choice(population_array[-1], size = shape(population_array[-1]), p = (repro_probabilities/sum(repro_probabilities))).tolist()

        #mutation code
        mutant_array = random.choice([0,1], size = shape(new_pop_array), p = [(1-mu), mu])
        if len(set(new_pop_array)) < 2:
            for N in range(population_size):
                if mutant_array[N] == 1:
                    mutationw = triu(random.binomial(1,mutlink,(matDim,matDim))
                              *(random.normal(0,mutsize, (matDim,matDim))))
                    mutWm = genotypes_dict[new_pop_array[N]][0] + mutationw
                    mutationb = random.binomial(1,mutlink,matDim)*random.normal(0, mutsize, matDim)
                    mutWb = genotypes_dict[new_pop_array[N]][1] + mutationb
                    new_pop_array[N] = n_genotypes
                    genotypes_dict[n_genotypes] = [mutWm,mutWb]
                    n_genotypes += 1
        
        
        
        #updating output arrays
        #resident is just whichever genotype has the
        #largest frequency in the population
        current_resident = max(set(population_array[-1]), key=population_array[-1].count)
        if current_resident != previous_resident:
            wmhist[ninvas] = genotypes_dict[current_resident][0]
            bhist[ninvas] = genotypes_dict[current_resident][1]
            fithistory[ninvas] = fit_dict[current_resident][current_resident][0]
            nethist[ninvas] = fit_dict[current_resident][current_resident][3]
            ninvas += 1
            previous_resident = current_resident
        population_array.append(new_pop_array.copy())

        print('net out (Δmm, Δmr, rr)  : ({:.5f}, {:.5f}, {:.5f})'.format(nout[0]-nout[2], nout[1]-nout[2], nout[2])) if verbose == 2 else None

    return {'n_invas' : ninvas,
            'n_mutants' : n_genotypes,
            'w_max_hist' : w_max_history,
            'timesteps' : Tmax,
            'invas_hist' : range(Tmax),
            'fit_hist' : fithistory[0:i],
            'wm_hist' : wmhist[0:i],
            'b_hist' : bhist[0:i],
            'net_hist' : nethist[0:i]}

# Plot network with igraph
def plotNetwork(wm):
    """
    Plot network using igraph package

    wm: weight matrix
    """
    import igraph as ig

    # create graph of only upper triangle
    g = ig.Graph.Weighted_Adjacency(triu(wm,1).tolist())

    # change edge widths and vertex colors
    g.es["width"] = ig.rescale(g.es["weight"], out_range=(0.0, 4))
    g.vs['color'] = 'blue'
    g.vs[0]['color'] = 'green'
    g.vs[g.vs.indices[-1]]['color'] = 'red'

    return ig.plot(g, "output.png")


## Main function to run simulation
def main():
    parser = ArgumentParser(prog='command', description='Evolution of interacting networks')

    pars = ['nreps', 'tmax', 'rounds', 'population_size', 'mu', 'b', 'c', 'd', 'r', 'nnet', 'initIn', 'initstddev', 'mutsize', 'mutlink',
            'discount', 'seed', 'outputfile']

    parsdefault = dict(zip(pars,
                           [1, 1000, 10, 100, 0.01, 2, 1, 1, 0.0, 6, 0.1, 1, 0.1, 0.5,
                            0.9, 0, 'output.h5']))
    
    parstype    = dict(zip(pars,
                           [int, int, int, int, float, float, float, float, float, int, float, float, float, float, float, int, str]))

    parshelp    = dict(zip(pars,
                           ['Number of times to replicate simulation (default: %(default)s)',
                            'Number of time steps (invasions) (default: %(default)s)',
                            'Number of rounds of game play (default: %(default)s)',
                            'Population Size for WF selection',
                            'Probability of mutation per birth',
                            'payoff benefit (default: %(default)s)',
                            'payoff cost (default: %(default)s)',
                            'payoff synergy (default: %(default)s)',
                            'relatedness coefficient (default: %(default)s)',
                            'Number of nodes in the network (default: %(default)s)',
                            'Initial network input (game contribution) (default: %(default)s)',
                            'Initial weight matrix std dev (default: %(default)s)',
                            'Size of mutational steps (std dev) (default: %(default)s)',
                            'Probability weight mutates (default: %(default)s)',
                            'Payoff discount rate (negative value = use last round) (default: %(default)s)',
                            'seed for random number generator; if set to zero, use system time (default: %(default)s)',
                            'output file name for .h5 file (default: %(default)s)']))

    for par in pars:
        parser.add_argument("--"+par, default=parsdefault[par],
                            type=parstype[par], help=parshelp[par])

    parser.add_argument('-v', '--verbose', action="count",
                        help="increase output verbosity")

    args = parser.parse_args()

    global verbose
    verbose = args.verbose

    # set random seed specifically if given (set to system clock otherwise)
    if (args.seed):
        random.seed(args.seed)

    data = dict()
    for t in range(args.nreps):
        print('{}: running replicate {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), t+1))
        sys.stdout.flush() # flush stdout to make sure the print statement

        initWm = random.normal(0, args.initstddev, (args.nnet, args.nnet))
        initWb = random.normal(0, args.initstddev, (args.nnet))

        if args.discount >= 0:
            fitFunc = lambda mutWm, mutWb, mutInit, resWm, resWb, resInit: fitnessOutcomeEntireHist(args.b, args.c, args.d, args.rounds, mutWm, mutWb, mutInit, resWm, resWb, resInit, args.discount)
        else:
            fitFunc = lambda mutWm, mutWb, mutInit, resWm, resWb, resInit: fitnessOutcome(args.b, args.c, args.d, args.rounds, mutWm, mutWb, mutInit, resWm, resWb, resInit)
        simoutput = simulation(initWm, initWb, args.initIn, args.population_size, args.mu, args.b, args.c, args.d, args.r, args.rounds, args.tmax, args.mutsize, args.mutlink, fitFunc)

        for key in simoutput.keys():
            if key in data:
                data[key] = concatenate((data[key], array(simoutput[key], ndmin=1)))
            else:
                data[key] = array(simoutput[key], ndmin=1)

        print('{}: replicate {} done'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), t+1))
        print("Number of Fixed Invasions: ", simoutput['n_invas'])
        outputdump = simoutput
    plotNetwork(simoutput["wm_hist"][0])
    # create hdf5 to save data
    file = h5py.File(args.outputfile, "w")

    # save data to hdf5 file
    for key in data.keys():
        file.create_dataset(key, data=data[key])

    # save parameter values as attributes
    for par in set(vars(args).keys()) - {'verbose'}:
        file.attrs[par] = getattr(args, par)

    # file.close()
    return outputdump

# Run main function
if __name__ == "__main__":
    output = main()
    
    #plots the final replicate
    fig, ax = plt.subplots()
    for w, label in zip(array(output['w_max_hist']).T.tolist(), ['rr', 'mr', 'rm', 'mm']):
        ax.scatter(output['invas_hist'], w, label = label)
    plt.xlabel('Timesteps')
    plt.ylabel("fitness")
    ax.legend()