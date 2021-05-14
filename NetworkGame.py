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
from scipy.stats import mode
# random.seed(100)


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

####################################################################
## calulate resident-mutant fitness matrix from contributions at end of repeated cooperative
## investment game
##
## fitness = 1 + b * (partner) - c * (self) + d * (self) * (partner)
####################################################################

def fitnessOutcome (b, c, d, fitness_benefit_scale, rounds, mutWm, mutWb, mutInit, resWm, resWb, resInit):
    [rmOut, mrOut] = repeatedNetworkGame(rounds, resWm, resWb, resInit, mutWm, mutWb, mutInit)

    wmr = 1 + ((b * rmOut - c * mrOut + d * rmOut*mrOut)*fitness_benefit_scale)
    wrm = 1 + ((b * mrOut - c * rmOut + d * rmOut*mrOut)*fitness_benefit_scale)
   
    return [[wmr, wrm], [rmOut, mrOut]]

##################################################################
## calulate resident-mutant fitness matrix from contributions throughout the game history,
## discounted at rate delta going from present to the past in the repeated cooperative
## investment game
##
## fitness = 1 + b * (partner) - c * (self) + d * (self) * (partner)
##################################################################
def fitnessOutcomeEntireHist (b, c, d, fitness_benefit_scale, rounds, mutWm, mutWb, mutInit, resWm, resWb, resInit, delta):
    [rmOut, mrOut] = repeatedNetworkGameEntireHist(rounds, resWm, resWb, resInit, mutWm, mutWb, mutInit)
    discount = exp(-delta*(rounds-1-arange(0,rounds)))
    td = discount.sum()

    wmr = 1 + ((dot((b * rmOut - c * mrOut + d * rmOut*mrOut), discount))*fitness_benefit_scale)
    wrm = 1 + ((dot((b * mrOut - c * rmOut + d * rmOut*mrOut), discount))*fitness_benefit_scale)

    return [[wmr, wrm], [dot(rmOut, discount)/td, dot(mrOut, discount)/td]]


##################################################################
# Calculates fitness from games played, stores the value of each pair,
# and tracks the maximum observed fitness for each matchup type at,
# each timepoint.
##################################################################

def pairwise_fitness(population_array, repro_probabilities, genotypes_dict, fit_dict, fitFunc, w_max, init_max):
    shuffled_population = population_array.copy()
    random.shuffle(shuffled_population)
    mean_init = zeros((2*len(population_array)))
    payoffs = zeros(2*len(population_array))
    for n1, n2, site in zip(population_array, shuffled_population, range(len(population_array))):
        if n1 not in fit_dict.keys():
            fit_dict[n1] = {}
        if n2 not in fit_dict[n1].keys():
            
            fit_dict[n1][n2] = fitFunc(genotypes_dict[n2][0], genotypes_dict[n2][1], genotypes_dict[n2][2], genotypes_dict[n1][0], genotypes_dict[n1][1], genotypes_dict[n1][2])
            
        p = fit_dict[n1][n2][0]
        
        for w_i in range(2):
            if p[w_i] > w_max[w_i]:
                w_max[w_i] = p[w_i]
        
        if max([genotypes_dict[n1][2], genotypes_dict[n2][2]]) > init_max:
            init_max = max([genotypes_dict[n1][2], genotypes_dict[n2][2]])
        mean_init[site] = genotypes_dict[n1][2]
        mean_init[site+len(population_array)] = genotypes_dict[n2][2]
        payoffs[site] = fit_dict[n1][n2][1][0]
        payoffs[site+len(population_array)] = fit_dict[n1][n2][1][1]
        repro_probabilities[site] = p[0]

    return repro_probabilities, fit_dict, w_max, init_max, mean(mean_init), mean(payoffs)

##################################################################
#   Mutation function, samples each child in the offspring generation,
#   and then applies mutation to nodes, edges, and init as below
##################################################################

def mutation_process(population_size, genotypes_dict, n_genotypes, new_pop_array, matDim, mutlink, mutsize, mutinitsize, mu):
    for N in range(population_size):
        if random.rand() < mu:
            mutationw = triu(random.binomial(1,mutlink,(matDim,matDim))
                      *(random.normal(0,mutsize, (matDim,matDim))))
            mutWm = genotypes_dict[new_pop_array[N]][0] + mutationw
            mutationb = random.binomial(1,mutlink,matDim)*random.normal(0, mutsize, matDim)
            mutWb = genotypes_dict[new_pop_array[N]][1] + mutationb
            mutation_init = random.normal(0, scale = mutinitsize)
            mutInit = genotypes_dict[new_pop_array[N]][2] + mutation_init
            new_pop_array[N] = n_genotypes
            genotypes_dict[n_genotypes] = [mutWm,mutWb,mutInit]
            n_genotypes += 1
            
    return genotypes_dict, n_genotypes, new_pop_array

##################################################################
# the main simulation code, iterating the sequential mutation and invasion process
##################################################################

def simulation (initWm, initWb, initIn, population_size, mu, b, c, d, r, rounds, Tmax, mutsize, mutinitsize, mutlink, fitFunc):
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
    nethist = zeros((histlen,2))

    # save initial values
    w, nout = fitFunc(resWm, resWb, initIn, resWm, resWb, initIn)
    fithistory[0] = w[0]
    wmhist[0] = resWm
    bhist[0] = resWb
    nethist[0] = nout
    ninvas = 0
    
    ################################################################
    #   Initializing population arrays, creating dict of genotypes,
    #   array of genotype indices, dicts for caching game results,
    #   and arrays for tracking history
    ################################################################
    
    genotypes_dict = {int(0) : [resWm, resWb, initIn]}
    population_array = zeros([Tmax, population_size], dtype = int)
    n_genotypes = 1
    fit_dict = {}
    previous_resident = int32(0)
    w_max_history = zeros((Tmax, 4))
    init_max_history = zeros(Tmax)
    mean_fitness_history = zeros(Tmax)
    mean_init_history = zeros(Tmax)
    mean_payoff_history = zeros(Tmax)
    
    
    ######################
    #     Time Start     #
    ######################
    
    for i in range(Tmax):
        
        ######################
        #  per time counters #
        ######################
        w_max = [0,0,0,0]
        init_max = float(0)
        repro_probabilities = zeros(population_size)
        current_population_array = population_array[i].copy()

        ######################
        #    reproduction    #
        ######################

        repro_probabilities, fit_dict, w_max, init_max, mean_init, mean_payoff = pairwise_fitness(current_population_array, repro_probabilities, genotypes_dict, fit_dict, fitFunc, w_max, init_max)
        new_pop_array = random.choice(current_population_array, 
                                      size = shape(current_population_array), 
                                      p = (repro_probabilities/sum(repro_probabilities))).tolist()

        ######################
        #    mutation code   #
        ######################
        
        genotypes_dict, n_genotypes, new_pop_array = mutation_process(population_size, genotypes_dict, n_genotypes, new_pop_array, matDim, mutlink, mutsize, mutinitsize, mu)
            
        
        ######################################################
        #   Updating output arrays. Resident is just         #
        #   whichever genotype has the largest frequency     #
        #   in the population. If more than mode exists,     #
        #   it picks the oldest genotype.                    #
        ######################################################

        current_resident = mode(new_pop_array)[0][0]
        if current_resident != previous_resident:
            wmhist[ninvas] = genotypes_dict[current_resident][0]
            bhist[ninvas] = genotypes_dict[current_resident][1]
            fithistory[ninvas] = fit_dict[current_resident][current_resident][0][0]
            ninvas += 1
            previous_resident = current_resident


        ##############################
        # update simulation history  #
        ##############################
        w_max_history[i] = w_max
        init_max_history[i] = init_max
        mean_fitness_history[i] = mean(repro_probabilities)
        mean_init_history[i] = mean_init
        mean_payoff_history[i] = mean_payoff
        if i != Tmax-1:
            population_array[i+1] = new_pop_array.copy()
        print('net out (Δmm, Δmr, rr)  : ({:.5f}, {:.5f}, {:.5f})'.format(nout[0]-nout[2], nout[1]-nout[2], nout[2])) if verbose == 2 else None

    return {'n_invas' : ninvas,
            'n_mutants' : n_genotypes,
            'w_max_hist' : w_max_history,
            'init_max_hist' : init_max_history,
            'mean_fitness_hist' : mean_fitness_history,
            'mean_init_hist' : mean_init_history,
            'timesteps' : Tmax,
            'invas_hist' : range(Tmax),
            'fit_hist' : fithistory[0:i],
            'wm_hist' : wmhist[0:i],
            'b_hist' : bhist[0:i]}


# Plot network with igraph
# def plotNetwork(wm):
#     """
#     Plot network using igraph package

#     wm: weight matrix
#     """
#     import igraph as ig

#     # create graph of only upper triangle
#     g = ig.Graph.Weighted_Adjacency(triu(wm,1).tolist())

#     # change edge widths and vertex colors
#     g.es["width"] = ig.rescale(g.es["weight"], out_range=(0.0, 4))
#     g.vs['color'] = 'blue'
#     g.vs[0]['color'] = 'green'
#     g.vs[g.vs.indices[-1]]['color'] = 'red'

#     return ig.plot(g, "output.png")


## Main function to run simulation
def main():
    parser = ArgumentParser(prog='command', description='Evolution of interacting networks')

    pars = ['nreps', 'tmax', 'rounds', 'population_size', 'mu', 'fitness_benefit_scale', 'b', 'c', 'd', 'r', 'nnet', 'initIn', 'initstddev', 'mutsize', 'mutinitsize', 'mutlink',
            'discount', 'seed', 'outputfile']

    parsdefault = dict(zip(pars,
                           [1, 1000, 10, 100, 0.01, 0.5, 2, 1, 0, 0 , 5, 0.1, 1, 0.1, 0.01, 0.5,
                            0.9, 0, 'output.h5']))
    
    parstype    = dict(zip(pars,
                           [int, int, int, int, float, float, float, float, float, float, int, float, float, float, float, float, float, int, str]))

    parshelp    = dict(zip(pars,
                           ['Number of times to replicate simulation (default: %(default)s)',
                            'Number of time steps (invasions) (default: %(default)s)',
                            'Number of rounds of game play (default: %(default)s)',
                            'Population Size for WF selection',
                            'Probability of mutation per birth',
                            'Multiplier of the the payoff calculated  per game round',
                            'payoff benefit (default: %(default)s)',
                            'payoff cost (default: %(default)s)',
                            'payoff synergy (default: %(default)s)',
                            'relatedness coefficient (default: %(default)s)',
                            'Number of nodes in the network (default: %(default)s)',
                            'Initial network input (game contribution) (default: %(default)s)',
                            'Initial weight matrix std dev (default: %(default)s)',
                            'Size of network mutational steps (std dev) (default: %(default)s)',
                            'Size of initial offer mutational steps (std dev) (default: %(default)s)',
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
            fitFunc = lambda mutWm, mutWb, mutInit, resWm, resWb, resInit: fitnessOutcomeEntireHist(args.b, args.c, args.d, args.fitness_benefit_scale, args.rounds, mutWm, mutWb, mutInit, resWm, resWb, resInit, args.discount)
        else:
            fitFunc = lambda mutWm, mutWb, mutInit, resWm, resWb, resInit: fitnessOutcome(args.b, args.c, args.d, args.fitness_benefit_scale, args.rounds, mutWm, mutWb, mutInit, resWm, resWb, resInit)
        simoutput = simulation(initWm, initWb, args.initIn, args.population_size, args.mu, args.b, args.c, args.d, args.r, args.rounds, args.tmax, args.mutsize, args.mutinitsize, args.mutlink, fitFunc)

        for key in simoutput.keys():
            if key in data:
                data[key] = concatenate((data[key], array(simoutput[key], ndmin=1)))
            else:
                data[key] = array(simoutput[key], ndmin=1)

        print('{}: replicate {} done'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), t+1))
        print("Number of Invasions: ", simoutput['n_invas'])
        print("Number of mutants: ", simoutput['n_mutants'])
#        outputdump = simoutput

    # create hdf5 to save data
    file = h5py.File(args.outputfile, "w")

    # save data to hdf5 file
    for key in data.keys():
        file.create_dataset(key, data=data[key])

    # save parameter values as attributes
    for par in set(vars(args).keys()) - {'verbose'}:
        file.attrs[par] = getattr(args, par)

    file.close()
 #   return outputdump

# Run main function
if __name__ == "__main__":
    main()
