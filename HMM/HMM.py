########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]

        # print("Num States", self.L)
        # print("Num observations", self.D)
        # print("Transition matrix", self.A)
        # print("obsbervation matrix", self.O)
        # print("A_start probabilities", self.A_start)

    # find initial probability. 
    # returns P(yi|xi) 
    def init_prob(self, x_i, y_i):
        yi_yiprev = self.A_start[y_i]
        xi_given_yi = self.O[y_i][x_i]
        return yi_yiprev * xi_given_yi

    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M)]
        seqs = [['' for _ in range(self.L)] for _ in range(M)]
        
        probs[0] = [self.init_prob(x[0], i) for i in range(self.L)]

        # x_i = index of this word
        # y_i = index of first class
        # returns P(yi|xi)  
        def yi_given_xi_prob(x_i, y_i, probs, probsRow):
            xi_given_yi = self.O[y_i][x_i]

            prob_y_xprefix = []
            for j in range(self.L):
                prob_y_xprefix.append(self.A[j][y_i] * probs[probsRow-1][j])
            prob_y_xprefix = np.array(prob_y_xprefix)
            max_yprev2y = np.max(prob_y_xprefix)
            max_yprev2y_i = np.argmax(prob_y_xprefix)

            max_yprev2y = max_yprev2y * xi_given_yi
            return (max_yprev2y, max_yprev2y_i) # returns probbability, index

        # at each time step, store probability 
        for i in range(1, M):
            for j in range(self.L):
                prob, seq = yi_given_xi_prob(x[i], j, probs, i)
                probs[i][j] = prob
                seqs[i][j] = seq

        # backtrack through the sequences and probabilities array
        max_seq = ''
        np_probs = np.array(probs)
        np_seqs = np.array(seqs)

        initRow = np_probs[len(np_probs)-1]
        max_prob_j = np.argmax(initRow)
        max_seq_j = max_prob_j
        max_seq = str(max_prob_j) + max_seq 
        for i in range(len(np_seqs)-1, -1, -1):
            seqRow = seqs[i]
            max_seq_j = seqRow[max_seq_j]
            max_seq = str(max_seq_j) + max_seq 
        return max_seq

    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.

        height = M    # number of vectors in alphas
        width = self.L  # vector size within alphas

        # x_i = index of this word
        # y_i = index of first class
        # yprev_i = index of second class
        # returns P(yi|xi)  
        def yi_given_xi_prob(x_i, y_i, alphas, alpharow):
            xi_given_yi = self.O[y_i][x_i]

            sum_alpha_yiprev = 0
            for j in range(width):
                sum_alpha_yiprev += self.A[j][y_i] * alphas[alpharow-1][j]
            result = sum_alpha_yiprev * xi_given_yi
            return result

        alphas = [[0. for _ in range(width)] for _ in range(height)]

        alphas[0] = [self.init_prob(x[0], i) for i in range(width)]
        if(normalize):
            total_alpha = sum(alphas[0])
            for j in range(len(alphas[0])):
                alphas[0][j] /= total_alpha

        for i in range(1, height):
            alphas[i] = [yi_given_xi_prob(x[i], j, alphas, i) for j in range(width)]
            if(normalize):
                total_alpha = sum(alphas[i])
                for j in range(len(alphas[i])):
                    alphas[i][j] /= total_alpha
        return alphas

    
    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''
        def yi_given_xi_prob(x_i, y_i, betas, betarow):
            sum_yiprev = 0
            for j in range(width):
                O_term = self.O[j][x_i]
                A_term = self.A[y_i][j]
                betas_term = betas[betarow+1][j]
                sum_yiprev +=  A_term * betas_term * O_term
            return sum_yiprev

        M = len(x)      # Length of sequence.
        width = self.L
        height = M+1

        betas = [[0. for _ in range(width)] for _ in range(height)]

        betas[-1] = [1 for _ in range(width)]
        if(normalize):
            total_beta = sum(betas[-1])
            for j in range(len(betas[-1])):
                betas[-1][j] /= total_beta

        for i in range(height-2, 0, -1):
            betas[i] = [yi_given_xi_prob(x[i], j, betas, i) for j in range(width)]
            if(normalize):
                total_beta = sum(betas[i])
                for j in range(len(betas[i])):
                    betas[i][j] /= total_beta
        return betas

    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.
        # go from y = a to y(i+1) = b
        def get_A_ab(a, b):
            n = 0       # numerator
            d = 0       # denominator
            for i in range(len(Y)):
                y = Y[i]
                for j in range(1, len(y)):
                    if(y[j] == b and y[j-1] == a):
                        n += 1
                    if(y[j] == a):
                        d += 1
            A_a2b = n / d
            return A_a2b

        for a in range(self.L):
            for b in range(self.L):
                self.A[a][b] = get_A_ab(a, b)

        # Calculate each element of O using the M-step formulas.
        # x = w, y = z
        def get_O_wz(w, z):
            n = 0       # numerator
            d = 0       # denominator
            for i in range(len(Y)):
                y = Y[i]
                x = X[i]
                for j in range(0, len(y)):
                    if(x[j] == w and y[j] == z):
                        n += 1
                    if(y[j] == z):
                        d += 1
            O_wz = n / d
            return O_wz

        for z in range(self.L):
            for w in range(self.D):
                self.O[z][w] = get_O_wz(w, z)


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        # a_marginal[i][a] -> P(time step = i, y_i = a)
        def get_a_marginal(i, a, alphas, betas):
            n = alphas[i][a] * betas[i][a]
            d = 0
            for z in range(self.L):
                d += alphas[i][z] * betas[i][z] 
            result = n / d
            return result

        # ab_marginal[i][a][b] -> P(time step = i, y_i-1 = a, y_i = b)
        def get_ab_marginal(i, a, b, x, alphas, betas):
            x_i = x[i]
            n = alphas[i-1][a] * betas[i][b] * self.A[a][b] * self.O[b][x_i]
            d = 0
            for a_p in range(self.L):
                for b_p in range(self.L):
                    d += alphas[i-1][a_p] * betas[i][b_p] * self.A[a_p][b_p] * self.O[b_p][x_i]
            result = n / d
            return result

        # calculate probability of (a->b)
        def get_A_ab(a, b, M, a_marginal, ab_marginal):
            # numerator sums over P(yi-1=a, yi=b)
            n = 0
            d = 0
            for i in range(1, M):
                n += ab_marginal[i][a][b]
            for i in range(0, M):
                d += a_marginal[i][b]
            return (n, d)

        # get O for x=w, y=z
        def get_O_wz(w, z, x, M, a_marginal):
            # numerator sums over P(yi-1=a, yi=b)
            n = 0
            d = 0
            for i in range(0, M):
                n += a_marginal[i][z] * (x[i] == w)
                d += a_marginal[i][z]
            return (n, d)

        for iteration in range(N_iters):
            print("Iteration progress: ", iteration / N_iters)
            # print("A", self.A)
            # print("O", self.O)
            A_numerator = np.zeros((self.L, self.L))
            A_denominator = np.zeros((self.L, self.L))
            O_numerator = np.zeros((self.L, self.D))
            O_denominator = np.zeros((self.L, self.D))
            for seq_i, seq in enumerate(X):
                M = len(seq)
                alphas = np.array(self.forward(seq, normalize=True))
                betas = np.array(self.backward(seq, normalize=True))
                betas = betas[1:]

                # get y marginals
                a_marginal = [[0. for _ in range(self.L)] for _ in range(M)]
                ab_marginal = [[[0. for _ in range(self.L)] for _ in range(self.L)] for _ in range(M+1)]

                for i in range(M):
                    for a in range(self.L):
                        a_marginal[i][a] = get_a_marginal(i, a, alphas, betas)

                for i in range(1, M):
                    for a in range(self.L):
                        for b in range(self.L):
                            ab_marginal[i][a][b] = get_ab_marginal(i, a, b, seq, alphas, betas)
                
                # storing new A and O (their numerator and denominator)
                for a in range(self.L):
                    for b in range(self.L):
                        n, d = get_A_ab(a, b, M, a_marginal, ab_marginal)
                        A_numerator[a][b] += n
                        A_denominator[a][b] += d
                        # get_A_ab(a, b) is really A_ba

                for z in range(self.L):
                    for w in range(self.D):
                        n, d = get_O_wz(w, z, seq, M, a_marginal)
                        O_numerator[z][w] += n
                        O_denominator[z][w] += d
                # print("start!")
                # print(ab_marginal[1])
                # print(ab_marginal[-2])

            # apply changes to A and O
            new_A = A_numerator / A_denominator
            new_O = O_numerator / O_denominator
            self.A = np.ndarray.tolist(new_A)
            self.O = np.ndarray.tolist(new_O)


    # given 1D arr, find the point where p should go. 
    # p > cumulative up to arr[i], but less than arr[i+1]
    def choose_index_wprob(self, p, arr):
        for i in range(len(arr)):
            p -= arr[i]
            if(p <= 0):
                return i
        return len(arr)-1

    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        # start
        p = random.uniform(0, 1)
        y = self.choose_index_wprob(p, self.A_start)
        states.append(y)
        p = random.uniform(0, 1)
        x = self.choose_index_wprob(p, self.O[y])
        emission.append(x)

        for i in range(1, M):
            p = random.uniform(0, 1)
            y = self.choose_index_wprob(p, self.A[y])
            states.append(y)
            p = random.uniform(0, 1)
            x = self.choose_index_wprob(p, self.O[y])
            emission.append(x)

        return emission, states

    # generates an emission of length M, given a starting position in the markov chain.
    def generate_emission_backwards(self, M, y_start):
        emission = []
        states = []
        np_A = np.array(self.A)
        np_O = np.array(self.O)

        # start
        p = random.uniform(0, 1)
        y = y_start
        states.append(y)
        p = random.uniform(0, 1)
        x = self.choose_index_wprob(p, np_O[y])
        emission.append(x)

        for i in range(1, M):
            p = random.uniform(0, 1)
            y = self.choose_index_wprob(p, np_A[:,y])
            states.append(y)
            p = random.uniform(0, 1)
            x = self.choose_index_wprob(p, np_O[:,y])
            emission.append(x)

        # emission and states are reversed. reverse them to get it right
        list.reverse(emission)
        list.reverse(states)
        return emission, states

    # generates an emission with number of syllables restricted
    # basically just regenerates the word until you get right number of syllables. 
    # also generates backwards
    def generate_emission_syllables(self, N_syllables, y_start, syllable_dict):
    '''
        N_syllables = int = number of syllables you want
        y_start = int = starting y position in markov chain
    '''
        emission = []
        states = []
        np_A = np.array(self.A)
        np_O = np.array(self.O)

        # start
        p = random.uniform(0, 1)
        y = y_start
        states.append(y)
        p = random.uniform(0, 1)
        x = self.choose_index_wprob(p, np_O[y])
        emission.append(x)

        for i in range(1, M):
            p = random.uniform(0, 1)
            y = self.choose_index_wprob(p, np_A[:,y])
            states.append(y)
            p = random.uniform(0, 1)
            x = self.choose_index_wprob(p, np_O[:,y])
            emission.append(x)

        # emission and states are reversed. reverse them to get it right
        list.reverse(emission)
        list.reverse(states)
        return emission, states



    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)
        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)
    random.seed(2019)
    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
