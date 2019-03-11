########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang
# Description:  Set 5
########################################

from HMM import unsupervised_HMM
from Utility import Utility

def unsupervised_learning(X, n_states, N_iters):
    '''
    Trains an HMM using supervised learning on the file 'ron.txt' and
    prints the results.

    Arguments:
        n_states:   Number of hidden states that the HMM should have.
        X: sequence list = [[s0], [s1], [s2], ...]
    '''

    # Train the HMM.
    HMM = unsupervised_HMM(X, n_states, N_iters)

    # Print the transition matrix.
    print("Transition Matrix:")
    print('#' * 70)
    for i in range(len(HMM.A)):
        print(''.join("{:<12.3e}".format(HMM.A[i][j]) for j in range(len(HMM.A[i]))))
    print('')
    print('')

    # Print the observation matrix. 
    print("Observation Matrix:  ")
    print('#' * 70)
    for i in range(len(HMM.O)):
        print(''.join("{:<12.3e}".format(HMM.O[i][j]) for j in range(len(HMM.O[i]))))
    print('')
    print('')

    return HMM

# make shakespeare sentence
def generate_shakespeare(HMM, N_sentences, sentence_length):
    sentences = []
    for i in range(N_sentences):
        sentence, state = HMM.generate_emission(sentence_length)
        sentences.append(sentence)

    return sentences

if __name__ == '__main__':
    print('')
    print('')
    print('#' * 70)
    print("{:^70}".format("Running Code For Project 3 HMM"))
    print('#' * 70)
    print('')
    print('')

    sentence_list, word_lst = Utility.text_to_sequences2('./data/shakespeare.txt')

    hmm_model = unsupervised_learning(sentence_list, 4, 10)

    sentences = generate_shakespeare(hmm_model, 14, 10)
