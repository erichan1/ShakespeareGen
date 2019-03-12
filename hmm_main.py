########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang
# Description:  Set 5
########################################

from HMM import unsupervised_HMM
from Utility import Utility
import numpy as np

def unsupervised_learning(X, n_states, N_iters, ):
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

    # print("Transition Matrix:")
    # print('#' * 70)
    # for i in range(len(HMM.A)):
    #     print(''.join("{:<12.3e}".format(HMM.A[i][j]) for j in range(len(HMM.A[i]))))
    # print('')
    # print('')

    # Print the observation matrix. 

    # print("Observation Matrix:  ")
    # print('#' * 70)
    # for i in range(len(HMM.O)):
    #     print(''.join("{:<12.3e}".format(HMM.O[i][j]) for j in range(len(HMM.O[i]))))
    # print('')
    # print('')

    return HMM

def seq_to_sentence(seq, word_lst, syllable_dict):
    sentence = ''
    punctuation = [',','.','?','!',':',';']
    for num in seq: 
        word = str(word_lst[num])
        if word.isalpha():
            sentence += ' '
        sentence += word
        
    return sentence

# make shakespeare sentence
def generate_shakespeare(HMM, N_sentences, sentence_length, word_lst, syllable_dict):
    sentences = ''
    for i in range(N_sentences):
        seq, state = HMM.generate_emission(sentence_length)
        sentence = seq_to_sentence(seq, word_lst, syllable_dict)
        sentences += sentence + '\n'

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

    syllable_dict = Utility.make_syllable_dict()

    hmm_model = unsupervised_learning(sentence_list, 4, 10)

    # sentences = generate_shakespeare(hmm_model, 14, 10, word_lst, syllable_dict)

    # print(sentences)

    randomword = np.random.choice(word_lst)
    sentence = hmm_model.generate_emission_syllables(10, randomword, word_lst, syllable_dict)
