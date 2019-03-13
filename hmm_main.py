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
    punctuation = [',','.','?','!',':',';','(',')']
    for num in seq: 
        word = str(word_lst[num])
        if word.isalpha():
            sentence += ' '
        sentence += word
        
    return sentence

# make shakespeare sonnets
def generate_shakespeare(HMM, rhyme_pairs, word_lst, syllable_dict):
    poem = ''

    rhymepairs = []
    for i in range(7):
        index = np.random.randint(0, len(rhyme_pairs))
        rhymepairs.append(rhyme_pairs[index])

    line1 = hmm_model.generate_emission_syllables(10, rhymepairs[0][0], word_lst, syllable_dict)
    line3 = hmm_model.generate_emission_syllables(10, rhymepairs[0][1], word_lst, syllable_dict)
    line2 = hmm_model.generate_emission_syllables(10, rhymepairs[1][0], word_lst, syllable_dict)
    line4 = hmm_model.generate_emission_syllables(10, rhymepairs[1][1], word_lst, syllable_dict)

    line5 = hmm_model.generate_emission_syllables(10, rhymepairs[2][0], word_lst, syllable_dict)
    line7 = hmm_model.generate_emission_syllables(10, rhymepairs[2][1], word_lst, syllable_dict)
    line6 = hmm_model.generate_emission_syllables(10, rhymepairs[3][0], word_lst, syllable_dict)
    line8 = hmm_model.generate_emission_syllables(10, rhymepairs[3][1], word_lst, syllable_dict)

    line9 = hmm_model.generate_emission_syllables(10, rhymepairs[4][0], word_lst, syllable_dict)
    line11 = hmm_model.generate_emission_syllables(10, rhymepairs[4][1], word_lst, syllable_dict)
    line10 =  hmm_model.generate_emission_syllables(10, rhymepairs[5][0], word_lst, syllable_dict)
    line12 = hmm_model.generate_emission_syllables(10, rhymepairs[5][1], word_lst, syllable_dict)

    line13 = hmm_model.generate_emission_syllables(10, rhymepairs[6][0], word_lst, syllable_dict)
    line14 = hmm_model.generate_emission_syllables(10, rhymepairs[6][1], word_lst, syllable_dict)

    poem += (line1 + '\n' + line2 + '\n' + line3 + '\n' + line4 + 
        '\n' + line5 + '\n' + line6 + '\n' + line7 + '\n' + line8 + '\n' + line9 + '\n'
        + line10 + '\n' + line11 + '\n' + line12 + '\n' + line13 + '\n' + line14 + '\n') 

    return poem


    sentences = ''
    for i in range(N_sentences):
        seq, state = HMM.generate_emission(sentence_length)
        sentence = seq_to_sentence(seq, word_lst, syllable_dict)
        sentences += sentence + '\n'

    return sentences

# make shakespeare sonnets
def generate_haiku(HMM, rhyme_pairs, word_lst, syllable_dict):
    poem = ''

    rhymepairs = []
    for i in range(3):
        index = np.random.randint(0, len(rhyme_pairs))
        rhymepairs.append(rhyme_pairs[index])

    line1 = hmm_model.generate_emission_syllables(5, rhymepairs[0][0], word_lst, syllable_dict)
    line2 = hmm_model.generate_emission_syllables(7, rhymepairs[1][0], word_lst, syllable_dict)
    line3 = hmm_model.generate_emission_syllables(5, rhymepairs[2][0], word_lst, syllable_dict)

    poem += (line1 + '\n' + line2 + '\n' + line3 + '\n') 

    return poem


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

    shakespeare_seqlst, shakespeare_wordlst = Utility.text_to_sequences2(['./data/shakespeare.txt'])
    rhyme_pairs = Utility.get_rhyme_pairs(shakespeare_seqlst, shakespeare_wordlst)
    syllable_dict = Utility.make_syllable_dict()
    hmm_model = unsupervised_learning(shakespeare_seqlst, 6, 20)
    # save the A and O
    # np.savetxt('./data/A_matrix.txt', hmm_model.A)
    # np.savetxt('./data/O_matrix.txt', hmm_model.O)

    # Make Only shakespeare sonnets
    N_poems = 3
    for j in range(N_poems):
        print("Shakespeare Sonnet ", j)
        poem = generate_shakespeare(hmm_model, rhyme_pairs, shakespeare_wordlst, syllable_dict)
        print(poem)

    # Make haikus
    N_poems = 3
    for j in range(N_poems):
        print("Haiku ", j)
        poem = generate_haiku(hmm_model, rhyme_pairs, shakespeare_wordlst, syllable_dict)
        print(poem)

    # Train a model with both spenser and shakespeare
    shake_spenser_seqlst, shake_spenser_wordlst = Utility.text_to_sequences2(['./data/shakespeare.txt', './data/spenser.txt'])
    rhyme_pairs = Utility.get_rhyme_pairs(shake_spenser_seqlst, shake_spenser_wordlst)
    syllable_dict = Utility.make_syllable_dict()
    hmm_model = unsupervised_learning(shake_spenser_seqlst, 6, 20)
    # make the shakespeare_spenser sonnets
    N_poems = 3
    for j in range(N_poems):
        print("Shakespeare Sonnet ", j)
        poem = generate_shakespeare(hmm_model, rhyme_pairs, shake_spenser_wordlst, syllable_dict)
        print(poem)


