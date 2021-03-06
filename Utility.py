########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Avishek Dutta
# Description:  Set 5
########################################

class Utility:
    '''
    Utility for the problem files.
    '''

    def __init__(self):
        pass

    @staticmethod
    def load_sequence(n):
        '''
        Load the file 'sequence_data<n>.txt' for a given n.

        Arguments:
            n:          Sequence index.

        Returns:
            A:          The transition matrix.
            O:          The observation matrix.
            seqs:       Input sequences.
        '''
        A = []
        O = []
        seqs = []

        # For each file:
        with open("data/sequence_data{}.txt".format(n)) as f:
            # Read the parameters.
            L, D = [int(x) for x in f.readline().strip().split('\t')]

            # Read the transition matrix.
            for i in range(L):
                A.append([float(x) for x in f.readline().strip().split('\t')])

            # Read the observation matrix.
            for i in range(L):
                O.append([float(x) for x in f.readline().strip().split('\t')])

            # The rest of the file consists of sequences.
            while True:
                seq = f.readline().strip()
                if seq == '':
                    break
                seqs.append([int(x) for x in seq])

        return A, O, seqs

    @staticmethod
    def load_ron():
        '''
        Loads the file 'ron.txt'.

        Returns:
            moods:      Sequnces of states, i.e. a list of lists.
                        Each sequence represents half a year of data.
            mood_map:   A hash map that maps each state to an integer.
            genres:     Sequences of observations, i.e. a list of lists.
                        Each sequence represents half a year of data.
            genre_map:  A hash map that maps each observation to an integer.
        '''
        moods = []
        mood_map = {}
        genres = []
        genre_map = {}
        mood_counter = 0
        genre_counter = 0

        with open("data/ron.txt") as f:
            mood_seq = []
            genre_seq = []

            while True:
                line = f.readline().strip()

                if line == '' or line == '-':
                    # A half year has passed. Add the current sequence to
                    # the list of sequences.
                    moods.append(mood_seq)
                    genres.append(genre_seq)
                    # Start new sequences.
                    mood_seq = []
                    genre_seq = []
                
                if line == '':
                    break
                elif line == '-':
                    continue
                
                mood, genre = line.split()
                
                # Add new moods to the mood state hash map.
                if mood not in mood_map:
                    mood_map[mood] = mood_counter
                    mood_counter += 1

                mood_seq.append(mood_map[mood])

                # Add new genres to the genre observation hash map.
                if genre not in genre_map:
                    genre_map[genre] = genre_counter
                    genre_counter += 1

                # Convert the genre into an integer.
                genre_seq.append(genre_map[genre])

        return moods, mood_map, genres, genre_map

    @staticmethod
    def load_ron_hidden():
        '''
        Loads the file 'ron.txt' and hides the states.

        Returns:
            genres:     The observations.
            genre_map:  A hash map that maps each observation to an integer.
        '''
        moods, mood_map, genres, genre_map = Utility.load_ron()

        return genres, genre_map

    # from the syllable dictionary file make syllable dict
    # each word has [normal ,ending]
    @staticmethod
    def make_syllable_dict():
        syllable_dict = {}
        file = open('./data/Syllable_dictionary.txt')
        for line in file:
            split_line = line.split()
            syllables = [0, 0]
            for count in split_line[1:]:
                if(count[0] == 'E'):
                    syllables[1] = int(count[1])
                else:
                    syllables[0] = int(count)
                if(len(syllables) == 2):
                    break
            syllable_dict[split_line[0]] = syllables
        return syllable_dict

    # fname = './data/shakespeare.txt'
    # takes a list of filenames [file1, file2, ...]
    @staticmethod
    def text_to_sequences2(fnames):
        # import all lines w more than 1 word
        line_lst = []
        for fname in fnames:
            file = open(fname)
            for line in file:
                split_line = line.split()
                if(len(split_line) > 1):
                    line_lst.append(line.split())

        # get unique words.
        word_lst = []       # word_lst = unique words
        seq_lst = []        # line_lst -> seq_lst after number conversion
        punctuations = [',','.','?','!',':',';','(',')']
        counter = 0
        for i, line in enumerate(line_lst):
            seq_lst.append([])
            for j, word in enumerate(line):
                # bad bc I repeat code 
                has_punctuation = False     # this is bc I want to deal with word first
                # lowercase the word
                word = word.lower()
                if word[-1] in punctuations:
                    p = word[-1]        # punctuation at end
                    word = word[:-1]    
                    has_punctuation = True
                # this always occurs. deals with the word.
                if word in word_lst:
                    seq_lst[i].append(word_lst.index(word))
                else:   
                    seq_lst[i].append(counter)
                    word_lst.append(word)
                    counter += 1
                # means that we need to do punctuation
                if(has_punctuation):    
                    if p in word_lst:
                        seq_lst[i].append(word_lst.index(p)) 
                    else:   
                        seq_lst[i].append(counter) 
                        word_lst.append(p)
                        counter += 1
        return (seq_lst, word_lst)

    # get rhyme pairs specifically for shakespeare
    def get_rhyme_pairs(seq_lst, word_lst):
        if(len(seq_lst) % 14 != 0):
            print("Exception! Poem not composed of all 14 line quatrains!")
        poem_end = list(range(12,14))
        total_lines = len(seq_lst)

        rhyme_pairs = []
        n = 0
        while n < total_lines:
            
            end_words = []
            for i in range(14):
                last_word = word_lst[seq_lst[n + i][-1]]    # for each seq, get the last word
                if last_word.isalpha():
                    end_words.append(last_word) 
                else:
                    second_to_last_word = word_lst[seq_lst[n + i][-2]]
                    end_words.append(second_to_last_word)
            n += 14

            rhyme_pair1 = [end_words[0],end_words[2]]
            rhyme_pair2 = [end_words[1],end_words[3]]
            rhyme_pair3 = [end_words[4],end_words[6]]
            rhyme_pair4 = [end_words[5],end_words[7]]
            rhyme_pair5 = [end_words[8],end_words[10]]
            rhyme_pair6 = [end_words[9],end_words[11]]
            rhyme_pair7 = [end_words[12],end_words[13]]
            rhyme_pairs.append(rhyme_pair1)
            rhyme_pairs.append(rhyme_pair2)
            rhyme_pairs.append(rhyme_pair3)
            rhyme_pairs.append(rhyme_pair4)
            rhyme_pairs.append(rhyme_pair5)
            rhyme_pairs.append(rhyme_pair6)
            rhyme_pairs.append(rhyme_pair7)
            
        return rhyme_pairs
        
    

    # fname = './data/shakespeare.txt'
    @staticmethod
    def text_to_sequences(fname):
        def load_word_list(path):
            '''
            Loads a list of the words from the file at path <path>, removing all
            non-alpha-numeric characters from the file.
            '''
            with open(path) as handle:
                # Load a list of whitespace-delimited words from the specified file
                raw_text = handle.read().strip().split()
                # Strip non-alphanumeric characters from each word
                alphanumeric_words = map(lambda word: ''.join(char for char in word), raw_text)
                # Filter out words that are now empty (e.g. strings that only contained non-alphanumeric chars)
                alphanumeric_words = filter(lambda word: len(word) > 0, alphanumeric_words)
                # Convert each word to lowercase and return the result
                return list(map(lambda word: word.lower(), alphanumeric_words))

        all_text = load_word_list(fname)

        word_lst = []
        word_index = []

        numbers_lst = []
        for number in range(1,155):
            numbers_lst.append(str(number))

        n=0
        for word in all_text:
            if word not in word_lst and word not in numbers_lst: 
                word_lst.append(word)
                word_index.append(n)
                n+=1

        punctuations = [',','.','?','!',':',';']

        n = len(word_lst)
        for punct in punctuations:
            word_lst.append(punct)
            word_index.append(n)
            n+=1
            
        for i in range(len(word_lst)):
            if word_lst[i][len(word_lst[i])-1] in punctuations:
                word_lst[i] = word_lst[i][:-1]

        sentence_matrix = []

        file = open(fname)

        for line in file:
            line_lst = []
            words_in_line=line.split()
            if words_in_line != []:
                if words_in_line[0] not in numbers_lst:
                    for i in range(len(words_in_line)):
                        if words_in_line[i][len(words_in_line[i])-1] in punctuations:
                            words_in_line[i] = words_in_line[i][:-1]
                            words_in_line.append(words_in_line[i][len(words_in_line[i])-1]) 
                    for word in words_in_line:
                        for i in range(len(word_lst)):
                            if word_lst[i] == word:
                                line_lst.append(word_index[i])
                
                    sentence_matrix.append(line_lst)

        # returns sentence_matrix = [[1,4,2,3], [s1], [s2], ...]
        # word_lst = ['from','fairest','creatures',...]
        return sentence_matrix, word_lst
