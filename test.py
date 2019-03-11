import numpy as np


def filetest(fname):
	line_lst = []
	file = open(fname)
	for line in file:
		split_line = line.split()
		if(len(split_line) > 1):
			line_lst.append(line.split())

	word_lst = []		# word_lst = unique words
	seq_lst = []		# line_lst -> seq_lst after number conversion
	punctuations = [',','.','?','!',':',';']
	counter = 0
	for i, line in enumerate(line_lst):
		seq_lst.append([])
		for j, word in enumerate(line):
			# bad bc I repeat code 
			has_punctuation = False 	# this is bc I want to deal with word first
			if word[-1] in punctuations:
				p = word[-1]		# punctuation at end
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

# fname = './data/shakespeare.txt'
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
        if len(words_in_line) > 1:
            if words_in_line[0] not in numbers_lst:
                for i in range(len(words_in_line)):
                    if words_in_line[i][-1] in punctuations:
                        words_in_line[i] = words_in_line[i][:-1]
                        words_in_line.append(words_in_line[i][-1]) 
                print(words_in_line)
                for word in words_in_line:
                    for i in range(len(word_lst)):
                        if word_lst[i] == word:
                            line_lst.append(word_index[i])
            
                sentence_matrix.append(line_lst)

    # returns sentence_matrix = [[1,4,2,3], [s1], [s2], ...]
    # word_lst = ['from','fairest','creatures',...]
    return sentence_matrix, word_lst

# lst of lsts
def flatten(lst_lsts):
	new_lst = []
	for i in range(len(lst_lsts)):
		for j in range(len(lst_lsts[i])):
			new_lst.append(lst_lsts[i][j])
	return new_lst

if __name__ == '__main__':
	sentences, word_lst = filetest('./data/shakespeare.txt')
	flat_sentences = flatten(sentences)
	print(max(flat_sentences))
	print(len(word_lst))

