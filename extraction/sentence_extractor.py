import nltk
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk import WordNetLemmatizer


def pos_tagging(sentence):
    # lists to store the parts of the sentence
    grams_list = []
    bag_of_words = []

    # tokenize the sentence
    data_tokens = nltk.word_tokenize(sentence)

    # tags dictionary
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    # lemmatization function
    lemma_function = WordNetLemmatizer()

    words_matrix = [[]]
    tags_matrix = [[]]
    connector_idx = 0
    # iterates over the pos tag annotated tokens
    for token, tag in nltk.pos_tag(data_tokens):
        # filter by verbs, names and adjectives
        if tag[0:2] == 'VB' or tag[0:2] == 'VB' or tag[0:2] == 'NN' or tag[0:2] == 'JJ':
            # lemmatizes the word
            lemma = lemma_function.lemmatize(token, tag_map[tag[0]])
            # append the word and tag to the lists
            tags_matrix[connector_idx].append(tag[0:2])
            words_matrix[connector_idx].append(lemma)
            # append the word to the bag of words list
            bag_of_words.append(lemma)
        elif tag[0:2] == 'CC' or tag[0:2] == ',' or tag[0:2] == ':':
            # creates a new list to the new part of the sentence
            tags_matrix.append([])
            words_matrix.append([])
            connector_idx += 1

    patterns = """big_chunk:{<VB>*<JJ><NN>}"""
    chunker = nltk.RegexpParser(patterns)

    # method to search for tuple in the result
    def chunk_search(chunk_parent):
        for chunk_node in chunk_parent:
            if type(chunk_node) is nltk.Tree and len(chunk_node) == 3:
                # converts the tuples to lists
                words_unzip, tags_unzip = zip(*chunk_node)
                return list(words_unzip), list(tags_unzip)
        return None

    data2remove = []
    # in big sentences parts extract the useful words
    for idx, partial_sentence in enumerate(words_matrix):
        # if is bigger than 3 words
        if len(partial_sentence) > 3:
            chunk_result = chunker.parse(list(zip(partial_sentence, tags_matrix[idx])))
            tuple_r = chunk_search(chunk_result)
            if tuple_r is not None:
                words_matrix[idx], tags_matrix[idx] = tuple_r
            else:
                data2remove.append((words_matrix[idx], tags_matrix[idx]))

    # removes the big sentence indexes
    for words, tags in data2remove:
        tags_matrix.remove(tags)
        words_matrix.remove(words)

    # join the words in n-grams
    for idx, partial_sentence in enumerate(words_matrix):
        if len(partial_sentence) > 0:
            grams_list.append(' '.join(partial_sentence))

    return grams_list
