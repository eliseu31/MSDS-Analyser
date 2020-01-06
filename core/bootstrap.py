from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
import string
import itertools
import re


class BootstrapLabels:
    seed_tuples = {('store', 'cool'),
                   ('cause', 'allergic'),
                   ('flush', 'water'),
                   ('remove', 'exposure'),
                   ('remove', 'fresh', 'air'),
                   ('avoid', 'heat'),
                   ('medical', 'eyes'),
                   ('keep', 'container', 'closed'),
                   ('vapor pressure', 'container', 'rupture'),
                   ('dispose', 'federal', 'regulations')}
    # punctuation set
    punctuation = set(string.punctuation)

    def __init__(self):
        # pattern found based in that tuples
        self.patterns_founded = set()
        # new tuples founded based in the pattern
        self.tuples_founded = set()

    def search_patterns(self, data_row):
        # tokenizes the sentences
        sentence_tokens = sent_tokenize(data_row.lower())
        # iterate over each sentence
        for sentence in sentence_tokens:
            # search patterns using regex
            self.regex_pattern_extraction(sentence)

    def regex_pattern_extraction(self, sentence):
        # tokenizes the sentence
        tokens = word_tokenize(sentence)
        # check if exists any seed in the tokens
        all_tuples = map(lambda x_tuple: x_tuple if all(x in tokens for x in x_tuple) else None,
                         self.seed_tuples)
        existent_tuples = list(filter(lambda x_tuple: x_tuple is not None, all_tuples))
        # replace the tuples to add new patterns
        for current_tuple in existent_tuples:
            # filter word function
            def filter_function(x_token):
                # check if is from the current tuple
                if x_token in current_tuple:
                    return r'\s' + r'(\w+\s?\w+\s?\w+)'
                # check if is a stop word
                elif x_token in stopwords.words():
                    return r'\s' + r'\w+'
                # check if is punctuation
                elif x_token in self.punctuation:
                    return r'[a-zA-Z0-9]'
                # otherwise returns the original word
                else:
                    # transform special characters
                    x_token = re.escape(x_token)
                    return r'\s' + x_token
            # replace the seed words
            pattern_tokens = map(filter_function, tokens)
            # build the string pattern
            pattern_str = ''.join(list(pattern_tokens))
            # compile the actual pattern (remove the first space)
            pattern = re.compile(pattern_str[2:], re.IGNORECASE)
            # adds the pattern to the list
            self.patterns_founded.add(pattern.pattern)

    def search_tuples(self, data_row):
        # tokenizes the sentences
        sentence_tokens = sent_tokenize(data_row.lower())
        # iterate over each sentence
        for sentence in sentence_tokens:
            # search patterns using regex
            self.regex_tuple_extraction(sentence)

    def regex_tuple_extraction(self, sentence):
        # function to execute each pattern
        def run_pattern(x_pattern):
            # search for the pattern
            search_result = re.search(x_pattern, sentence, re.M)
            # check the result
            if search_result is not None:
                # get the groups
                groups_tuple = search_result.groups()
                # splits all the results
                tuples_list = list(map(lambda x_item: tuple(x_item.split(' ')), groups_tuple))
                # joins all the entities
                return tuple(itertools.chain(*tuples_list))
            # otherwise returns Nne
            return None

        # check which patterns founds
        original_tuples = map(run_pattern, self.patterns_founded)
        # filter the tuples
        filtered_tuples = set(filter(lambda x_tuple: x_tuple is not None, original_tuples))
        # update the actual set
        self.tuples_founded.update(filtered_tuples)

    def fit(self):
        pass

    def transform(self, data_row):
        pass
