from sklearn.base import BaseEstimator
from nltk import sent_tokenize
from collections import Counter
from scipy.sparse import csr_matrix
from joblib import delayed, Parallel
from itertools import chain
import numpy as np
import spacy

# load spaCy mode
spacy_model = spacy.load('en')


class SpacyRelationExtraction(BaseEstimator):

    def __init__(self, n_relation=20, n_jobs=8):
        # stores the number of jobs
        self.n_jobs = n_jobs
        # list where are stored the relations
        self.relations_list = []
        # stores the number of relations
        self.n_relations = n_relation

    def get_feature_names(self):
        # convert each relation tuple to one string
        str_relations = list(map(lambda x: ' '.join(x), self.relations_list))
        # return the string list
        return str_relations

    def fit(self, x, y=None):
        # method to apply to each row
        def row_scan(data_row):
            # row relations
            row_relations = []
            # tokenizes the sentences
            sentence_tokens = sent_tokenize(data_row.lower())
            # iterate over each sentence
            for sentence in sentence_tokens:
                # extract relations using spacy
                curr_relations = self.extract_currency_relations(sentence)
                # appends the relations to the list
                row_relations += curr_relations
            # return the row relations
            return row_relations

        # apply to each row the method
        # relations_matrix = list(x.apply(row_scan).values)
        relations_matrix = self.parallelize_apply(x, row_scan)
        join_relations = chain.from_iterable(relations_matrix)
        # sort according to the count
        sorted_relations = Counter(join_relations).most_common(self.n_relations)
        # filter the most popular
        self.relations_list, _ = zip(*sorted_relations)

    def transform(self, x, y=None):
        # method to apply to each row
        def row_scan(data_row):
            # initializes the results dict
            row_results = [0] * self.n_relations
            # tokenizes the sentences
            sentence_tokens = sent_tokenize(data_row.lower())
            # iterate over each sentence
            for sentence in sentence_tokens:
                # extract relations using spacy
                curr_relations = self.search_currency_relations(sentence)
                # updates the actual results
                row_results = [sum([curr_r, tot_r]) for curr_r, tot_r in zip(curr_relations, row_results)]
            # returns the list with the results
            return row_results

        # apply to each row the method
        transformed_data = self.parallelize_apply(x, row_scan)
        # returns the transformed data
        return csr_matrix(transformed_data)

    def fit_transform(self, x, y=None):
        # fit the estimator to the x
        self.fit(x)
        # get the values from the transform
        transformed_data = self.transform(x)
        # returns the transformed data
        return transformed_data

    def parallelize_apply(self, df, function):
        # splits the data
        data_chunks = np.array_split(df, self.n_jobs)
        # uses parallel processes
        with Parallel(n_jobs=self.n_jobs) as parallel:
            # process the files
            jobs_return = parallel(delayed(lambda chunk, f: list(chunk.apply(f).values))(chunk, function)
                                   for chunk in data_chunks)
            # joins all the jobs returns
            parallel_return = list(chain.from_iterable(jobs_return))
        # returns the jobs return
        return parallel_return

    def search_currency_relations(self, sentence):
        # extract the current relations
        curr_relations = self.extract_currency_relations(sentence)
        # check if the relations are in the list
        search_results = list(map(lambda x_relation: 1 if x_relation in curr_relations else 0,
                                  self.relations_list))
        # returns the results
        return search_results

    def extract_currency_relations(self, sentence):
        # creates the model to each sentence
        doc = spacy_model(sentence)
        # Merge entities and noun chunks into one token
        spans = list(doc.ents) + list(doc.noun_chunks)
        spans = self.filter_spans(spans)
        # reconstructs the tokenizer
        with doc.retokenize() as retokenizer:
            for span in spans:
                retokenizer.merge(span)
        # get the relations between the entities
        relations = []
        spans_text = list(map(lambda x: x.text, spans))
        for token in filter(lambda x: x.text in spans_text, doc):
            # checks the syntactic relation
            if token.dep_ in ("attr", "dobj"):
                # finds the subject
                subject = [w for w in token.head.lefts if w.dep_ == "nsubj"]
                if subject:
                    relations.append((subject[0].lemma_, token.lemma_))
            elif token.dep_ == "pobj" and token.head.dep_ == "prep":
                relations.append((token.head.head.lemma_, token.lemma_))
        return relations

    @staticmethod
    def filter_spans(spans):
        # Filter a sequence of spans so they don't contain overlaps
        # For spaCy 2.1.4+: this function is available as spacy.util.filter_spans()
        sorted_spans = sorted(spans,
                              key=lambda x_span: (x_span.end - x_span.start, -x_span.start),
                              reverse=True)
        result = []
        seen_tokens = set()
        for span in sorted_spans:
            # Check for end - 1 here because boundaries are inclusive
            if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
                result.append(span)
            seen_tokens.update(range(span.start, span.end))
        result = sorted(result, key=lambda x_span: x_span.start)
        return result
