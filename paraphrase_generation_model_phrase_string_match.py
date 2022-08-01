from __future__ import unicode_literals
import os
from paraphrase_generation.models.paraphrase_generation_model_base import \
    ParaphraseGenerationModelBase
import en_core_web_sm
import textacy
import pandas
import pickle
import math
import kenlm
import re
from math import exp
from shared.nlp_tools.tokenizer import nltk_tokenize, nltk_detokenize
from collections import OrderedDict
from nltk.corpus import stopwords
import pandas as pd
import ast
from shared.nlp_tools.latex_utils import mask_all_latex


class ParaphraseGenerationModelPhraseStringMatch(ParaphraseGenerationModelBase):
    """
    This paraphrase generation model handles the logic related to generating
    paraphrases by extracting the verb phrase,noun phrase and n-gram  from the input sentence and finding
    its paraphrase from PPDB data set (You can learn more about the PPDB data set
    by visiting http://paraphrase.org/#/download). Please note that exact match
    of verb phrase will be searched in PPDB data set. The verb phrase will be extracted
    with the help of spacy and paraphrases will be inserted into the input sentence.
    If there is more than one match, list of paraphrases will be return by this model.

    If no paraphrases are found in PPDB dataset for the verb phrase then an empty list will
    be returned.

    Note: To read more about the different models for generating paraphrases and its documentation
    please check https://docs.google.com/document/d/19_OUc3pnY2Jf6SatPqXY0gNvHdXxwCsDRTgSmHDIliw/edit
    """

    def __init__(self, logging, config_path='./config_dev.json', load_ppdb_data=False,
                 ppdb_dataset_path="./paraphrase_generation/dataset/PPDB_M_TSV_DATASET.tsv",
                 use_previous_generated_paraphrases=False, path_of_previous_generated_paraphrase_file="",
                 ppdb_dict_pkl_file=""):
        """
        Initializes the basic configuration for paraphrase generation model.

        :param config_path: path of config file (String)
        :param ppdb_dataset_path: Path of processed ppdb dataset in format (sentence_1, sentence_2, ppdb1_score, ppdb2_score) (String)
        :param use_previous_generated_paraphrases: If this is enabled, then it will fetch generated paraphrases from file to save time.(Bool)
        :param path_of_previous_generated_paraphrase_file: path of previous generated paraphrase file (String)
        :param logging: for writing logs (logging object)
        :param ppdb_dict_pkl_file: path of preprocessed ppdb dict file

        Note:- To generate preprocess ppdb dataset please use script generate_csv_file_from_ppdb_dataset.py in scripts folder

        """
        super().__init__(logging, config_path)

        if load_ppdb_data:
            self._ppdb_dataset = self._get_ppdb_dataset(ppdb_dataset_path, ppdb_dict_pkl_file)
            self._logging.info('Size of ppdb dict %s', len(self._ppdb_dataset))
        else:
            self._ppdb_dataset = {}

        # self._nlp = spacy.load('en_core_web_lg')
        self._nlp = en_core_web_sm.load()
        self._use_previous_generated_paraphrases = use_previous_generated_paraphrases
        if use_previous_generated_paraphrases:
            self._previous_generated_paraphrases_dict = self._load_previous_generated_paraphrases_from_file(
                path_of_previous_generated_paraphrase_file)
        self._lm_weight = 1.0
        self._ppdbscore_weight = 1.0
        self._enable_spell_check = self._config.get('enable_spell_check', False)
        if self._enable_spell_check:
            from shared.spell_check_model import SpellCheckerModel
            self._spell_check_model = SpellCheckerModel(self._config,
                                                        self._config['spell_check_default_enchant_dictionary_language'],
                                                        [self._config['spell_check_word_frequency_lookup_table']])

        self._kenlm_bigram_model = kenlm.LanguageModel(self._config['bigram_model_path'])
        self._list_of_stopwords = set(stopwords.words('english'))

    def _load_previous_generated_paraphrases_from_file(self, file_path):
        """
        This function will load paraphrases and original sentence as a dict from previously generated file
        which has record of this data. The file should be in tsv format with two columns, where first column
        contains original sentence and second column contains list of it's paraphrases.

        :param file_path: path of previous generated paraphrases file (String)
        :return: sentence, list_of_paraphrases (dict)
        """

        if os.path.exists(file_path):
            previous_generated_paraphrases_dict = {}
            data = pd.read_csv(file_path, sep="\t", header=None)
            sentences = data.iloc[:, 0]
            list_of_paraphrases = data.iloc[:, 1]
            for sentence, paraphrases in zip(sentences, list_of_paraphrases):
                # ast.literal_eval is used to convert string representation of list in actual list object.
                # To know more check out this link https://stackoverflow.com/questions/10775894/converting-a-string-representation-of-a-list-into-an-actual-list-object
                previous_generated_paraphrases_dict[str(sentence)] = ast.literal_eval(paraphrases)
            return previous_generated_paraphrases_dict
        else:
            self._logging.info('ERROR: No previous file having generated paraphrases found.')
            return {}

    def _get_score_of_paraphrase_from_bigram_model(self, sentence, normalize=True):
        """
        This function calculates the score of given input sentence using language model.
        :param sentence: sentence (String)
        :param normalize: If enabled, language model score will be normalize based on num of words in a sentence(bool)
        :return: score from language model (Float)
        """
        bigram_score = 0
        if not sentence is None and not sentence == '':
            sentence = nltk_detokenize(sentence)
            sentence = self._preprocessing_for_bigram_model(sentence)
            # Changing the base to natural logarithm
            bigram_score = exp(self._kenlm_bigram_model.score(sentence) / 0.4342944819)

        bigram_score = max(bigram_score, 1.0e-60)

        if normalize:
            total_number_of_words = len(sentence.split())
            if not total_number_of_words == 0:
                return math.log(bigram_score) / total_number_of_words
            else:
                return math.log(bigram_score)

    def _preprocessing_for_bigram_model(self, sentence):
        """
        This function preprocess a sentence for the bigram language model by removing all the punctuation from the sentence.
        :param sentence:  str
        :return: sentence without any punctuation in str type
        """

        if sentence is None or sentence == '':
            return ''

        sentence = re.sub(r"[^\w\d'\s\d=\d+\d_\d*\d-]", '', sentence)

        mask_all_latex_equations_in_sentence = mask_all_latex(sentence).replace("<equation>", "equation").strip()

        return mask_all_latex_equations_in_sentence.lower()

    def _get_ppdb_dataset(self, ppdb_dataset_path, ppdb_dict_pkl_file="", save_dict=True, load_from_pkl_file=True):
        """
        This function will get the preprocessed ppdb dataset
        :param ppdb_dataset_path: path of preprocessed ppdb dataset (String)
        :param ppdb_dict_pkl_file: path of preprocessed ppdb dict file
        :param save_dict: If enabled, the dict created from ppdb datset will be saved in pkl file for later use. (Bool)
        :param load_from_pkl_file: If enabled, it will load dict from ppdb_dict.pkl file (Bool)
        :return: dictionary of ppdb dataset (Dict)
        """

        if load_from_pkl_file and os.path.exists(ppdb_dict_pkl_file):
            ppdb_dict = pickle.load(open(ppdb_dict_pkl_file, 'rb'))[0]

        else:

            if not os.path.exists(ppdb_dataset_path):
                raise ValueError("PPDB Dataset path doesn't exist")

            df = pandas.read_csv(ppdb_dataset_path, header=0, delimiter='\t')
            ppdb_dict = {}

            for index, row in df.iterrows():
                if row['sentence_1'] in ppdb_dict:
                    ppdb_dict[row['sentence_1']].append(
                        {'sentence_2': row['sentence_2'], 'ppdb_score1': row['ppdb1_score'],
                         'ppdb_score2': row['ppdb2_score']})
                else:
                    ppdb_dict[row['sentence_1']] = [
                        {'sentence_2': row['sentence_2'], 'ppdb_score1': row['ppdb1_score'],
                         'ppdb_score2': row['ppdb2_score']}]

                if row['sentence_2'] in ppdb_dict:
                    ppdb_dict[row['sentence_2']].append(
                        {'sentence_2': row['sentence_1'], 'ppdb_score1': row['ppdb1_score'],
                         'ppdb_score2': row['ppdb2_score']})
                else:
                    ppdb_dict[row['sentence_2']] = [
                        {'sentence_2': row['sentence_1'], 'ppdb_score1': row['ppdb1_score'],
                         'ppdb_score2': row['ppdb2_score']}]

            if save_dict:
                pickle.dump([ppdb_dict], open(ppdb_dict_pkl_file, 'wb'))

        return ppdb_dict

    def _extract_verb_phrase_from_sentence(self, sentence):
        """
        This function will extract verb phrases from the input sentences and will return the list of
        verb phrases.
        :param sentence: input sentence (String)
        :return: list of verb phrases in sentence (list of string)
        """

        verb_clause_pattern = r'<VERB>*<ADV>*<PART>*<VERB>+<PART>*'
        doc = textacy.Doc(sentence, lang='en_core_web_sm')
        lists = textacy.extract.pos_regex_matches(doc, verb_clause_pattern)
        list_of_verb_phrases = []
        for list in lists:
            if str(list.text).islower():
                list_of_verb_phrases.append(list.text)

        return list_of_verb_phrases

    def _extract_noun_phrase_from_sentence(self, sentence):
        """
        This function will extract verb phrases from the input sentences and will return the list of
        verb phrases.
        :param sentence: input sentence (String)
        :return: list of verb phrases in sentence (list of string)
        """

        noun_clause_pattern = textacy.constants.POS_REGEX_PATTERNS['en']['NP']
        doc = textacy.Doc(sentence, lang='en_core_web_sm')
        lists = textacy.extract.pos_regex_matches(doc, noun_clause_pattern)
        list_of_noun_phrases = []
        for list in lists:
            list_of_noun_phrases.append(list.text)

        if not len(list_of_noun_phrases) == 0:
            self._logging.info('Found noun phrases for sentence = %s , noun phrases list= %s', sentence,
                               list_of_noun_phrases)
        return list_of_noun_phrases

    def _get_phrase_replacement_from_ppdb_dataset(self, phrase):
        """
        This function returns the list of paraphrases of verb phrase from ppdb dataset.
        :param phrase: any phrase like noun phrase, verb phrase, adjective etc. (String)
        :return: list of paraphrases from ppdb dataset (list of String)
        """

        if str(phrase) in self._ppdb_dataset:
            possible_replacements = self._ppdb_dataset[str(phrase)]
            return possible_replacements
        else:
            if len(self._ppdb_dataset) == 0:
                print('WARNING: ppdb dict is empty')
            return []

    def generate_paraphrases_with_verb_replacement(self, sentence):
        """
        This function generate paraphrases by replacing verb phrases and ver in a sentence.
        :param sentence: text (string)
        :return: dictionary with paraphrases and score (dict)
        """

        preprocessed_sentence = sentence
        dict_of_paraphrases_with_scores = {}

        list_of_verb_phrases = self._extract_verb_phrase_from_sentence(preprocessed_sentence)
        for verb_phrase in list_of_verb_phrases:
            list_of_verb_phrase_split = str(verb_phrase).lower().split()
            if len([i for i in list_of_verb_phrase_split if i in self._list_of_stopwords]) == 0:
                possible_replacements = self._get_phrase_replacement_from_ppdb_dataset(verb_phrase)
                for replacement in possible_replacements:
                    # Please note that here we will replace all the occurrence of verb phrase with new verb phrase.
                    # First check whether new suggestion is actual word in english or not.
                    create_paraphrase = True
                    if self._enable_spell_check and not self._spell_check_model.is_word_in_dictionary(
                            str(replacement['sentence_2'])):
                        create_paraphrase = False

                    if create_paraphrase:
                        generated_paraphrase = str(
                            preprocessed_sentence.replace(" " + str(verb_phrase).strip() + " ",
                                                          " " + str(
                                                              replacement['sentence_2']) + " "))

                        # Check whether the new verb phrase is considered as a verb phrase in new generated paraphrase.
                        list_of_verb_phrases_in_generated_paraphrase = self._extract_verb_phrase_from_sentence(
                            generated_paraphrase)
                        if str(replacement['sentence_2']) in list_of_verb_phrases_in_generated_paraphrase:
                            bigram_score = self._get_score_of_paraphrase_from_bigram_model(generated_paraphrase)
                            dict_of_paraphrases_with_scores[
                                generated_paraphrase] = self._ppdbscore_weight * float(
                                replacement['ppdb_score2']) + self._lm_weight * float(bigram_score)

        if len(dict_of_paraphrases_with_scores) > 0:
            self._logging.info('verb replacement Paraphrases score dict %s', dict_of_paraphrases_with_scores)

        return dict_of_paraphrases_with_scores

    def generate_paraphrases_with_noun_replacement(self, sentence):
        """
        This function generate paraphrases by replacing noun phrases
        :param sentence: text (string)
        :return: dictionary with paraphrases and score (dict)
        """

        preprocessed_sentence = sentence
        dict_of_paraphrases_with_scores = {}

        list_of_noun_phrases = self._extract_noun_phrase_from_sentence(preprocessed_sentence)
        for noun_phrase in list_of_noun_phrases:
            possible_replacements = self._get_phrase_replacement_from_ppdb_dataset(noun_phrase)
            for replacement in possible_replacements:
                # Mote that here we will replace all the occurrence of noun phrase with new verb phrase.
                # First check whether new suggestion is actual word in english or not.
                create_paraphrase = True
                if self._enable_spell_check and not self._spell_check_model.is_word_in_dictionary(
                        str(replacement['sentence_2'])):
                    create_paraphrase = False

                if create_paraphrase:
                    generated_paraphrase = str(
                        preprocessed_sentence.replace(" " + str(noun_phrase).strip() + " ",
                                                      " " + str(
                                                          replacement['sentence_2']) + " "))

                    # Check whether the new verb phrase is considered as a verb phrase in new generated paraphrase.
                    list_of_noun_phrases_in_generated_paraphrase = self._extract_noun_phrase_from_sentence(
                        generated_paraphrase)
                    if str(replacement['sentence_2']) in list_of_noun_phrases_in_generated_paraphrase:
                        bigram_score = self._get_score_of_paraphrase_from_bigram_model(generated_paraphrase)
                        dict_of_paraphrases_with_scores[
                            generated_paraphrase] = self._ppdbscore_weight * float(
                            replacement['ppdb_score2']) + self._lm_weight * float(bigram_score)

        if not len(dict_of_paraphrases_with_scores) == 0:
            self._logging.info('noun replacement for sentence %s  dict = %s', sentence, dict_of_paraphrases_with_scores)

        return dict_of_paraphrases_with_scores

    def generate_paraphrases_with_n_gram_replacement(self, sentence):
        """
        This function generate paraphrases by replacing 1,2,3- gram in a sentence.
        :param sentence: text (string)
        :return: dictionary with paraphrases and score (dict)
        """

        dict_of_paraphrases_with_scores = {}
        preprocessed_sentence = sentence
        doc = textacy.Doc(preprocessed_sentence, lang='en_core_web_sm')
        list_of_ngram = []
        list_of_unigram = list(textacy.extract.ngrams(doc, 1, filter_stops=True, filter_punct=True, filter_nums=False))
        list_of_bigram = list(textacy.extract.ngrams(doc, 2, filter_stops=True, filter_punct=True, filter_nums=False))
        list_of_trigram = list(textacy.extract.ngrams(doc, 3, filter_stops=True, filter_punct=True, filter_nums=False))

        list_of_ngram.extend(list_of_unigram)
        list_of_ngram.extend(list_of_bigram)
        list_of_ngram.extend(list_of_trigram)

        for n_gram in list_of_ngram:
            possible_replacements = self._get_phrase_replacement_from_ppdb_dataset(n_gram)
            for replacement in possible_replacements:
                create_paraphrase = True
                if self._enable_spell_check and not self._spell_check_model.is_word_in_dictionary(
                        str(replacement['sentence_2'])):
                    create_paraphrase = False

                if create_paraphrase:
                    generated_paraphrase = str(
                        preprocessed_sentence.replace(" " + str(n_gram).strip() + " ",
                                                      " " + str(
                                                          replacement['sentence_2']) + " "))
                    bigram_score = self._get_score_of_paraphrase_from_bigram_model(generated_paraphrase)
                    dict_of_paraphrases_with_scores[
                        generated_paraphrase] = self._ppdbscore_weight * float(
                        replacement['ppdb_score2']) + self._lm_weight * float(bigram_score)

        if not len(dict_of_paraphrases_with_scores) == 0:
            self._logging.info('n-gram replacement found for sentence %s  dict = %s', sentence,
                               dict_of_paraphrases_with_scores)

        return dict_of_paraphrases_with_scores

    def generate_paraphrases(self, sentence, max_number_of_paraphrases=10, verb_phrase_replacement=True,
                             noun_phrase_replacement=False, n_gram_replacement=False):
        """
        This function will generate paraphrase of given input sentence.

        :param sentence: sentence (String)
        :param max_number_of_paraphrases: maximum number of paraphrases of sentence required (Integer)
        :param verb_phrase_replacement: if enabled verb phrase replacement will be searched in PPDB dict for generating
        paraphrases (bool)
        :param noun_phrase_replacement: if enabled noun phrase replacement will be searched in PPDB dict for generating
        paraphrases (bool)
        :param n_gram_replacement: if enabled (1,2,3)-gram replacement will be searched in PPDB dict for generating
        paraphrases (bool)
        :return: if model is able to generate paraphrase for input sentence, then
        list of top n paraphrases.(List of String)

        Note: confidence is the weighted sum of score we got from bigram language model and ppdb score for replacement.
        """

        if max_number_of_paraphrases == 0:
            return []

        if self._use_previous_generated_paraphrases:
            if str(sentence) in self._previous_generated_paraphrases_dict:
                list_of_paraphrases = self._previous_generated_paraphrases_dict[sentence]
                if len(list_of_paraphrases) > max_number_of_paraphrases:
                    return list_of_paraphrases[0:max_number_of_paraphrases]
                else:
                    return list_of_paraphrases

        # First preprocess input sentence
        # preprocessed_sentence = self._preprocess_sentence(sentence)
        preprocessed_sentence = sentence
        if not preprocessed_sentence == "":
            dict_of_paraphrases_with_scores = {}

            # Firstly do it for verb phrase
            if verb_phrase_replacement:
                paraphrases_dict_with_verb_replacements = self.generate_paraphrases_with_verb_replacement(
                    preprocessed_sentence)
                dict_of_paraphrases_with_scores.update(paraphrases_dict_with_verb_replacements)

            if noun_phrase_replacement:
                paraphrases_dict_with_noun_replacements = self.generate_paraphrases_with_noun_replacement(
                    preprocessed_sentence)
                dict_of_paraphrases_with_scores.update(paraphrases_dict_with_noun_replacements)

            if n_gram_replacement:
                paraphrases_dict_with_n_gram_replacements = self.generate_paraphrases_with_n_gram_replacement(
                    preprocessed_sentence)
                dict_of_paraphrases_with_scores.update(paraphrases_dict_with_n_gram_replacements)

            dict_of_paraphrases_with_scores = OrderedDict(
                sorted(dict_of_paraphrases_with_scores.items(), key=lambda x: float(x[1]), reverse=True))

            all_list_of_paraphrases = list(dict_of_paraphrases_with_scores.keys())
            if len(all_list_of_paraphrases) > max_number_of_paraphrases:
                return all_list_of_paraphrases[0:max_number_of_paraphrases]
            else:
                return all_list_of_paraphrases
        else:
            return []
