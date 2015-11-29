from collections import namedtuple
import math
import string
from enum import Enum
import numpy as np
import sklearn.linear_model
import sklearn.feature_extraction

from globals import *

ParentTransformation = namedtuple('ParentTransformation',
                                  ['parentword', 'transformtype'])

class ParentType(Enum):
    STOP = 'STOP'
    PREFIX = 'PREFIX'
    SUFFIX = 'SUFFIX'
    MODIFY = 'MODIFY'
    DELETE = 'DELETE'
    REPEAT = 'REPEAT'

class MorphoChain(object):
    def __init__(self, wordvectors, vocab, alphabet=string.ascii_lowercase):
        self.wordvectors = wordvectors
        self.vocab = vocab
        self.alphabet = alphabet

    def getParentsFeatures(self, w):
        parentsAndFeatures = {}
        for z in self.genCandidates(w):
            parentsAndFeatures[z] = self.getFeatures(w, z)
        z = ParentTransformation(w, ParentType.STOP)
        parentsAndFeatures[z] = getFeatures(
            w, z, max(d['cos'] for d in parentsAndFeatures.items()))
        return parentsAndFeatures

    def getFeatures(self, w, z, maxCosSimilarity=None):
        """
        Get features for word-parent pair
        """
        MAX_PREF_LEN = 5
        MAX_SUFF_LEN = 5
        d = {'BIAS': 1}
        if len(w) == len(z.parentword):
            if len(w) > 2:
                d['STP_E_'] = w[:-2]
                d['STP_B_'] = w[:2]
            d['STP_COS_' + str(int(10 * maxCosSimilarity))] = 1
            d['STP_LEN_' + str(len(w))] = 1
            return d
        d['parentword'] = z.parentword  # Too large?
        d['transformtype'] = z.transformtype
        # cosine similarity between word and parent
        d['cos'] = self.similarity(w, z.parentword)

        lenparent = len(z.parentword)
        # affix
        if z.transformtype == ParentType.PREFIX:
            affix = w[:-lenparent]
            # list of prefixes  #TODO: Maxlength of suffix is a param??
            if len(affix) > MAX_PREF_LEN or affix not in PREFIXES:
                affix = "UNK"
            d['affix+type'] = "PREFIX_" + affix
            for prefix in PREFIXES:
                # FIX: what if z.parentword does not have a vector?
                difference = self.wordvectors[w] - self.wordvectors[z.parentword]
                cos_sim = self.similarity(difference, self.wordvectors[prefix])
                # FIX: this is currently storing only the last prefix into d['diff']
                d['diff'] = prefix + "_" + cos_sim
        else:  # some sort of suffix
            if z.transformtype == ParentType.SUFFIX:
                affix = w[lenparent:]
            elif z.transformtype == ParentType.REPEAT:
                affix = w[lenparent + 1:]
            elif z.transformtype == ParentType.DELETE:
                affix = w[lenparent - 1:]
            elif z.transformtype == ParentType.MODIFY:
                affix = w[len(z.parentword):]
            # list of suffixes  #TODO: Maxlength of suffix is a param??
            if len(affix) > MAX_SUFF_LEN or affix not in SUFFIXES:
                affix = "UNK"
            d['affix+type'] = "SUFFIX_" + affix
            for suffix in SUFFIXES:
                difference = self.wordvectors[w] - self.wordvectors[z.parentword]
                cos_sim = self.similarity(difference, self.wordvectors[suffix])
                d['diff'] = suffix + "_" + cos_sim

        # affix correlation TODO
        # parent is not in word list
        if z.parentword not in self.vocab:
            d['out_of_vocab'] = 1
        # if parent in word list - log frequency
        else:
            d['parent_in_word_list'] = math.log10(self.vocab[z.parentword])
        # presence in English dictionary
        d['parent_in_dict'] = 0

        #TODO USE dp -- extend C(w) using existing affixes and word2vec

    #TODO add prunning heuristic
    def genCandidates(self, word):
        candidates = []
        for x in range(len(word) // 2, len(word)):
            parent = word[:x]
            if parent:
                candidates.append(ParentTransformation(parent, ParentType.SUFFIX))
            #planning - plan - (n)ing
            if x > 0 and word[x] == word[x - 1]:
                candidates.append(ParentTransformation(parent, ParentType.REPEAT))
            if parent and parent[-1] in self.alphabet:
                for l in self.alphabet:
                    if l != parent[-1]:
                        # Ignored lines 526-27 (checks if new parent is a word???)
                        # and checks similiary btwn words and parents
                        candidates.append(ParentTransformation(parent[:-1] + l, ParentType.MODIFY))
            # libraries - librar(y) -ies ?? (Delete)
            if len(parent) < len(word) - 1 and word[x:] in SUFFIXES:
                for l in self.alphabet:
                    # TODO check if parent+l is a word
                    candidates.append(ParentTransformation(parent + l, ParentType.DELETE))
        for x in range(1, len(word) // 2):
            parent = word[x:]
            candidates.append(ParentTransformation(parent, ParentType.PREFIX))
        # Stopping condition handled in getParentsAndFeatures
        # candidates.append(ParentTransformation(word, ParentType.STOP))
        return candidates

    def score(self, word, parent, paretn_type):
        return 0

    # predicts top k candidates given a word
    def predict(self, word, k=1):
        candidates = self.genCandidates(word)
        scores = [(p, p_type, self.score(word, p, p_type)) for p, p_type in candidates]
        top_k = sorted(scores, key=lambda x:x[2])[:k]
        return top_k
        #Add some stuff for multinomial later??
        #STOP FACTOR TODO ???

    def genSeg(self, word):
        if "-" in word:
            segmentation = ""
            for part in word.split('-'):
                segmentation += "-" + self.genSeg(part)
            return segmentation[1:]

        if "\'" in word:
            parts = word.split('\'')
            # I combined 2 if statements, need to check if correct (need to get
            # list suffixes
            if len(parts) == 2 and parts[1] in SUFFIXES:
                #TODO update suffix freq dist
                pass
            segmentation = self.genSeg(parts[0])
            for x in len(parts):
                segmentation += "-" + part
                return segmentation[1:]
        candidate = predict(word)
        if candidate[1] == "STOP":
            return word
        parent = candidate[0]
        p_len = len(parent)
        if candidate[1] == ParentType.SUFFIX:
            suffix = word[p_len:]
            #TODO if in suffix list - change dist
            return self.genSeg(parent) + "-" + suffix
        elif candidate[1] == "REPEAT":
            return self.genSeg(parent) + word[p_len] + "-" + word[p_len + 1:]
        elif candidate[1] == "MODIFY":
            return self.genSeg(parent)[:-1] + word[p_len - 1] + "-" + word[p_len:]
        elif candidate[1] == "DELETE":
            parent_seg = self.genSeg(parent)
            if parent_seg[-2] == '-':
                return parent_seg[:-1] + word[p_len-1:]
            return parent_seg[:-1] + "-" + word[p_len-1:]




    def genAffixesList(self):
        pass

    def genAffixCorrelation(self):
        pass

    def similarity(self, w1, w2):
        if w1 not in self.wordvectors or w2 not in self.wordvectors:
            # is this what we want to return? or just 0?
            return None
        return self.wordvectors.similarity(w1, w2)
