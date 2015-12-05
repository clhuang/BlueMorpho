from collections import namedtuple
import math
import string
from enum import Enum
import numpy as np
import scipy.spatial.distance
import sklearn.linear_model
from sklearn.feature_extraction import DictVectorizer

from params import *

ParentTransformation = namedtuple('ParentTransformation',
                                  ['parentword', 'transformtype'])


class ParentType():
    STOP = 'STOP'
    PREFIX = 'PREFIX'
    SUFFIX = 'SUFFIX'
    MODIFY = 'MODIFY'
    DELETE = 'DELETE'
    REPEAT = 'REPEAT'


class MorphoChain(object):
    def __init__(self, wordvectors, vocab, affixes, dictionary=None,
                 alphabet=string.ascii_lowercase, dictvectorizer=None,
                 weightvector=None):
        self.wordvectors = wordvectors
        self.vocab = vocab
        self.dictionary = dictionary
        self.prefixes, self.suffixes = affixes
        self.alphabet = alphabet
        self.dictvectorizer = dictvectorizer or DictVectorizer()

    def getParentsFeatures(self, w):
        """
        Return dict from possible parents to feature dict
        """
        parentsAndFeatures = {}
        for z in self.genCandidates(w):
            parentsAndFeatures[z] = self.getFeatures(w, z)
        z = ParentTransformation(w, ParentType.STOP)
        parentsAndFeatures[z] = self.getFeatures(
            w, z, max((d['cos'] for d in parentsAndFeatures.values()), default=0))
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
                d['STP_E_'] = w[-2:]
                d['STP_B_'] = w[:2]
            d['STP_COS_' + str(int(10 * maxCosSimilarity))] = 1
            d['STP_LEN_' + str(len(w))] = 1
            return d
        # d['parentword'] = z.parentword  # Too large?
        d['transformtype'] = z.transformtype
        # cosine similarity between word and parent
        d['cos'] = self.similarity(w, z.parentword)
        parent = z.parentword
        lenparent = len(z.parentword)
        # affix
        if z.transformtype == ParentType.PREFIX:
            affix = w[:-lenparent]
            # list of prefixes  #TODO: Maxlength of suffix is a param??
            if len(affix) > MAX_PREF_LEN or affix not in self.prefixes:
                affix = "UNK"
            d['affix+type_PREFIX'] = affix
            # for prefix in self.prefixes:
                # if w in self.wordvectors and parent in self.wordvectors:
                    # # FIX: what if z.parentword does not have a vector?
                    # difference = self.wordvectors[w] - self.wordvectors[parent]
                    # cos_sim = scipy.spatial.distance.cosine(difference, self.prefixvectors[prefix])
                    # # FIX: this is currently storing only the last prefix into d['diff']
                    # d['diffpre_' + prefix] = cos_sim
            # if affix in PREFIXNEIGHBOURS:
                # for n in PREFIXNEIGHBOURS[affix]:
                    # if parent + n in vocab:
                        # d['neighbours_COR_S'] = affix
        else:  # some sort of suffix
            if z.transformtype == ParentType.SUFFIX:
                affix = w[lenparent:]
            elif z.transformtype == ParentType.REPEAT:
                affix = w[lenparent + 1:]
                d['char_REPEAT'] = parent[-1]
            elif z.transformtype == ParentType.DELETE:
                affix = w[lenparent - 1:]
                d['char_DELETE'] = parent[-1]
            elif z.transformtype == ParentType.MODIFY:
                affix = w[lenparent:]
                d['char_MODIFY'] = parent[-1] + "_" + w[lenparent - 1]
            # list of suffixes
            if len(affix) > MAX_SUFF_LEN or affix not in self.suffixes:
                affix = "UNK"
            d['affix+type_SUFFIX'] = affix
            # for suffix in self.suffixes:
                # if w in self.wordvectors and parent in self.wordvectors:
                    # difference = self.wordvectors[w] - self.wordvectors[parent]
                    # cos_sim = scipy.spatial.distance.cosine(difference, self.suffixvectors[suffix])
                    # d['diffsuff_' + suffix ] = cos_sim
            # # affix correlation TODO check for each case
            # if affix in SUFFIXNEIGHBOURS:
                # for n in SUFFIXNEIGHBOURS[affix]:
                    # if w[:lenparent - len(affix)] + n in vocab:
                        # d['neighbours_COR_S'] = affix
         # parent is not in word list
        if z.parentword not in self.vocab:
            d['out_of_vocab'] = 1
        # if parent in word list - log frequency
        else:
            d['parent_in_word_list'] = math.log10(self.vocab[z.parentword])
        # presence in English dictionary
        # d['parent_in_dict'] = z.parentword in self.dictionary

        #TODO USE dp -- extend C(w) using existing affixes and word2vec
        return d

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
                        # TODO do we want word count threshhold?
                        newParent = parent[:-1] + l
                        SIM_THRESH = 0.2
                        if newParent in self.vocab and self.similarity(word, newParent) > SIM_THRESH:
                            candidates.append(ParentTransformation(newParent, ParentType.MODIFY))
            # libraries - librar(y) -ies ?? (Delete)
            if len(parent) < len(word) - 1 and word[x:] in self.suffixes:
                for l in self.alphabet:
                    newParent = parent + l
                    SIM_THRESH = 0
                    if newParent in self.vocab and self.similarity(word, newParent) > SIM_THRESH:
                        candidates.append(ParentTransformation(newParent, ParentType.DELETE))
        for x in range(1, (len(word) + 2) // 2):
            parent = word[x:]
            candidates.append(ParentTransformation(parent, ParentType.PREFIX))
        # Stopping condition handled in getParentsAndFeatures
        # candidates.append(ParentTransformation(word, ParentType.STOP))
        return candidates

    def genNeighbors(self, w, k=5):
        k = min(k, (len(w)) // 2)
        ne = set([w])
        def swap(word, i):
            return word[:i] + word[i+1] + word[i] + word[i+2:]
        for i in range(k):
            ne.add(swap(w, i))
        for i in range(len(w) - k - 1, len(w) - 1):
            ne.add(swap(w, i))
        for i in range(min(k, (len(w) // 2) - 1)):
            # for j in range(max(k + 2, len(w) - k - 1), len(w) - 1):
            ne.add(swap(swap(w, i), len(w) - i - 2))
        ne.discard(w)
        return ne

    def genTrainingData(self):
        '''
        Returns a tuple containing:
            Giant feature training matrix
            Words/neighbors appearing in order, and how many z's they have
        '''
        curid = 0
        dicts = []
        idxs = {}
        widsneighbors = []
        nzs = []

        def addword(word):
            nonlocal curid
            parentsFeatures = self.getParentsFeatures(word)
            idxs[word] = curid
            curid += 1
            dicts.extend(parentsFeatures.values())
            nzs.append(len(parentsFeatures))

        for word in self.vocab:
            if word not in idxs:
                addword(word)
            neighbors = self.genNeighbors(word)
            for neighbor in neighbors:
                if neighbor not in idxs:
                    addword(neighbor)
            widsneighbors.append((idxs[word], [idxs[neighbor] for neighbor in neighbors]))

        X = self.dictvectorizer.fit_transform(dicts)
        return X, nzs, widsneighbors

    def scoreFeatures(self, featureDict):
        fv = dictvectorizer.transform(featureDict)
        return fv.dot(self.weightvector)

    # predicts top k candidates given a word
    def predict(self, word, k=1):
        parentsFeatures = self.getParentsFeatures(word)
        parentsScores = {parent: self.scoreFeatures(features)
                         for parent, features in parentsFeatures.items()}
        return parentsScores

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
            if len(parts) == 2 and parts[1] in self.suffixes:
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

    def similarity(self, w1, w2):
        if w1 not in self.wordvectors or w2 not in self.wordvectors:
            # is this what we want to return? or just 0?
            return 0.0
        return self.wordvectors.similarity(w1, w2)
