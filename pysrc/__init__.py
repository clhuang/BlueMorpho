from collections import namedtuple
import math
import sklearn.linear_model
imprt sklearn.feature_extraction

alphabet = 'abcdefghijklmnopqrstuvwxyz'
VECTOR_SIZE = 200

ParentTransformation = namedtuple('ParentTransformation',
                                  'parentword', 'transformtype')


class MorphoFeatureGen(object):
    def __init__(self, wordvectors):
        self.wordvectors = wordvectors

    def getFeatures(self, w, z):
        """
        Get features for word-parent pair
        """
        MAX_PREF_LEN = 5
        MAX_SUFF_LEN = 5
        d = {}
    #stop condition TODO
        if len(w) == len(z.parentword):
            return 0
        d['parentword'] = z.parentword  #Too large?
        d['transformtype'] = z.transformtype
        # cosine similarity between word and parent
        d['cos'] = self.wordvectors.similarity(w, z.parentword)
        # affix
        if z.transformtype == PREFIX:
            affix = w[:len(w) - len(z.parentword)]
            if len(affix) > MAX_PREF_LEN or affix not in list_prefix: #list of prefixes  #TODO: Maxlength of suffix is a param??
                affix = "UNK"
            d['affix+type'] = "PREFIX_" + affix
            for prefix in list_prefix:
                difference = self.wordvectors[w] - self.wordvectors[z.parentword]
                cos_sim = self.wordvectors.similarity(difference, self.wordvectors[prefix])
                d['diff'] = prefix + "_" + cos_sim
        else:
            if z.transformtype == SUFFIX:
                affix = w[len(z.parentword):]
            elif z.transformtype == REPEAT:
                affix = w[len(z.parentword) + 1:]
                #only works for suffix
            elif z.transformtype == DELETE:
                affix = w[len(z.parentword) - 1:]
            elif z.transformtype == MODIFY:
                affix = w[len(z.parentword):]
            if len(affix) > MAX_SUFF_LEN or affix not in list_suffix: #list of suffixes  #TODO: Maxlength of suffix is a param??
                affix = "UNK"
            d['affix+type'] = "SUFFIX_" + affix
            for suffix in list_suffix:
                difference = self.wordvectors[w] - self.wordvectors[z.parentword]
                cos_sim = self.wordvectors.similarity(difference, self.wordvectors[suffix])
                d['diff'] = suffix + "_" + cos_sim

        # affix correlation TODO
        # parent is not in word list
        if z.parentword not in vocab:
            d['out_of_vocab'] = 1
        # if parent in word list - log frequency
        else:
            d['parent_in_word_list'] = math.log10(vocab[z.parentword])
        # presence in English dictionary
        d['parent_in_dict'] = 0

        #TODO USE dp -- extend C(w) using existing affixes and word2vec
