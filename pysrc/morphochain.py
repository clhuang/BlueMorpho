from collections import namedtuple, Counter
import sys
import math
import string
import itertools
from enum import Enum
import numpy as np
import scipy.spatial.distance
import sklearn.linear_model
from sklearn.feature_extraction import DictVectorizer

from pysrc import accuracy

ParentTransformation = namedtuple('ParentTransformation',
                                  ['parentword', 'transformtype', 'olangconfidence'])

TOPNEIGHBOURS = 1


class ParentType():
    STOP = 'STOP'
    PREFIX = 'PREFIX'
    SUFFIX = 'SUFFIX'
    MODIFY = 'MODIFY'
    DELETE = 'DELETE'
    REPEAT = 'REPEAT'
    OLANG = 'OLANG'


class MorphoChain(object):
    def __init__(self, wordvectors, vocab, affixes, affixNeighbours, dictionary=None,
                 alphabet=string.ascii_lowercase, dictvectorizer=None,
                 weightvector=None, segmentations=None):
        self.wordvectors = wordvectors
        self.vocab = vocab
        self.dictionary = dictionary
        self.prefixes, self.suffixes = affixes
        self.alphabet = alphabet
        self.dictvectorizer = dictvectorizer
        self.prefixNeighbours, self.suffixNeighbours = affixNeighbours
        self.weightvector = weightvector
        self.segmentations = segmentations

    def setWeightVector(self, weightvector):
        self.weightvector = weightvector

    def getParentsFeatures(self, w):
        """
        Return dict from possible parents to feature dict
        """
        parentsAndFeatures = {}
        parentt = None
        for z in self.genCandidates(w):
            if z.transformtype == ParentType.STOP:
                parentt = z
                continue
            parentsAndFeatures[z] = self.getFeatures(w, z)
        parentsAndFeatures[parentt] = self.getFeatures(
            w, z, max([d['cos'] for d in parentsAndFeatures.values()] or [0]))
        return parentsAndFeatures

    def getFeatures(self, w, z, maxCosSimilarity=None):
        """
        Get features for word-parent pair
        """
        MAX_PREF_LEN = 5
        MAX_SUFF_LEN = 5
        d = {'BIAS': 1}
        if z.transformtype == ParentType.STOP:
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
        if z.transformtype == ParentType.OLANG:
            pass  # features for OLANG will be in duolingo, this is just to prevent
                  # base features from messing up
        elif z.transformtype == ParentType.PREFIX:
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
            if affix in self.prefixNeighbours:
<<<<<<< Updated upstream
                for n, score in self.prefixNeighbours[affix][:TOPNEIGHBOURS]:
                    if n + parent in self.vocab:
                        d['neighbours_COR_P'] = affix
                for n, score in self.prefixNeighbours[affix][:TOPNEIGHBOURS]:
=======
                for n, score in self.prefixNeighbours[affix][:ParentType.TOPNEIGHBOURS]:
>>>>>>> Stashed changes
                    if n + parent in self.vocab:
                        d['neighbours_COR_P'] = affix
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
            if affix in self.suffixNeighbours:
                for n, score in self.suffixNeighbours[affix][:TOPNEIGHBOURS]:
                    if w[:-len(affix)] + n in self.vocab:
                        d['neighbours_COR_S'] = affix
         # parent is not in word list
        if z.parentword not in self.vocab:
            d['out_of_vocab'] = 1
        # if parent in word list - log frequency
        else:
            d['parent_in_word_list'] = math.log10(self.vocab[z.parentword])
        # presence in English dictionary
        #d['parent_in_dict'] = z.parentword in self.dictionary

        #TODO USE dp -- extend C(w) using existing affixes and word2vec
        return d

    #TODO add prunning heuristic
    def genCandidates(self, word):
        candidates = []
        for x in range(len(word) // 2, len(word)):
            parent = word[:x]
            if parent:
                candidates.append(ParentTransformation(parent, ParentType.SUFFIX, None))
            #planning - plan - (n)ing
            if x > 0 and word[x] == word[x - 1]:
                candidates.append(ParentTransformation(parent, ParentType.REPEAT, None))
            if parent and parent[-1] in self.alphabet:
                for l in self.alphabet:
                    if l != parent[-1]:
                        # TODO do we want word count threshhold?
                        newParent = parent[:-1] + l
                        SIM_THRESH = 0.2
                        if newParent in self.vocab and self.similarity(word, newParent) > SIM_THRESH:
                            candidates.append(ParentTransformation(newParent, ParentType.MODIFY, None))
            # libraries - librar(y) -ies ?? (Delete)
            if len(parent) < len(word) - 1 and word[x:] in self.suffixes:
                for l in self.alphabet:
                    newParent = parent + l
                    SIM_THRESH = 0
                    if newParent in self.vocab and self.similarity(word, newParent) > SIM_THRESH:
                        candidates.append(ParentTransformation(newParent, ParentType.DELETE, None))
        for x in range(1, (len(word) + 2) // 2):
            parent = word[x:]
            candidates.append(ParentTransformation(parent, ParentType.PREFIX, None))
        candidates.append(ParentTransformation(word, ParentType.STOP, None))
        # Stopping condition handled in getParentsAndFeatures
        # candidates.append(ParentTransformation(word, ParentType.STOP))
        return candidates

    def genNeighbors(self, w, k=5):
        k = min(k, (len(w)) // 2)
        ne = set([w])

        def swap(word, i):
            return word[:i] + word[i + 1] + word[i] + word[i + 2:]
        for i in range(k):
            ne.add(swap(w, i))
        for i in range(len(w) - k - 1, len(w) - 1):
            ne.add(swap(w, i))
        for i in range(min(k, (len(w) // 2) - 1)):
            # for j in range(max(k + 2, len(w) - k - 1), len(w) - 1):
            ne.add(swap(swap(w, i), len(w) - i - 2))
        return ne

    def genTrainingData(self):
        '''
        Returns a tuple containing:
            Giant feature training matrix
            Words/neighbors appearing in order, and how many z's they have
        '''
        curid = [0] ## ghetto hack because python 2 doesn't have nonlocal
        dicts = []
        widsneighbors = []
        nzs = []

        def addword(word):
            parentsFeatures = self.getParentsFeatures(word)
            curid[0] += 1
            dicts.extend(parentsFeatures.values())
            nzs.append(len(parentsFeatures))

        for word in self.vocab:
            widx = curid[0]
            addword(word)
            neighbors = self.genNeighbors(word)
            neighboridxs = []
            for neighbor in neighbors:
                neighboridxs.append(curid[0])
                addword(neighbor)
            widsneighbors.append((widx, neighboridxs))

        if self.dictvectorizer is None:
            self.dictvectorizer = DictVectorizer()
            X = self.dictvectorizer.fit_transform(dicts)
        else:
            X = self.dictvectorizer.transform(dicts)
        return X, nzs, widsneighbors

    def genGoldsegTrainingData(self):
        dicts = []
        nzs = []
        cxs = []
        chainsets = self.genGoldChains().values()
        chains = itertools.chain(*chainsets)
        wordpairs = itertools.chain(*chains)
        for word, parent in wordpairs:
            parentsFeatures = self.getParentsFeatures(word)

            parents = list(parentsFeatures.keys())
            features = list(parentsFeatures.values())
            try:
                idx = parents.index(parent)
            except ValueError:
                continue
            nzs.append(len(parents))
            cxs.append(idx + len(dicts))
            dicts.extend(features)

        X = self.dictvectorizer.transform(dicts)
        return X, nzs, cxs

    def scoreFeatures(self, featureDict):
        fv = self.dictvectorizer.transform(featureDict)
        return np.asscalar(fv.dot(self.weightvector))

    # predicts top k candidates given a word, and scores
    def predict(self, word, k=None):
        parentsFeatures = self.getParentsFeatures(word)
        parentsScores = Counter({parent: self.scoreFeatures(features)
                         for parent, features in parentsFeatures.items()})
        return parentsScores.most_common(k)

    def predict_logprobs(self, word, k=None):
        parentsFeatures = self.getParentsFeatures(word)
        parentsScores = Counter({parent: self.scoreFeatures(features)
                         for parent, features in parentsFeatures.items()})
        normc = np.log(sum(np.exp(i) for i in parentsScores.values()))
        for parent in parentsScores:
            parentsScores[parent] -= normc
        return parentsScores.most_common(k)

    def genSeg(self, word):
        if len(word) < 3:
            return word
        SEG_SEP = "/"
        if "-" in word:
            segmentations = []
            for part in word.split('-'):
                segmentations.append(self.genSeg(part))
            return (SEG_SEP + '-' + SEG_SEP).join(segmentations)

        if "\'" in word:
            parts = word.split('\'')
            segmentation = self.genSeg(parts[0])
            for part in parts[1:]:
                segmentation += SEG_SEP + '\'' + part
            return segmentation

        candidate = self.predict(word)[0][0]
        if candidate[1] == ParentType.STOP:
            return word
        parent = candidate[0]
        p_len = len(parent)
        if candidate[1] == ParentType.SUFFIX:
            suffix = word[p_len:]
            #TODO if in suffix list - change dist
            return self.genSeg(parent) + SEG_SEP + suffix
        elif candidate[1] == ParentType.REPEAT:
            return self.genSeg(parent) + word[p_len] + SEG_SEP + word[p_len + 1:]
        elif candidate[1] == ParentType.MODIFY:
            return self.genSeg(parent)[:-1] + word[p_len - 1] + SEG_SEP + word[p_len:]
        elif candidate[1] == ParentType.DELETE:
            parent_seg = self.genSeg(parent)
            if parent_seg[-2] == SEG_SEP:
                return parent_seg[:-1] + word[p_len-1:]
            return parent_seg[:-1] + SEG_SEP + word[p_len-1:]
        elif candidate[1] == ParentType.PREFIX:
            return word[:-p_len] + SEG_SEP + self.genSeg(parent)
        else:
            lencommonprefix = 0
            for a, b in zip(word, parent):
                if a != b:
                    break
                lencommonprefix += 1
            return word[lencommonprefix:] + self.genSeg(parent)[:lencommonprefix]

    def similarity(self, w1, w2):
        if w1 not in self.wordvectors or w2 not in self.wordvectors:
            return 0.0
        return self.wordvectors.similarity(w1, w2)

    def computeAccuracy(self, segmentations=None, verbose=False):
        if segmentations is None:
            segmentations = self.segmentations
        wordlist = segmentations.keys()
        predictions = zip(wordlist, map(self.genSeg, wordlist))
        return accuracy.score(segmentations, predictions, verbose)

    def computeParentValidity(self, segmentations=None, k=1):
        """
        segmentations: as computed by readCorpus.
        k: look at k highest parents
        """
        if segmentations is None:
            segmentations = self.segmentations
        SEG_SEP = '/' # change if SEG_SEP changes above
        num = 0
        correct_segs = 0
        correct_parent = 0
        optimal_parent = 0
        for word in segmentations:
            gold_segs, gold_tags = segmentations[word]
            num += 1
            seg = self.genSeg(word).split(SEG_SEP)
            if seg in gold_segs:
                correct_segs += 1
            parents = self.predict(word, k)
            found = False
            found_optimal = False
            for parent in parents:
                parent = parent[0].parentword
                if parent == word:
                    if [word] in gold_segs:
                        found = True
                        found_optimal = True
                else:
                    for g_seg in gold_segs:
                        for a in range(len(g_seg)):
                            for b in range(a + 1, len(g_seg) + 1):
                                w = ''.join(g_seg[a:b])
                                if parent == w:
                                    found  = True
                        if parent == ''.join(g_seg[1:]) or parent == ''.join(g_seg[:-1]):
                            found_optimal = True
            if found:
                correct_parent += 1
            if found_optimal:
                optimal_parent += 1
        print('%s correct segmentations of out %s' % (correct_segs, num))
        print('%s valid parents of out %s' % (correct_parent, num))
        print('%s optimal parents of out %s' % (optimal_parent, num))
        return float(correct_segs) / num, float(correct_parent) / num, float(optimal_parent) / num

    def genGoldChains(self, segmentations=None):
        if segmentations is None:
            segmentations = self.segmentations
        def parentScore(parent, word): # can we train on these weights :)
            FREQ_WEIGHT = 1
            SIM_WEIGHT = 1
            SIM_BASE = 12
            LEN_WEIGHT = 1
            LEN_BASE = 5
            LENGTH_WEIGHT = 30
            LENGTH_POWER = 2
            score = 0
            if parent in self.vocab:
                score += FREQ_WEIGHT * math.log(self.vocab[parent])
            score += SIM_WEIGHT * SIM_BASE ** max(0, self.similarity(parent, word))
            score += LEN_WEIGHT * LEN_BASE ** (float(len(parent)) / len(word))
            score -= LENGTH_WEIGHT / max(1, len(parent)) ** LENGTH_POWER
            return score
        def chain(segs, tags):
            def getAffix(s, t):
                return s if t[0] == '+' else t[:t.find('_')]
            def suffix_type(s, t):
                if t[0] == '+':
                    return ParentType.SUFFIX
                t = t[:t.find('_')]
                if s == t:
                    return ParentType.SUFFIX
                elif len(s) == len(t):
                    return ParentType.MODIFY
                elif s[:-1] == t:
                    return ParentType.REPEAT
                else:
                    return ParentType.DELETE
            if len(segs) <= 1:
                root = getAffix(segs[0], tags[0])
                return [(root, ParentTransformation(root, ParentType.STOP, None))]
            word = ''.join(segs[:-1]) + getAffix(segs[-1], tags[-1])
            word = word.replace('~', '') # null segments
            parent_suf = ''.join(segs[:-2]) + getAffix(segs[-2], tags[-2])
            parent_suf = parent_suf.replace('~', '')
            pre_parent = ''.join(segs[1:-1]) + getAffix(segs[-1], tags[-1])
            pre_parent = pre_parent.replace('~', '')
            type_suf = suffix_type(segs[-2], tags[-2])
            pre_pair = [(word, ParentTransformation(pre_parent, ParentType.PREFIX, None))]
            pair_suf = [(word, ParentTransformation(parent_suf, type_suf, None))]
            if tags[-1][0] == '+': # inflectional
                return pair_suf + chain(segs[:-1], tags[:-1])
            score_suf = parentScore(parent_suf, word)
            pre_score = parentScore(pre_parent, word)
            if pre_score > score_suf:
                return pre_pair + chain(segs[1:], tags[1:])
            else:
                return pair_suf + chain(segs[:-1], tags[:-1])
        d = {}
        for word, (g_segs, g_tags) in segmentations.items():
            d[word] = []
            for segs, tags in zip(g_segs, g_tags):
                d[word].append(chain(segs, tags))
        return d
