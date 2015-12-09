from pysrc import *
from pysrc.morphochain import MorphoChain, ParentTransformation, ParentType
import string
import editdistance
import pickle


TWOLANG_LANG = 'tur'
TWOLANG_VEC = 'full'
TWOLANG_VOCAB = 'med'
TWOLANG_SUPERVISED = 0.5

class TwoLangMorphoChain(MorphoChain):
    def __init__(self, wordvectors, vocab, affixes, affixNeighbours, eWordToEngParents,
                 dictionary=None,
                 alphabet=string.ascii_lowercase, dictvectorizer=None,
                 weightvector=None, segmentations=None,
                 secondLangChain=None):
        super(TwoLangMorphoChain, self).__init__(
            wordvectors, vocab, affixes, affixNeighbours, dictionary,
            alphabet, dictvectorizer, weightvector,
            segmentations)
        self.secondLangChain = secondLangChain
        self.eWordToEngParents = eWordToEngParents

    def genCandidates(self, word):
        ocandidates = super(TwoLangMorphoChain, self).genCandidates(word)
        if word not in self.eWordToEngParents:
            return ocandidates
        canddict = {cand.parentword: cand for cand in ocandidates}
        for parent, transscore in self.eWordToEngParents[word]:
            if parent in canddict:
                canddict[parent] = canddict[parent]._replace(olangconfidence=transscore)
            else:
                canddict[parent] = ParentTransformation(parent, ParentType.OLANG, transscore)

        return list(canddict.values())

    def getFeatures(self, w, z, maxCosSimilarity=None):
        d = super(TwoLangMorphoChain, self).getFeatures(w, z, maxCosSimilarity)
        if z.olangconfidence is not None:
            d['olangconfidence'] = z.olangconfidence
        else:
            d['no_translated_parents'] = 1
        if z.transformtype != ParentType.OLANG and z.olangconfidence is not None:
            d['repeat'] = z.olangconfidence
        else:
            d['no_repeat'] = 1
        return d


def parentHeuristic(word, parent):
    dist = editdistance.eval(word, parent)
    return dist < 6 and dist < len(word) / 2 and dist < len(parent) / 2


def init2LangCache(translations, invTranslations, secondLangChain):
    tWordToEngParents = {}
    for tWord in invTranslations:  # every Turkish word that can be translated to
        turkishParents = [(z.parentword, c) for z, c in secondLangChain.predict(tWord)]
        englishParents = {}
        turkishParents.sort(key=lambda x: x[1])  # sort by increasing confidence
        for tParent, c in turkishParents:
            if tParent in invTranslations:
                for eParent in invTranslations[tParent]:
                    englishParents[eParent] = c  # englishParents[eParent] is maximum confidence

        tWordToEngParents[tWord] = sorted(englishParents.items(), key=lambda x: x[1])

    eWordToEngParents = {}
    for eWord, tTranslations in translations.items():
        possibleParents = []
        for tWord in tTranslations:
            possibleParents.extend(tWordToEngParents[tWord])
        possibleParents.sort(key=lambda x: x[1])
        englishParents = {}
        for parent, score in possibleParents:
            englishParents[parent] = score
        eWordToEngParents[eWord] = [(parent, score) for parent, score in possibleParents.items()
                                            if parentHeuristic(eWord, parent)]
    return eWordToEngParents


if __name__ == "__main__":
    mc_args = (get_wordvectors(TWOLANG_VEC, TWOLANG_LANG), get_wordlist(TWOLANG_VOCAB, TWOLANG_LANG)) + \
            get_prefixes_affixes(TWOLANG_LANG)
    mc_kwargs = {}
    with open('out_py/dictvectorizer.%s.%s-%s.p' % (TWOLANG_LANG, TWOLANG_LANG, TWOLANG_LANG), 'rb') as f:
        mc_kwargs['dictvectorizer'] = pickle.load(f)
    with open('out_py/weights.%s%s.%s-%s.p' % (TWOLANG_LANG, '.supervised%s' % TWOLANG_SUPERVISED
                                                if TWOLANG_SUPERVISED else '', TWOLANG_VOCAB, TWOLANG_VEC), 'rb') as f:
        mc_kwargs['weightvector'] = pickle.load(f)

    turkishMorpho = MorphoChain(*mc_args, **mc_kwargs)
    translations, invTranslations = read_dictionary('sozluk.txt')
    cache = init2LangCache(translations, invTranslations, turkishMorpho)
    with open('data/eWordToEParents.p', 'wb') as f:
        pickle.dump(cache, f)
