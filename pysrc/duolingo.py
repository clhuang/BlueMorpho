from morphochain import MorphoChain, ParentTransformation, ParentType
import string
import editdistance


class TwoLangMorphoChain(MorphoChain):
    def __init__(self, wordvectors, vocab, affixes, affixNeighbours, translations,
                 dictionary=None,
                 alphabet=string.ascii_lowercase, dictvectorizer=None,
                 weightvector=None, segmentations=None,
                 secondLangChain=None):
        super(TwoLangMorphoChain, self).__init__(
            wordvectors, vocab, affixes, affixNeighbours, dictionary,
            alphabet, dictvectorizer, weightvector,
            segmentations, translations)
        self.translations, self.invTranslations = translations
        self.secondLangChain = secondLangChain
        self.init2LangCache()

    def parentHeuristic(word, parent):
        dist = editdistance.eval(word, parent)
        return dist < 6 and dist < len(word) / 2 and dist < len(parent) / 2

    def init2LangCache(self):
        tWordToEngParents = {}
        for tWord in self.invTranslations:  # every Turkish word that can be translated to
            turkishParents = [(z.parentword, c) for z, c in self.secondLangChain.predict(tWord)]
            englishParents = {}
            turkishParents.sort(key=lambda x: x[1])  # sort by increasing confidence
            for tParent, c in turkishParents:
                if tParent in self.invTranslations:
                    for eParent in self.invTranslations[tParent]:
                        englishParents[eParent] = c  # englishParents[eParent] is maximum confidence

            tWordToEngParents[tWord] = sorted(englishParents.items(), key=lambda x: x[1])

        self.eWordToEngParents = {}
        for eWord, tTranslations in self.translations.items():
            possibleParents = []
            for tWord in tTranslations:
                possibleParents.extend(tWordToEngParents[tWord])
            possibleParents.sort(key=lambda x: x[1])
            englishParents = {}
            for parent, score in possibleParents:
                englishParents[parent] = score
            self.eWordToEngParents[eWord] = [(parent, score) for parent, score in possibleParents.items()
                                             if self.parentHeuristic(eWord, parent)]

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
