from morphochain import MorphoChain
import string

class TwoLangMorphoChain(MorphoChain):
    def __init__(self, wordvectors, vocab, affixes, affixNeighbours, dictionary=None,
                 alphabet=string.ascii_lowercase, dictvectorizer=None,
                 weightvector=None, segmentations=None, translations=None,
                 secondLangChain=None):
        super(TwoLangMorphoChain).__init__(
            wordvectors, vocab, affixes, affixNeighbours, dictionary,
            alphabet, dictvectorizer, weightvector,
            segmentations, translations)
        self.secondLangChain = secondLangChain

    def gen2LangCandidates(self, word):
        candidates = []
        for tWord in self.translations(word):
            tParent = self.secondLangChains.predict(tWord)
            for parent in self.invTranslations(tParent):
                if parent in self.secondLangChains.vocab:
                    candidates.append(parent)

    def genCandidates(self, word):
        ocandidates = super(TwoLangMorphoChain, self).genCandidates(word)
#some more stuff



