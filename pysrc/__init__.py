from pysrc import fileio, morphochain, objective
import pprint
import os

try:
    import cPickle as pickle
except:
    import pickle


def get_prefixes_affixes(lang='eng'):
    assert lang == 'eng' or lang == 'tur'

    with open('data/%s_prefix_list.p' % lang, 'rb') as f:
        prefixes = pickle.load(f)
    with open('data/%s_suffix_list.p' % lang, 'rb') as f:
        suffixes = pickle.load(f)
    with open('data/%s_prefix_corr2.p' % lang, 'rb') as f:
        prefixes_corr = pickle.load(f)
    with open('data/%s_suffix_corr2.p' % lang, 'rb') as f:
        suffixes_corr = pickle.load(f)
    affixes = (prefixes, suffixes)
    affix_corr = (prefixes_corr, suffixes_corr)
    return affixes, affix_corr


def get_segmentations(lang='eng'):
    assert lang == 'eng' or lang == 'tur'

    with open('data/%s_traincorpus.p' % lang, 'rb') as f:
        trainsegmentations = pickle.load(f)
    with open('data/%s_devcorpus.p' % lang, 'rb') as f:
        devsegmentations = pickle.load(f)
    return trainsegmentations, devsegmentations


def get_wordlist(size='small', lang='eng'):
    if size == 'full':
        fname = 'data/wordlist-2010.%s.txt' % lang
    else:
        fname = 'data/wordlist-2010.%s.%s.txt' % (lang, size)
    return fileio.read_wordcounts(fname)


def get_wordvectors(size='small', lang='en'):
    if lang == 'eng':
        lang = 'en'
    if lang == 'tur':
        lang = 'tr'
    if size == 'full':
        size = 'filtered'
    file_v = 'data/%s-wordvectors200_%s.txt' % (lang, size)
    binfile_v = file_v[:-3] + 'bin'
    if os.path.isfile(binfile_v):
        return fileio.load_wordvectors(binfile_v, binary=True)
    return fileio.load_wordvectors(file_v)
