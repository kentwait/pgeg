
"""
Created on Tue Oct 14 15:50:10 2014
tajima_d.py
@author: kent
"""

from __future__ import print_function
import numpy as np


def pairwise_polymorphism(pairwise_array):
    counter = 0
    for i in range(len(pairwise_array[0])):
        counter += 0 if pairwise_array[0,i] == pairwise_array[1,i] else 1
    return counter


def generate_pairwise_combinations(lst):
    for i in range(len(lst)):
        element1 = lst[i]
        for element2 in lst[i+1:]:
            yield (element1, element2)


def theta_hat_pi(msa_array):
    """Returns the average number of polymorphisms between two sequences

    This is an unbiased estimator of population mutation parameter Theta
    """
    polymorphisms = []
    for _ in generate_pairwise_combinations(msa_array):
        count = pairwise_polymorphism(np.array(_))
        polymorphisms.append(count)
    return sum(polymorphisms)/float(len(polymorphisms))

def theta_hat_S(msa_array):
    '''
    '''
    # Get number of segragating sites
    S = 0
    for i in range(len(msa_array[0])):
        uniques = set(msa_array[:,i])
        if len(uniques) > 1:
            S += 1

    # Compute summation 1/i
    n = len(msa_array)
    a1 = 0
    for i in range(1,n):
        a1 += 1/float(i)

    return S/float(a1), S, a1

def tajima_d(msa_array):
    '''Returns Tajima's d statistic
    '''
    theta_pi = theta_hat_pi(msa_array)
    theta_S, S, a1 = theta_hat_S(msa_array)

    # Compute standard error --> sqrt(variance)
    n = len(msa_array)
    a1 = sum([1/float(i) for i in range(1, n)])
    a2 = sum([1/np.square(float(i)) for i in range(1, n)])
    b1 = (n + 1) / float(3 * (n - 1))
    b2 = (2 * (np.square(n) + n + 3)) / float(9*n * (n - 1))
    c1 = b1 - (1/float(a1))
    c2 = b2 - ((n + 2) / float(a1*n)) + (a2 / float(np.square(a1)))
    e1 = c1 / float(a1)
    e2 = c2 / float(np.square(a1) + a2)

    std_error = np.sqrt((e1*S) + (e2*S*(S - 1)))
    d = theta_pi - theta_S

    return d, {'theta_pi': theta_pi,
               'theta_S': theta_S,
               'std_error': std_error,
               'S': S,
               'n': n}

def generate_window(msa_array, window_size=100, increment=1):
    '''
    '''
    for i in range(0, len(msa_array[0]) - window_size, increment):
        yield msa_array[:, i:i+window_size]


if __name__ == '__main__':
    from Bio import AlignIO
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute Tajima\'s d given a multiple sequence alignment')
    parser.add_argument('-i', '--input', help='Multiple sequence alignment (FASTA)', required=True)
    args = vars(parser.parse_args())
    print('Calculating Tajima\'s D using alignment {0}...'.format(args['input']), end='\n\n')
    codon_aln = AlignIO.read(args['input'], 'fasta')
    codon_array = np.array([[str(rec.seq[i:i+3]) for i in xrange(0,len(rec),3) if i+3 <= len(rec)] for rec in codon_aln], np.character, order="F")
    d, dct = tajima_d(codon_array)

    print('Number of sequences (n):', dct['n'])
    print('Alignment length:', len(codon_array[0]), end='\n\n')
    print('Number of segregating sites (S):', dct['S'])

    print(u'\u03B8\u03C0' + ':', dct['theta_pi'])
    print(u'\u03B8' + 's:', dct['theta_S'])
    print('std error:', dct['std_error'], end='\n\n')

    print('d:', d)
    print('D:', d/float(dct['std_error']), end='\n\n')
