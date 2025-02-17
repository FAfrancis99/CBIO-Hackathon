import itertools
import random

import numpy as np
import pandas as pd


class KMer:

    def __init__(self, k):
        """
    Initialize the k-mer object.

    Args:
    k: int, the "k" in k-mer
    """
        self.keys_set = ["".join(p) for p in itertools.product(['A', 'T', 'C', 'G'], repeat=k)]
        self.k = k
        self.letters = ['A', 'T', 'C', 'G']
        self.multiplyBy = 4 ** np.arange(k - 1, -1,
                                         -1)  # Multiplier for each digit in the k-number system
        self.n = 4 ** k  # Number of possible k-mers

    def add_kmer_features_to_dataframe(self, df,
                                       write_number_of_occurrences=False):
        """
    Add k-mer features to the DataFrame.

    Args:
    df: pandas DataFrame, must contain a 'genes' column with DNA sequences
    write_number_of_occurrences: bool, if True, count occurrences; else, use percentage
    """
        kmer_features_list = []
        for seq in df['genes']:
            kmer_feature = self.obtain_kmer_feature_for_one_sequence(
                seq.upper(), write_number_of_occurrences)
            kmer_features_list.append(kmer_feature)

        kmer_features_df = pd.DataFrame(kmer_features_list,
                                        columns=[str(i) for i in
                                                 self.keys_set])
        kmer_features_df.fillna(0, inplace=True)
        df = pd.concat([df, kmer_features_df], axis=1)
        return df

    def obtain_kmer_feature_for_one_sequence(self, seq,
                                             write_number_of_occurrences=False):
        """
    Get the k-mer feature vector for a single DNA sequence.

    Args:
    seq: str, a DNA sequence
    write_number_of_occurrences: bool, if False, use percentage; else count occurrences
    """
        number_of_kmers = len(seq) - self.k + 1
        kmer_feature = {}

        for i in range(number_of_kmers):
            this_kmer = seq[i:(i + self.k)]
            this_numbering = self.kmer_numbering_for_one_kmer(this_kmer)
            kmer_feature[self.convert_ambiguity(this_kmer)] = kmer_feature.get(self.convert_ambiguity(this_kmer), 0) + 1

        if not write_number_of_occurrences:
            kmer_feature = {i: k / number_of_kmers for i, k in kmer_feature.items()}

        return kmer_feature

    def convert_ambiguity(self, kmer_):
        # Define the ambiguity codes and their possible substitutions
        ambiguity_dict = {
            'N': ['A', 'G', 'C', 'T'],
            'K': ['G', 'T'],
            'W': ['A', 'T'],
            'Y': ['C', 'T'],
            'M': ['A', 'C'],
            'B': ['C', 'G', 'T'],
            'V': ['A', 'C', 'G'],
            'S': ['C', 'G'],
            'R': ['A', 'G']
        }

        # Replace each ambiguity code with a random choice from its possible nucleotides
        return ''.join(
            random.choice(
                ambiguity_dict[char]) if char in ambiguity_dict else char
            for char in kmer_
        )
    def kmer_numbering_for_one_kmer(self, kmer):
        """
    Convert a k-mer into its numeric index.

    Args:
    kmer: str, a k-mer
    """
        kmer = self.convert_ambiguity(kmer)
        digits = [self.letters.index(letter) for letter in kmer]
        numbering = np.dot(digits, self.multiplyBy)
        return numbering
