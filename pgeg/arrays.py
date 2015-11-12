import os
import numpy as np
import re
from collections.abc import MutableMapping
from collections import OrderedDict, namedtuple, Counter
import collections
import pandas as pd

# Nucleotide and amino acid constants
BASES = 'TCAG'
CODONS = [a+b+c for a in BASES for b in BASES for c in BASES]
AMINO_ACIDS = 'ARNDCQEGHILKMFPSTWYV'

# Generate genetic code from CODONS and AMINO ACIDS
#
# Description
# -----------
# `transl_code` is an intermediate string with AA letter value at even positions and corresponding count at odd
# positions. `transl_code` creates a new string `transl` of amino acids matching the order of codons in `CODONS`.
# `transl` is created by inserting AA letters into the string based on the number right of it.
#
# Once created, `transl` has the same length as well as order of its corresponding codon in `CODONS`.
# Then, the corresponding codon-AA pair are stored in an OrderedDict `GENETIC_CODE`.
# Note that stop codon 1-letter AA are "*"
#
# Codon-AA pairs can also be retrieved by position in the `GENETIC_CODE` OrderedDict.
# The function `genetic_code_by_index` uses 1-based indexing to retrieve codon-AA pairs such that the first pair in the
# OrderedDict is retrieved using x[1] instead of x[0].
#
transl_code = 'F2L6I3M1V4S4P4T4A4Y2*2H2Q2N2K2D2E2C2*1W1R4S2R2G4'
transl = ''.join([str(a*int(b)) for a,b in zip(transl_code[::2], transl_code[1::2])])
GENETIC_CODE = OrderedDict(zip(CODONS, transl))
STOP_CODONS = ['TGA', 'TAG', 'TAA']
genetic_code_by_index = lambda x: list(GENETIC_CODE.items())[x - 1]

# Determine degeneracy of genetic code for a particular codon
#
# Description
# -----------
# Codon-AA do not have unique one-to-one correspondence. Many AAs correspond to more than one codon.
# To determine if the AA a particular codon is coding for is degenerate and if so, how many codons code for the same AA
# (-fold degeneracy), the number of times an AA appears as the value in `GENETIC_CODE` is counted and referenced back
# to the particular codon.
_codon_to_aa_count = Counter(GENETIC_CODE.values())
CODON_FOLD = OrderedDict(
    sorted([(codon, _codon_to_aa_count[aa]) for codon, aa in GENETIC_CODE.items()], key=lambda x: x[1]))

MSA = namedtuple('MSA', 'ids alignment')

class Sequence(MutableMapping):
    """Multiple sequence class

    This class
    """
    def __init__(self, input_obj, seqtype='nucl', name='', description=''):
        """
        Make a new instance of a Sequence object from a dictionary, file or FASTA-formatted string.

        @param input_obj: Object used to populate a Sequence object. This may be one of the following:
        - Dictionary-like object whose id are sequence record names and values are the corresponding sequences
        - Path to a FASTA file
        - FASTA-formatted string
        @param seqtype: 'nucl' (Nucleotide), 'prot' (Protein), 'cod' (Codon-based)
        @param name: Name of the set of sequence records
        """
        self.seqtype = seqtype
        self.name = name
        self.description = description

        if isinstance(input_obj, dict):
            records = input_obj
            self._ids = list(records.keys())
            self._sequences = list(records.values())
        elif isinstance(input_obj, str):
            # Test if file path
            if os.path.exists(input_obj):
                records = Sequence.parse_fasta(input_obj)
                self._ids = [key.split(' ')[0] for key in records.keys()]
                self._sequences = list(records.values())
            else:
                raise NotImplementedError('Passing FASTA-formatted strings are not yet supported. \
                Instantiate using an OrderedDict or passing a valid filepath instead.')


    # Restrict setting "ids" attribute outside of __init__
    @property
    def ids(self):
        return self._ids

    @ids.setter
    def ids(self, value):
        raise AttributeError('Setting ids using this method is not permitted.')

    @ids.deleter
    def ids(self):
        raise AttributeError('Deleting ids using this method is not permitted.')

    # Restrict setting "sequences" attribute outside of __init__
    @property
    def sequences(self):
        return self._sequences

    @sequences.setter
    def sequences(self, value):
        raise AttributeError('Setting sequences using this method is not permitted.')

    @sequences.deleter
    def sequences(self):
        raise AttributeError('Deleting sequences using this method is not permitted.')

    def __setitem__(self, key, sequence):
        self.ids.append(key)
        self.sequences.append(sequence)

    def __getitem__(self, keys):
        if isinstance(keys, collections.Iterable) and not isinstance(keys, str):
            return_list = []
            for key in keys:
                if key in self.ids:
                    return_list.append(self.sequences[self.ids.index(key)])
                else:
                    raise KeyError('Key "{0}" does not exist'.format(key))
            return return_list
        else:
            key = keys
            if key in self.ids:
                return self.sequences[self.ids.index(key)]
            else:
                raise KeyError('Key "{0}" does not exist'.format(key))

    def __delitem__(self, key):
        if key in self.ids:
            index = self.ids.index(key)
            self.ids.remove(key)
            self.sequences.pop(index)
        else:
            raise KeyError('Key "{0}" does not exist'.format(key))

    def __iter__(self):
        for key, sequence in zip(self.ids, self.sequences):
            yield key, sequence

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return 'Sequences({0})'.format([(k, v) for k, v in zip(self.ids, self.sequences)])



    def tofasta(self, path, linewidth=60):
        with open(path, 'w') as f:
            print(Sequence.array_to_fasta(self.ids, self.sequences, linewidth=linewidth), file=f)

    def align(self, aln_file='out.aln', program='muscle'):
        """
        Calls an external alignment program to align the sequences (nucleotide or protein). Returns an Alignment object.

        @param aln_file: File path of resulting multiple sequence alignment.
        @param program: External program to be called for multiple sequence alignment. Currently supported programs are
            'muscle' (Muscle), 'mafft' (MAFFT), 'clustalw' (ClustalW), 'clustalo' (Clustal Omega), 'prank' (PRANK).

            To ensure that this method works properly, make sure that these programs are installed and
            accessible from the system's PATH.
        @return: Evogen Alignment object of the resulting multiple sequence alignment.
        """
        # check if program is in choices or not. if not return an error
        choices = ['muscle', 'mafft', 'clustalo']
        assert program in choices, Exception('Program not supported. Choose from the following: \
        "muscle", "mafft", "clustalw", "clustalo", "prank"')

        # Write the Sequence object to file
        seqfile = '{0}.{1}'.format(self.name, 'fna' if self.seqtype == 'nucl' else 'faa')
        self.tofasta(seqfile)

        # Default to MUSCLE
        if program == 'mafft':
            cmd_str = 'mafft --auto {i} > {o}'.format(i=seqfile, o=aln_file)
        elif program == 'clustalw':  # TODO : ClustalW hook
            raise Exception('ClustalW support is in development.')
        elif program == 'clustalo':
            cmd_str = 'clustalo -i {i} -o {o} '.format(i=seqfile, o=aln_file)
        elif program == 'prank':  # TODO : PRANK hook
            raise Exception('PRANK support is in development.')
        else:
            # Default to MUSCLE
            cmd_str = 'muscle -in {i} -out {o}'.format(i=seqfile, o=aln_file)
        # TODO : change to subprocess
        os.system(cmd_str)
        if self.seqtype == 'nucl':
            return NucleotideAlignment(aln_file)
        elif self.seqtype == 'prot':
            return ProteinAlignment(aln_file)

    @staticmethod
    def parse_fasta(path):
        """
        Parse sequences in FASTA format entirely using only built-ins.

        @param path: File path (aboslute or relative) where the FASTA file is located.
        @return: OrderedDict sequence records. FASTA headers are stored as dictionary keys and its corresponding
            sequence is stored as its value.
        """
        keys = []
        sequence = ''
        sequences = []
        with open(path, 'r') as f:
            for line in f.readlines():
                if line.startswith('>'):
                    keys.append(line[1:-1])
                    if sequence:
                        sequences.append(sequence)
                        sequence = ''
                else:
                    sequence += re.sub('\s', '', line)
            if sequence:
                sequences.append(sequence)
        return OrderedDict(zip(keys, sequences))

    @staticmethod
    def array_to_fasta(keys, sequences, linewidth=60):
        """
        Writes a FASTA-formatted string based on a corresponding name and sequence lists

        @param keys: List of record names
        @param sequences: List of sequences (list of lists or 2d ndarray)
        @param linewidth: Number of characters per line
        @return: FASTA-formatted string
        """
        fasta_str = ''
        for key, sequence in zip(keys, sequences):
            sequence = ''.join(sequence)
            header = '>{0}'.format(key)
            fasta_str += header + '\n'
            for i in range(0, len(sequence), linewidth):
                curr_line = sequence[i:i+linewidth]
                fasta_str += curr_line + '\n'
        return fasta_str
    @staticmethod
    def composition(sequence_obj, seqtype='nucl'):
        """
        Return the composition of a Sequence
        @return: namedtuple of percent makeup for each character except gaps
        """
        #assert re.search('^[ATCG\-]+$', sequence), 'Input sequence contains characters other than A,T,C,G,-'
        composition_of = OrderedDict()
        characters = BASES if seqtype in ['nucl', 'cod'] else AMINO_ACIDS
        for seqid, sequence in zip(sequence_obj.ids, sequence_obj.sequences):
            char_counts = Counter(sequence.upper())
            total = sum([v for k, v in char_counts.items() if k != '-'])
            composition_of[seqid] = OrderedDict([(k, char_counts[k]/float(total)) for k in characters])
        return pd.DataFrame(composition_of)


class NucleotideSequence(Sequence):
    def __init__(self, input_obj, name='', description=''):
        super().__init__(input_obj, name=name, seqtype='nucl', description=description)
        self.translated = self.__translate()

    @staticmethod
    def nucleotide_to_codon(nucleotide_str):
        """
        Generator that converts a nucleotide triplet into its corresponding codon

        @param nucleotide_str: Nucleotide sequence (str or list)
        @yield: 3-character string codon
        """
        if len(nucleotide_str) % 3 != 0:
            raise ValueError('Sequence length is not a multiple of three ({0}).'.format(len(nucleotide_str)))
        for j in range(0, len(nucleotide_str), 3):
            if j+3 <= len(nucleotide_str):
                yield nucleotide_str[j:j+3]

    def __translate(self):
        """
        Translates nucleotide sequences into amino acid sequences.

        Assumes that the nucleotide sequence is coding,
        in-frame, and the start of the ORF corresponds to the beginning of the nucleotide sequence.
        @return: ProteinSequence object
        """
        transl_dct = OrderedDict()
        for key, nt_seq in zip(self.ids, self.sequences):
            transl_dct[key] = ''.join([CODON_TO_AA[cod.upper()]
                                       for cod in NucleotideSequence.nucleotide_to_codon(nt_seq)])
        return ProteinSequence(transl_dct)

    def codonalign(self, codon_aln_file='out.ffn.aln', program='muscle'):
        """
        Aligns by codons using a protein alignment generated by an external program. Returns a CodonAlignment object.

        Assumes the sequence is in-frame and is a coding sequences. First, the sequences are translated into proteins,
        which are aligned by an external program. The resulting protein alignment is used as an anchor to align
        nucleotide sequences.

        @param codon_aln_file: File path of resulting codon-aligned multiple sequence alignment
        @param program: External program to be called to align translated protein sequences. Currently supported
            programs are 'muscle' (Muscle), 'mafft' (MAFFT), 'clustalw' (ClustalW), 'clustalo' (Clustal Omega),
            'prank' (PRANK)
        @return: CodonAlignment object of the resulting codon-based multiple sequence alignment.
        """
        if self.seqtype != 'nucl':
            raise Exception('Seqtype must be "nucl" (nucleotide) to perform codon alignment.')
        for i, sequence in enumerate(self.sequences):
            if len(sequence) % 3 != 0:
                raise ValueError('"{0}" sequence length is not a multiple of three ({1}).'
                                 .format(self.ids[i], len(sequence)))

        # check if program is in choices or not. if not return an error
        choices = ['muscle', 'mafft', 'clustalo']
        assert program in choices, 'Program not supported. Choose from the following: \
        "muscle", "mafft", "clustalw", "clustalo", "prank"'

        # Write translated Sequence object to file
        transl_seqfile = '{0}.transl.{1}'.format(self.name, 'faa')
        self.translated.tofasta(transl_seqfile)

        # Align protein sequences
        aa_aln = self.translated.align(aln_file=codon_aln_file, program=program)

        # Adjust codons based on amino acid alignment
        codon_aln = OrderedDict()
        i = 0
        for nt_seq, aa_aln_seq in zip(self.sequences, aa_aln.sequences):
            codon = NucleotideSequence.nucleotide_to_codon(nt_seq)
            codon_aln[self.ids[i]] = ''.join([next(codon) if aa != '-' else '---' for aa in list(aa_aln_seq)])
            i += 1
        codon_aln = CodonAlignment(codon_aln)
        codon_aln.tofasta(codon_aln_file)
        return codon_aln

    def basecomp(self):
        """
        Return the base composition of a NucleotideSequence
        @return: namedtuple of 'A', 'T', 'G', 'C', 'AT', 'GC' percent
        """
        #assert re.search('^[ATCG\-]+$', sequence), 'Input sequence contains characters other than A,T,C,G,-'
        basecomp_of = super().composition(self, seqtype=self.seqtype).T
        basecomp_of['AT'] = basecomp_of['A'] + basecomp_of['T']
        basecomp_of['GC'] = basecomp_of['G'] + basecomp_of['C']
        return basecomp_of.T


class ProteinSequence(Sequence):
    def __init__(self, input_obj, name='', description=''):
        super().__init__(input_obj, name=name, seqtype='prot', description=description)

    def aacomp(self):
        return super().composition(self, seqtype=self.seqtype)


class GenericAlignment(MutableMapping):
    """
    Multiple Sequence Alignment base class
    --------------------------------------

    The object specified by this class is a combination of a list of sequence names and a 2d numpy ndarray that
    represents the alignment. Thus, each sequence record is a key-value pairing of a sequence name and its corresponding
    sequence in the ndarray.

    Records can be accessed by its key like a dictionary. In addition, multiple records can be accessed simultaneously
    by passing a list. Record values are returned as a numpy ndarray based on the order of keys passed.

    Methods that permutate the alignment other than adding or deleting records will return a new instance of the
    alignment. No in-place changes take place when using these methods.

    This is the base class for NucleotideAlignment, ProteinAlignment and CodonAlignment.
    """
    def __init__(self, input_obj, seqtype, charsize=1, name='', description=''):
        """
        Create an Alignment object

        @param input_obj: Alignment objects can be instantiated by passing one of the following:
        - Dictionary-like object
        - File path (absolute or relative)
        - FASTA-formatted string #TODO
        @param charsize: Number of characters that define a column of the alignment.
        """
        # Description
        self.seqtype = seqtype
        self.name = name
        self.description = description
        self.charsize = charsize

        # Tuple containing an ordered list of ids and a numpy array of the sequence
        if isinstance(input_obj, tuple):
            if len(input_obj) == 2:
                if isinstance(input_obj[0], list):
                    self._ids = input_obj[0]
                else:
                    raise TypeError('First item of tuple is not a list.')
                if isinstance(input_obj[1], np.ndarray):
                    self._sequences = input_obj[1]
                else:
                    raise TypeError('Second item in tuple is not a numpy ndarray.')
        else:
            list_of_sequences = []

            # dictionary of id as index and sequence stored as str as the value
            if isinstance(input_obj, dict):
                self._ids = []
                for k, v in input_obj.items():
                    if not isinstance(v, str):
                        raise TypeError('Sequence "{0}" is not a string.'.format(k))
                    self._ids.append(k)
                    list_of_sequences.append(v)
                # Check if sequences are of equal length
                list_of_sequence_lengths = set([len(s) for s in list_of_sequences])
                if len(list_of_sequence_lengths) > 1:
                    raise ValueError('Unequal sequence lengths.')
            # String path to FASTA file
            elif isinstance(input_obj, str):
                # Check if it is a valid path, and the file exists
                if os.path.exists(input_obj):
                    # Parse FASTA file
                    fasta_dct = self.parse_fasta(input_obj)
                    # Store record ID as list
                    self._ids = list(fasta_dct.keys())
                    list_of_sequences = list(fasta_dct.values())
                # TODO : Test if the string is in FASTA format
                else:
                    raise Exception('Passing FASTA-formatted strings are not yet supported. \
                    Instantiate using an OrderedDict or passing a valid filepath instead.')
            # Store sequences as a numpy array. Order of array rows should correspond to the order of IDs
            # in the record list

            # Check if length is divisible by charsize
            assert len(list_of_sequences[0]) % self.charsize == 0, \
                ValueError('Alignment length is not divisible by 3 ({} nt)'.format(len(list_of_sequences[0])))
            self._sequences = np.array(
                [[seq[j:j+self.charsize] for j in range(0, len(seq), self.charsize) if j+self.charsize <= len(seq)]
                 for seq in list_of_sequences], dtype='U1' if charsize == 1 else 'U3')

        # Size of alignment
        self.count = len(self)
        self.length = self.sequences.shape[-1]
        self.shape = self.sequences.shape

    # TODO : Make a "restricted" descriptor for any type of attribute that should not be changed outside of __init__
    # Restrict setting "ids" attribute outside of __init__
    @property
    def ids(self):
        return self._ids

    @ids.setter
    def ids(self, value):
        raise AttributeError('Setting ids using this method is not permitted.')

    @ids.deleter
    def ids(self):
        raise AttributeError('Deleting ids using this method is not permitted.')

    # Restrict setting "sequences" attribute outside of __init__
    @property
    def sequences(self):
        return self._sequences

    @sequences.setter
    def sequences(self, value):
        raise AttributeError('Setting sequences using this method is not permitted.')

    @sequences.deleter
    def sequences(self):
        raise AttributeError('Deleting sequences using this method is not permitted.')

    def __setitem__(self, key, value):
        """
        Add or update a sequence record.

        @param key: Record name
        @param value: Sequence
        """
        if key in self.ids:
            raise KeyError('Key name {0} already in use.'.format(key))
        else:
            # Check if has the same length as the number of cols of the current array
            if len(value) == self.length:
                self.ids.append(key)
                self.sequences = np.vstack(
                    [self.sequences, [value[j:j+self.charsize] for j in range(0, len(value), self.charsize)
                                      if j+self.charsize <= len(value)]])
            else:
                raise ValueError('New sequence length {0} does not match alignment length {1}.'
                                 .format(len(value), self.length))

    def __getitem__(self, keys):
        """
        Retrieve a record or multiple records using its corresponding key. Pass a list to retrieve multiple entries.

        @param keys: Record name or list of record names
        """
        if isinstance(keys, collections.Iterable) and not isinstance(keys, str):
            index_list = []
            for key in keys:
                if key in self.ids:
                    index_list.append(self.ids.index(key))
                else:
                    raise Exception('Key "{0}" does not exist'.format(key))
            return self.sequences[index_list]
        else:
            key = keys
            if key in self.ids:
                index = self.ids.index(key)
            else:
                raise KeyError('Key "{0}" does not exist'.format(key))
            return self.sequences[index]

    def __delitem__(self, key):
        if key in self.ids:
            index = self.ids.index(key)
            self.ids.remove(key)
            self.sequences = np.delete(self.sequences, index, axis=0)  # 0 means by row
        else:
            raise KeyError('Key "{0}" does not exist'.format(key))

    def __iter__(self):
        for key, sequence in zip(self.ids, self.sequences):
            yield key, sequence

    def __len__(self):
        # Return the number of samples in the alignment
        return len(self.ids)

    def __repr__(self):
        return 'keys({0})\n{1}'.format(repr(self.ids), repr(self.sequences))

    def __add__(self, other):
        # Check if self.ids and other.ids match
        if set(self.ids) != set(other.ids):
            raise KeyError('Keys do not match.')
        if self.seqtype != other.seqtype:
            raise ValueError('Seqtypes do not match.')
        other_order = [other.ids.index(key) for key in self.ids]
        return type(self)(
            MSA(ids=self.ids, alignment=np.concatenate((self.sequences, other.sequences[other_order]), axis=1)),
            self.seqtype)

    def __iadd__(self, other):
        return self + other

    def head(self):
        return type(self)(MSA(ids=self.ids[:5], alignment=self.sequences[:5]), self.seqtype)

    def tail(self):
        return type(self)(MSA(ids=self.ids[-5:], alignment=self.sequences[-5:]), self.seqtype)

    def colx(self, *args):
        """
        Returns a length-wise subset of the alignment.

        @param slice: Inclusive start and exclusive end position of the subset. Follows Python slice conventions.
        @return: Alignment object of the alignment subset
        """
        if len(args) == 1:
            return type(self)(MSA(ids=self.ids, alignment=self.sequences[:, args[0]]), self.seqtype)
        elif len(args) == 2:
            return type(self)(MSA(ids=self.ids, alignment=self.sequences[:, args[0]:args[1]]), self.seqtype)
        elif len(args) == 3:
            return type(self)(MSA(ids=self.ids, alignment=self.sequences[:, args[0]:args[1]:args[2]]),
                              self.seqtype)
        else:
            raise Exception('Method uses 3 integer arguments at most.')

    def subset(self, keys):
        return type(self)(MSA(ids=keys, alignment=self[keys]),
                              self.seqtype)

    def labelpartition(self, label, start, end, coding=True):  # TODO
        pass

    def xgap(self):
        """
        Remove columns from the current alignment that contain at least one gap character. Returns a new alignment.

        @return: Alignment object containing the gap-free alignment
        """
        gapchar = '-'*self.charsize
        xgap_cols = []
        for i in range(self.length):
            if gapchar not in self.sequences[:, i]:
                xgap_cols.append(i)
        return type(self)(MSA(ids=self.ids, alignment=self.sequences[:, xgap_cols]), self.seqtype)

    def bootstrap(self):
        """
        Resamples the columns of the alignment and returns a new instance.

        @return: Alignment object containing the resampled alignment
        """
        randlist = np.random.random(self.length, self.length, replace=True)
        return type(self)(MSA(ids=self.ids, alignment=self.sequences[:, randlist]), self.seqtype)

    def reorder(self, ordered_key_list):
        """
        Reorder alignment based on the order of a a given list of keys
        @param ordered_key_list: list
        @return: Alignment object containing the reordered alignment
        """
        sequences = []
        for key in ordered_key_list:
            index = self.ids.index(key)
            sequences.append(self.sequences[index])
        return type(self)(MSA(ids=ordered_key_list, alignment=np.array(sequences)), self.seqtype)

    def tofasta(self, path, linewidth=60):
        """
        Save the alignment as a FASTA-formatted file

        @param linewidth: Number of characters per line
        """
        # TODO : Check if basedir of path exists
        with open(path, 'w') as f:
            print(self.__class__.alignment_to_fasta(self, linewidth=linewidth), file=f)

    def tophylip(self, path):  # TODO
        pass

    @staticmethod
    def parse_fasta(path):
        return Sequence.parse_fasta(path)

    @staticmethod
    def concat(*alignments):
        """
        Concatenate multiple Alignment objects together
        @param alignments: Alignment objects
        @return: Alignment object concatenated in order
        """
        concaternated_alignment = alignments[0]
        for alignment in alignments[1:]:
            concaternated_alignment += alignment
        return concaternated_alignment

    @staticmethod
    def alignment_to_fasta(alignment, linewidth=60):
        """
        Save the alignment as a FASTA-formatted file

        @param alignment: Alignment object
        @param linewidth: Number of characters per line
        @return: FASTA-formatted string
        """
        return Sequence.array_to_fasta(alignment.ids, alignment.sequences, linewidth=linewidth)

    @staticmethod
    def composition(alignment_obj, seqtype='nucl'):
        """
        Return the character composition of an Alignment
        @return: namedtuple of percent makeup for each character except gaps
        """
        sequence_obj = Sequence(
            OrderedDict([(seqid, ''.join(sequence_as_list))
                         for seqid, sequence_as_list in zip(alignment_obj.ids, alignment_obj.sequences)]))
        return Sequence.composition(sequence_obj, seqtype=seqtype)


class NucleotideAlignment(GenericAlignment):

    def __init__(self, input_obj, name='', description=''):
        super().__init__(input_obj, 'nucl', charsize=1, name=name, description=description)

    def basecomp(self):
        """
        Return the base composition of a NucleotideAlignment
        @return: namedtuple of 'A', 'T', 'G', 'C', 'AT', 'GC' percent
        """
        #assert re.search('^[ATCG\-]+$', sequence), 'Input sequence contains characters other than A,T,C,G,-'
        basecomp_of = super().composition(self, seqtype=self.seqtype).T
        basecomp_of['AT'] = basecomp_of['A'] + basecomp_of['T']
        basecomp_of['GC'] = basecomp_of['G'] + basecomp_of['C']
        return basecomp_of.T


class ProteinAlignment(GenericAlignment):

    def __init__(self, input_obj, name='', description=''):
        super().__init__(input_obj, 'prot', charsize=1, name=name, description=description)

    def aacomp(self):
        return super().composition(self, seqtype=self.seqtype)


class CodonAlignment(NucleotideAlignment):

    def __init__(self, input_obj, name='', description=''):
        GenericAlignment.__init__(self, input_obj, seqtype='cod', charsize=3, name=name, description=description)
        self.ntalignment = NucleotideAlignment(
            MSA(ids=self.ids, alignment=np.array([list(''.join(seq)) for seq in self.sequences], dtype='U1')))
        self.pos = dict()
        self.pos[1] = self.ntalignment.colx(0, None, 3)
        self.pos[2] = self.ntalignment.colx(1, None, 3)
        self.pos[3] = self.ntalignment.colx(2, None, 3)

    def make_raxml_codon_partition_file(self, save_path):
        """
        Make RAxML partition file for a codon alignment.

        @param save_path: Partition file save path
        """
        # TODO : check if basedir of save_path exists
        ordinal_suffix = {1: 'st', 2: 'nd', 3: 'rd', 4: 'th', 5: 'th', 6: 'th', 7: 'th', 8: 'th', 9: 'th', 0: 'th'}
        # TODO : Warn if file already exists
        with open(save_path, 'w') as f:
            for i in range(1, 4):
                print('DNA, {0}{1}pos={0}-{2}\\3'.format(i, ordinal_suffix[i], self.length*3), file=f)

    @staticmethod
    def composition(codon_aln_obj, fold_counts=(), pos=2):
        """
        Return the character composition of a CodonAlignment depending on codon position and fold count
        @param codon_aln_obj: CodonAlignment object
        @param fold_counts: 1,2,3,4,6 fold codon degeneracy
        @param pos: 0 (1st position), 1 (2nd position), 2 (3rd position)
        @return: namedtuple of percent makeup for each character except gaps
        """
        codon_filter_set = set([codon for codon, fold in CODON_FOLD.items() if fold in fold_counts])
        filtered_sequences = OrderedDict()
        for seqid, seq_as_list in zip(codon_aln_obj.ids, codon_aln_obj.sequences):
            if pos == -1:
                filtered_sequences[seqid] = ''.join([codon for codon in seq_as_list if codon in codon_filter_set])
            else:
                filtered_sequences[seqid] = ''.join([codon[pos] for codon in seq_as_list if codon in codon_filter_set])
        sequence_obj = Sequence(filtered_sequences)
        return Sequence.composition(sequence_obj, seqtype='cod')