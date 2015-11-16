import os
import re
import pandas as pd
from collections import OrderedDict, namedtuple
from scipy.spatial.distance import hamming
from bioseq import GENETIC_CODE

Codon_change = namedtuple('Codon_change', ['anc', 'der'])


def generate_one_hit_codon_table():
    """Creates a lookup table for one-hit codon changes using a numbered index.

    Returns
    -------
    OrderedDict

    """
    _one_hit_codon = OrderedDict()
    for i in range(1, 65):
        for j in range(1, 65):
            anc = GENETIC_CODE.by_index(i)[0]
            der = GENETIC_CODE.by_index(j)[0]
            dist = hamming(list(anc), list(der))
            if dist <= 1/float(3):
                _one_hit_codon[(i,j)] = Codon_change(GENETIC_CODE.by_index(i), GENETIC_CODE.by_index(j))
    return _one_hit_codon

one_hit_codon = generate_one_hit_codon_table()


def one_hit_codon_by_index(x):
    return list(one_hit_codon.items())[x - 1]  # input is 1-indexed but python is 0-indexed


def read_hmdata(path):
    """Parses HM_summary_summ text file into a pandas DataFrame

    Parameters
    ----------
    path : path to HM_summary_summ

    Returns
    -------
    DataFrame
        Representation of data as a DataFrame. Row represent each lineage and columns show values for each type of
        substitution.

    Notes
    -----
    Header abbreviations:
    s1 : GC->AT, synonymous
    s2 : AT->GC, synonymous
    s3 : AT->AT, synonymous
    s4 : GC->GC, synonymous
    r1 : GC->AT, non-synonymous (replacement)
    r2 : AT->GC, non-synonymous (replacement)
    r3 : AT->AT, non-synonymous (replacement)
    r4 : GC->GC, non-synonymous (replacement)
    """
    df = pd.read_table(path, sep='\t', header=0, skiprows=7, index_col=0, usecols=[i for i in range(10)])
    df.drop('method', axis=1, inplace=True)
    return df


def make_hmdata_bootstrap_panel(path_list, hm_summary_path='HM_summary/HM_summary'):
    """Create a Panel of HM_summary_summ data taken from a bootstrap dataset

    If the current bootstap path is `result_bin_0` and the hm_summary_path is `HM_summary/HM_summary`:
    the complete path to the HM_summary folder will be `result_bin_0/HM_summary/HM_summary`.

    Parameters
    ----------
    path_list : list
        List of folder paths of the bootstrapped dataset - can be absolute or relative. For example "result_bin_0"
    hm_summary_path : path
        Relative path to HM_summary where the root is a bootstrap folder

    Returns
    -------
    Panel
        The pandas Panel is composed of lineages in the first axis, types of substitutions in the second axis, and the
        bootstrap replicate in the third axis.

    """
    df_array = OrderedDict()
    for bootstrap_folder in path_list:
        current_path = os.path.join(bootstrap_folder, hm_summary_path)
        HM_grand_summary_path = os.path.join(current_path, 'HM_summary_summ')
        i = int(bootstrap_folder.split('_')[-1])  # 0-index is original data, >1 are bootstraps
        df_array[i] = read_hmdata(HM_grand_summary_path)
    return pd.Panel(df_array)


def parse_hm01s_line(line):
    """Create a dataframe for a single lineage based on a single line from the hm01s result file

    Parameters
    ----------
    line : str
        Should be a valid result line from the HM01s file. Comments, headers, and other text should not be passed to
        this function

    Returns
    -------
    DataFrame

    """
    codon_result = namedtuple('codon_result', ['anc_i', 'anc_cod', 'anc_aa', 'der_i', 'der_cod', 'der_aa', 'hm'])
    current_result = dict()
    for a, b in [string.split('\t') for string in re.findall('\d+\t\d+\.\d+', line)]:
        indexes, codon_states = one_hit_codon_by_index(int(a))
        hm_prob = float(b)
        result_index = list(one_hit_codon.keys()).index(indexes) + 1
        current_result[result_index] = codon_result(
            indexes[0], codon_states.anc[0], codon_states.anc[-1],
            indexes[1], codon_states.der[0], codon_states.der[-1], hm_prob)

    # Fill in empty indexes with 0 and make df
    for i in range(1, 641):
        if i not in current_result.keys():
            indexes, codon_states = one_hit_codon_by_index(i)
            current_result[i] = codon_result(
                            indexes[0], codon_states.anc[0], codon_states.anc[-1],
                            indexes[1], codon_states.der[0], codon_states.der[-1], 0)
    df = pd.DataFrame(current_result).T
    df.columns = codon_result._fields
    return df


def parse_hm01s_file(path, lineage_lookup=None):
    """Creates a Panel for all the codon changes inferred for all lineages in the analysis

    Parameters
    ----------
    path : str or path
        Location of the HM01s file. Usually found in `result_bin_x/8_alncds_HM/alncds_model1_HM01s`
    lineage_lookup : dict-like
        Lookup table for lineage names. HM01s is an ordered list indexed from 0 so lineage_lookup must contain a
        dictionary whose keys correspond to order in HM01s and values are the actual names for those lineages,
        for example "14..1" for the branch between nodes 14 and 1.

    Returns
    -------
    Panel

    """
    results_dict = dict()
    lin_cnt = 0
    with open(path, 'r') as f:
        for line in f.readlines():
            if line.startswith('/*'):
                continue
            else:
                if re.search('^[A-Za-z0-9]+\n$', line):
                    pass  # header lines, can be used for multi-result parsing
                else:
                    assert lin_cnt in lineage_lookup.keys(), KeyError()
                    i = lineage_lookup[lin_cnt] if lineage_lookup else lin_cnt
                    results_dict[i] = parse_hm01s_line(line)
                    lin_cnt += 1
    return pd.Panel(results_dict)