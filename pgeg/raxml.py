import os
from collections import namedtuple
from random import randint

RAXML_PATH = 'raxmlHPC-PTHREADS-AVX'
MODELS = ['GTRGAMMA', 'GTRGAMMAI', 'GTRCAT', 'GTRCATI']
RAXML_INFO_PREFIX = 'RAxML_info'
RAXML_LOG_PREFIX = 'RAxML_log'
RAXML_RESULT_PREFIX = 'RAxML_result'
RAXML_BESTTREE_PREFIX = 'RAxML_bestTree'
RAXML_PARSTREE_PREFIX = 'RAxML_parsimonyTree'
RAXML_BSTREE_PREFIX = 'RAxML_bootstrap'
RAXML_BP_PREFIX = 'RAxML.bipartitions'
RAXML_BPBRANCH_PREFIX = 'RAxML_bipartitionsBranchLabels'

Parameter = namedtuple('Parameter', 'argument value')

def add_file_argument(argument, file_path, error_message='File {0} does not exist.', as_str=True):
    if os.path.isfile(file_path):
        parameter = Parameter(argument=argument, value=file_path)
    else:
        raise FileNotFoundError(error_message.format(file_path))
    return '{argument} {value}'.format_map(parameter._asdict()) if as_str else parameter


def add_dir_argument(argument, dir_path, error_message='Directory {0} does not exist.', as_str=True,
                     make_absolute=True):
    if os.path.isdir(dir_path):
        dir_path = os.path.abspath(dir_path) if make_absolute else dir_path
        parameter = Parameter(argument=argument, value=dir_path)
    else:
        raise NotADirectoryError(error_message.format(dir_path))
    return '{argument} {value}'.format_map(parameter._asdict()) if as_str else parameter


def std_options_statement(cls, partition_file_path=None, output_path=None,
                              model='GTRGAMMA', suffix: str='tree', threads: int=1):
        cmd_lst = list()
        cmd_lst.append('-n {suffix}'.format(suffix=suffix))

        if partition_file_path:
            cmd_lst.append(add_file_argument('-q', partition_file_path,
                                             error_message='Partition file {0} does not exist.'))

        if output_path:
            cmd_lst.append(add_dir_argument('-w', output_path))

        if model in cls.MODELS:
            cmd_lst.append('-m {model}'.format(model=model))
        else:
            raise ValueError('"{0}" is not a valid model.'.format(model))

        if 'PTHREADS' in cls.RAXML_PATH:
            cmd_lst.append('-T {threads}'.format(threads=os.cpu_count() if threads == -1 else threads))
        else:
            if threads:
                raise Warning('RAxML configured for this task is not multithreaded.')
        return cmd_lst


def make_parsimony_tree(cls, alignment_path, partition_file_path=None, output_path=None,
                            model='GTRGAMMA', suffix: str='pars_tree', seed: int=None, threads: int=1):
        """
        Run RAxML to make a parsimony tree.

        @param alignment_path: Path to alignment file in PHYLIP or FASTA formats
        @param partition_file_path: Path to partition file in RAxML format
        @param output_path: Directory where RAxML will write files
        @param model: Model of Nucleotide substitution
        Currently implemented are
            - GTRGAMMA/I
            - GTRCAT/I
        @param suffix: File name suffix for all generated files
        @param seed: Random seed for parsimony inference
        @param threads: Number of threads for RAxML PTHREADS
        @return: Dictionary of output files and corresponding filenames
        """
        cmd_lst = cls.std_options_statement(partition_file_path=partition_file_path, output_path=output_path,
                                            model=model, suffix=suffix, threads=threads)
        cmd_lst.insert(0, '{raxml} -y'.format(raxml=cls.RAXML_PATH))
        cmd_lst.append('-p {seed}'.format(seed=seed if seed else randint(0, 999999)))
        cmd_lst.append(add_file_argument('-s', alignment_path, error_message='Alignment file {0} does not exist.'))

        # print(' '.join(cmd_lst))
        os.system(' '.join(cmd_lst))
        outfiles =  {'info': '{info}.{suffix}'.format(info=cls.RAXML_INFO_PREFIX, suffix=suffix),
                     'tree': '{prefix}.{suffix}'.format(prefix=cls.RAXML_PARSTREE_PREFIX, suffix=suffix)}
        abs_output_path = os.path.abspath(output_path) if output_path else os.path.abspath('.')
        return {k: os.path.join(abs_output_path, filename) for k, filename in outfiles.items()
                if os.path.isfile(os.path.join(abs_output_path, filename))}

def integrate_bootstrap_trees(cls, bootstrap_file_path, given_tree_file_path, partition_file_path=None,
                                  output_path=None, model='GTRGAMMA', suffix: str='bp_tree', threads: int=1):
        """
        Draw bipartitions on a given tree based a set of multple trees (bootstrap)

        @param bootstrap_file_path: Path to multiple trees file in Newick format.
        @param given_tree_file_path: Path to base tree in Newick format.
        @param partition_file_path: Path to partition file in RAxML format.
        If None, RAxML will perform an unpartitioned analysis.
        @param output_path: Directory where RAxML will write files
        @param model: Model of Nucleotide substitution
        Currently implemented are
            - GTRGAMMA/I
            - GTRCAT/I
        @param suffix: File name suffix for all generated files
        @param threads: Number of threads for RAxML PTHREADS
        @return: Dictionary of output files and corresponding filenames
        """
        cmd_lst = cls.std_options_statement(partition_file_path=partition_file_path, output_path=output_path,
                                            model=model, suffix=suffix, threads=threads)
        cmd_lst.insert(0, '{raxml} -f b'.format(raxml=cls.RAXML_PATH))
        cmd_lst.append(add_file_argument('-z', bootstrap_file_path, error_message='Bootstrap file {0} does not exist.'))
        cmd_lst.append(add_file_argument('-t', given_tree_file_path,
                                         error_message='Best tree file {0} does not exist.'))
        # print(' '.join(cmd_lst))
        os.system(' '.join(cmd_lst))
        outfiles = {'info': '{info}.{suffix}'.format(info=cls.RAXML_INFO_PREFIX, suffix=suffix),
                    'bptree': '{prefix}.{suffix}'.format(prefix=cls.RAXML_BSTREE_PREFIX, suffix=suffix),
                    'bpbranchtree': '{prefix}.{suffix}'.format(prefix=cls.RAXML_BPBRANCH_PREFIX, suffix=suffix)}
        abs_output_path = os.path.abspath(output_path) if output_path else os.path.abspath('.')
        return {k: os.path.join(abs_output_path, filename) for k, filename in outfiles.items()
                if os.path.isfile(os.path.join(abs_output_path, filename))}
