import os
# from mpi4py import MPI
import pandas as pd
import re
from copy import deepcopy
# TODO : Add proper docstrings

class Codeml(object):
    """
    Codeml object to generate control file and run the program
    """

    CODEML_CMD = 'codeml'

    def __init__(self, control_filename='codeml.ctl'):
        """
        Define the control filename to be used for writing down set parameters
        """
        self.control_filename = control_filename
        self.results = None

    def generate_control_file(self, seqfile, treefile, run_name='analysis01', control_filename='codeml.ctl',
                              seqtype='codon'):
        """
        Convenience method to set the most commonly used parameters.
        """
        self.control_filename = control_filename
        self.parameters = CodemlControlFile(seqfile, treefile=treefile, outfile=run_name, seqtype=seqtype)
        self.parameters.make(self.control_filename)

    def run(self):
        """
        Run codeml based on the generated control file
        """
        os.system('{0} {1}'.format(self.__class__.CODEML_CMD, self.control_filename))
        # After finishing
        self.results = CodemlOutput('')


class CodemlControlFile(object):
    """
    Codeml control file object to make write the control file
    """
    def __init__(self, seqfile, treefile=None, outfile='analysis01', seqtype='codon', partitioned=False):
        self.seqfile = seqfile
        self.outfile = outfile
        self.noisy = 9 #  0,1,2,3,9: how much rubbish on the screen
        self.verbose = 0 # 1: detailed output, 0: concise output

        # Tree
        if treefile:
            assert os.path.exists(treefile)
            self.treefile = treefile
            self.runmode = 0 # 0: user tree; 1: semi-automatic; 2: automatic 3: StepwiseAddition; (4,5):PerturbationNNI; -2: pairwise
            self.fix_blength = 0 # 0: ignore; 1: as initial values; 2: fixed
        else:
            self.runmode = -2 # 0: user tree; 1: semi-automatic; 2: automatic 3: StepwiseAddition; (4,5):PerturbationNNI; -2: pairwise

        # Sequence type
        assert seqtype in ['codon', 'aa', 'translate']
        # 1:codons; 2:AAs; 3:codons-->AAs
        if seqtype == 'codon':
            self.seqtype = 1
        elif seqtype == 'aa':
            self.seqtype = 2
            self.aaDist = 0 # 0:equal, +:geometric; -:linear, 1-6:G1974,Miyata,c,p,v,a
            #self.aaRatefile = ''
        elif seqtype == 'translate':
            self.seqtype = 3
            self.aaDist = 0 # 0:equal, +:geometric; -:linear, 1-6:G1974,Miyata,c,p,v,a
            #self.aaRatefile = ''
        else:
            raise Exception('Input value not valid: codon|aa|translate')
        self.CodonFreq = 2 # 0:1/61 each, 1:F1X4, 2:F3X4, 3:codon table
        self.clock = 0 #  0:no clock, 1:clock; 2:local clock

        # Partitioned analysis
        if partitioned:
            self.Mgene = 0
            self.Malpha = 0

        # Codon models
        # 0:one, 1:b, 2:2 or more dN/dS ratios for branches
        # AA models
        # 0:poisson, 1:proportional, 2:Empirical, 3:Empirical+F, 6:FromCodon, 8:REVaa_0, 9:REVaa(nr=189)
        self.model = 0
        self.icode = 0

        # Parameters
        # dN/dS ratio
        self.fix_omega = 0
        self.omega = 0.4

        # Transition/tranversion ratio
        self.fix_kappa = 0
        self.kappa = 0

        # Shape parameter of the gamma distribution for variable substitution rates across sites
        self.fix_alpha = 1
        self.alpha = 0
        self.ncatG = 0 # discrete gamma models

        # Independence of rates at adjacent sites where rho is the corelation parameter
        self.fix_rho = 1
        self.rho = 0

        #self.RateAncestor = 0
        #self.getSE = 0
        #self.Small_Diff = 0.5e-6
        self.cleandata = 1 # remove sites with ambiguity data (1:yes, 0:no)?
        self.method = 0 # 0: simultaneous; 1: one branch at a time (no applicable to clock models 1,2,3)


    def make(self, control_filename='codeml.ctl'):
        """
        Writes the parameters into a control text file
        Parameters
        ----------
        control_filename : str
            Specifies the filename to save as
        """
        self.__control_filename = control_filename
        with open(self.__control_filename, 'w') as f:
            for k,v in self.__dict__.items():
                if not k.startswith('_'):
                    print(k, '=', v, file=f)


class CodemlOutput(object):
    """
    Parses Codeml output
    """
    def __init__(self, output_filename, filedir='.'):
        self.__root_dir = filedir
        self.__codeml_output = output_filename
        self.dN_NG= CodemlOutput.read_codeml_pairwise(os.path.join(self.__root_dir, '2NG.dN'))
        self.dS_NG = CodemlOutput.read_codeml_pairwise(os.path.join(self.__root_dir, '2NG.dS'))
        self.omega_NG = CodemlOutput.matrix_omega(self.dN_NG, self.dS_NG)

        self.tree = CodemlOutput.get_base_tree(os.path.join(self.__root_dir, self.__codeml_output))
        self.tree_substitutions = CodemlOutput.get_codon_subs_tree(os.path.join(self.__root_dir, self.__codeml_output))
        self.tree_dN_dS = CodemlOutput.get_dN_dS_table(os.path.join(self.__root_dir, self.__codeml_output))
        self.tree_dN = CodemlOutput.transform_to_labeled_tree(self.tree, self.tree_substitutions, self.tree_dN_dS['dN'])
        self.tree_dS = CodemlOutput.transform_to_labeled_tree(self.tree, self.tree_substitutions, self.tree_dN_dS['dS'])


    @staticmethod
    def read_codeml_pairwise(filename):
        # Read from text file
        with open(os.path.join(filename), 'r') as f:
            sample_ids = []
            pairwise = {}
            for i,line in enumerate(f.readlines()):
                if i == 0:
                    num_sequences = re.search('\d+', line).group(0)
                else:
                    columns = re.split('\s+', line)
                    current_sample_id = columns[0]
                    sample_ids.append(current_sample_id)
                    pairwise[(current_sample_id, current_sample_id)] = 0
                    if i != 1:
                        for value, sample_id in zip(columns[1:], sample_ids[:-1]):
                            pairwise[(current_sample_id, sample_id)] = float(value)
        # Convert into a dataframe
        return CodemlOutput.convert_pairwise_to_dataframe(pairwise, sample_ids)


    @staticmethod
    def read_codeml_pairwise_tuples(filename):
        # Read from text file
        with open(os.path.join(filename), 'r') as f:
            sample_ids = []
            pairwise = {}
            for i,line in enumerate(f.readlines()):
                if i == 0:
                    num_sequences = re.search('\d+', line).group(0)
                else:
                    columns = re.split('\s+', line)
                    current_sample_id = columns[0]
                    sample_ids.append(current_sample_id)
                    pairwise[(current_sample_id, current_sample_id)] = 0
                    if i != 1:
                        for value, sample_id in zip(columns[1:], sample_ids[:-1]):
                            pairwise[(current_sample_id, sample_id)] = float(value)
        # Convert into a dataframe
        return pairwise

    @staticmethod
    def convert_pairwise_to_dataframe(pairwise, sample_ids):
        series_dict = {}
        for sample_id in sample_ids:
            series_dict[sample_id] = pd.Series({k[-1] if k[0]==sample_id else k[0]: v for k,v in pairwise.items() if sample_id in k})
        return pd.DataFrame(series_dict)

    @staticmethod
    def compute_omega(dN, dS):
        if (dN == 0) and (dS == 0):
            return 0
        elif dS == 0:
            return float('inf')
        else:
            return dN/float(dS)

    @staticmethod
    def matrix_omega(dN, dS):
        assert dN.shape == dS.shape
        pairwise_omega_dict = {}
        for i in range(0, dN.shape[0]):
            for j in range(0, dN.shape[1]):
                pairwise_omega_dict[(dN.index[i], dN.index[j])] = CodemlOutput.compute_omega(dN.ix[i,j], dS.ix[i,j])
        return CodemlOutput.convert_pairwise_to_dataframe(pairwise_omega_dict, list(dN.index))

    @staticmethod
    def get_codon_subs_tree(codeml_output):
        """
        Retrieve the name-labeled tree with the number of nucleotide substitutions per codon from the codeml
        output file

        Codeml outputs the substitution per code tree somewhere after the number-labeled tree in the output file.
        To find this tree, this method looks for the first line which starts with "tree ". From there, the name-labeled
        is on the 4th line past the "starts with 'tree ' line".

        This entire line contains the Newick-formatted tree so there is no need to perform regular expression matching.

        Parameters
        ----------
        codeml_output : str, filepath

        Returns
        -------
        str
            Newick tree with named nodes
        """
        codon_subs_tree = ''
        with open(codeml_output, 'r') as f:
            tree_flag = False
            to_tree_line_counter = 3
            for i,line in enumerate(f.readlines()):
                if tree_flag:
                    if to_tree_line_counter > 0:
                        to_tree_line_counter -= 1
                        continue
                    else:
                        codon_subs_tree = line[:-1]
                        tree_flag = False
                        break
                else:
                    if line.startswith('tree length =') and (to_tree_line_counter > 0):
                        tree_flag = True
        return codon_subs_tree

    @staticmethod
    def get_base_tree(codeml_output):
        """
        Retrieves the numbered tree codeml from the codeml output file.

        Codeml outputs the base tree with the label "TREE # 1". This method looks for the first line with that starts
        with "TREE" and grabs the substring with the pattern "\(.+\)\;".

        The base tree is a Newick-formatted tree that uses numbers for extant node labels.
        The first node (1) is based on the first entry on a the multiple sequence alignment. The numbering of the
        subsequent extant nodes is also based on how they are listed in the multiple sequence alignment.
        From node 1, codeml traverses the tree up until it reached the trifurcation and numbers internal nodes as
        it encounters them. From there, it traverses the tree deep-first and numbers both internal and external nodes
        as it encounters them.

        Parameters
        ----------
        codeml_output : str, filepath

        Returns
        -------
        str
            Newick tree with numbered nodes
        """
        base_tree = ''
        with open(codeml_output, 'r') as f:
            for i,line in enumerate(f.readlines()):
                if line.startswith('TREE'):
                    base_tree = re.search('\(.+\)\;',line).group(0)
        return base_tree.replace(' ', '')

    @staticmethod
    def transform_to_template_tree(base_tree):
        """
        Inserts branch signposts unto the base tree

        Parameters:
        base_tree : str

        Returns
        -------
        str
            Newick-formatted name-labelled tree
        """
        sample_num_list = list(map(int, re.findall('\d+', base_tree)))
        counter = max(sample_num_list)

        num_flag = False
        current_number = ''
        stack = []
        stack_num = []
        labeled_tree = ''
        for char in base_tree:
            if num_flag:
                if char in ['1','2','3','4','5','6','7','8','9','0']:
                    current_number += char
                elif char == ')':
                    if (stack[-1], ')') == ('(',')'):
                        close_branch_number = stack_num.pop()
                        labeled_tree += '{1}:{0}..{1}'.format(close_branch_number, current_number)
                        labeled_tree += char

                        stack.pop()
                        if len(stack) > 0:
                            labeled_tree += ':{0}..{1}'.format(stack_num[-1], close_branch_number)
                        num_flag = False
                    else:
                        raise Exception()
                elif char == ',':
                    labeled_tree += '{1}:{0}..{1}'.format(stack_num[-1], current_number)
                    labeled_tree += char
                    num_flag = False
            else:
                if char in ['1','2','3','4','5','6','7','8','9','0']:
                    current_number = char
                    num_flag = True
                elif char == '(':
                    stack.append(char)
                    counter += 1
                    stack_num.append(counter)
                    labeled_tree += char
                elif char == ')':
                    labeled_tree += char
                    if (stack[-1], ')') == ('(',')'):
                        stack.pop()
                        close_branch_number = stack_num.pop()
                        labeled_tree += ':{0}..{1}'.format(stack_num[-1], close_branch_number)
                    else:
                        raise Exception()
                else:
                    labeled_tree += char
        return labeled_tree

    @staticmethod
    def transform_to_labeled_tree(base_tree, name_tree, label_value_dct):
        """
        Replaces the numbers in the number-labeled base tree with names from the name tree, and then
        adds branch length data

        Parameters
        ----------
        base_tree : str, Newick format
        name_tree : str, Newick format
        label_value_dict : list of key-value pairs, can be a Series or a dictionary

        Return
        ------
        str
            Newick formatted labeled tree
        """
        template_tree = CodemlOutput.transform_to_template_tree(base_tree)
        new_tree = deepcopy(template_tree)
        for k,v in label_value_dct.items():
            k = k.replace('.','\.')
            new_tree = re.sub("(?<=\D){0}(?=\D)".format(k), v, new_tree)

        offset = 0
        labeled_tree = deepcopy(new_tree)
        for start, end, name in [(s.start(), s.end(), k) for s,k in zip(re.finditer('\d+(?=\:)',new_tree),
                                                                        re.findall('([A-Za-z0-9]+)\:', name_tree))]:
            offset_start = start + offset
            offset_end = end + offset
            labeled_tree = labeled_tree[:offset_start] + name + labeled_tree[offset_end:]
            offset += len(name) - (offset_end - offset_start)
        return labeled_tree

    @staticmethod
    def get_dN_dS_table(codeml_output):
        with open(codeml_output, 'r') as f:
            line_countdown = 3
            table_flag = False
            table_dict = {}
            for i,line in enumerate(f.readlines()):
                if table_flag:
                    if line_countdown > 0:
                        line_countdown -= 1
                    else:
                        if re.search('^\s\s\d', line):
                            cols = re.split('\s+', line[2:-1])
                            table_dict[cols[0]] = cols[1:]
                        else:
                            table_flag = False
                else:
                    if line.startswith('dN & dS') and line_countdown == 3:
                        table_flag = True
                        continue

        df = pd.DataFrame(table_dict).T
        df.columns = ['t', 'N', 'S', 'dN/dS', 'dN', 'dS', 'NdN', 'SdS']
        return df


def execute_codeml(seqfile, treefile):
    analysis = Codeml()
    analysis_number = int(re.search('(\d+)\.list$', seqfile).group(0))
    analysis.generate_control_file(seqfile, treefile, run_name='{:0>4d}'.format(analysis_number))
    analysis.run()


def array_execute_codeml(list_of_alignments, list_of_treefiles):
    assert len(list_of_alignments) == len(list_of_treefiles)
    for aln, tree in zip(list_of_alignments, list_of_treefiles):
        print(aln, tree, end='... ')
        basename_aln = os.path.basename(aln)
        basename_tree = os.path.basename(tree)
        working_dir = os.path.dirname(aln)
        # Change working dir
        last_pwd = os.getcwd()
        os.chdir(working_dir)
        execute_codeml(basename_aln, basename_tree)
        os.chdir(last_pwd)
        print('Done.')


# def parallel(list_of_alignments, list_of_treefiles):
#     assert len(list_of_alignments) == len(list_of_treefiles)
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#     num_files = len(list_of_alignments) / size
#     list_of_alignments_per_node = [list_of_alignments[x] for x in range(rank, num_files, size)]
#     list_of_treefiles_per_node = [list_of_treefiles[x] for x in range(rank, num_files, size)]
#     # Execute
#     array_execute_codeml(list_of_alignments_per_node, list_of_treefiles_per_node)

