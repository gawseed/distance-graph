from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType

class Arguments():
    def __init__(self):
        self.args = self.parse_args()
        self.node_name = ''
        self.label_name = ''
        self.id_name = ''
        self.output_names = self.args.output_names
        
        self.__init_file_args()
        self.get_outputNames()
    
    def __init_file_args(self):
        self.file_args = list(filter(None,[self.args.input_file1, self.args.input_file2, self.args.input_file3, self.args.input_file4]))
        self.check_fileArgs(self.file_args, self.output_names)
        
    def parse_args(self):
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                                description=__doc__,
                                epilog='Example Usage: python distance-graph.py -1 ../example/data1.fsdb command data1 ip -2 ../example/data2.fsdb commands data2 ip -3 ../example/data3.fsdb cmd data3 id -on command source ip --templatize -stop ../example/nix_commands.txt -e 0.05 -w 10 -H 8 -fs 10 -ns 250 -E edgelist.fsdb -c clusterlist.fsdb distance-graph.png')

        parser.add_argument("-1", "--input_file1", type=str, nargs=4,
                            required=True, help="Pass [required] input filename, [required] node column name (e.g. command), label name of the data (e.g. data1), node's identifier column name (e.g. ip). Pass empty string '' if do not have label name or identifier column.")

        parser.add_argument("-2", "--input_file2", type=str, default=None, nargs=4,
                            help="Pass second [required] input filename, [required] node column name (e.g. command), label name of the data (e.g. data2), node's identifier column name (e.g. ip). Pass empty string '' if do not have label name or identifier column.")

        parser.add_argument("-3", "--input_file3", type=str, default=None, nargs=4,
                            help="Pass third [required] input filename, [required] node column name (e.g. command), label name of the data (e.g. data3), node's identifier column name (e.g. ip). Pass empty string '' if do not have label name or identifier column.")

        parser.add_argument("-4", "--input_file4", type=str, default=None, nargs=4,
                            help="Pass fourth [required] input filename, [required] node column name (e.g. command), label name of the data (e.g. data3), node's identifier column name (e.g. ip). Pass empty string '' if do not have label name or identifier column.")

        parser.add_argument("-on", "--output_names", type=str, nargs=3,
                            required=True, help="The names to use for output distance graph labels, edge list, and cluster list. Pass the [required] node name (e.g. command), column name for label (e.g. source), identifier name (e.g. ip). Pass empty string '' if input files do not have label name or identifier column.")

        parser.add_argument("--templatize", default=False, action="store_true",
                            help="Set this argument to templatize the nodes")

        parser.add_argument("--template-nodes", default=False, action="store_true",
                        help="Set this argument to graph template nodes only")

        parser.add_argument("-stop", "--stopwords", default=None, type=str,
                            help="Path to text file that contains stopwords separated by a new line")

        parser.add_argument("--temporal", type=int, default=None, nargs='+',
                            help="Pass the input file number(s) with 'new' data separated by a space to perform temporal analysis")

        parser.add_argument("-e", "--edge-weight", default=.95, type=float,
                            help="The edge weight threshold to use")

        parser.add_argument("-k", "--top-k", default=None, type=int,
                            help="Top k nodes to graph. If this is set, the edge weight threshold will not be used.")

        parser.add_argument("--top-k-edges", default=False, action="store_true",
                        help="Set this argument to graph top k edges for each node")

        parser.add_argument("-p", "--position", default=None, type=str,
                            help="Path to pickle file that contains NetworkX graph positional dictionary")

        parser.add_argument("-l", "--labels", default=None, type=str,
                            help="Path to pickle file that contains dictionary of node labels")

        parser.add_argument("-E", "--edge-list", default=None, type=str,
                            help="Output enumerated edge list to here")

        parser.add_argument("-c", "--cluster-list", default=None, type=str,
                            help="Output enumerated cluster list to here")

        parser.add_argument("-t", "--template-list", default=None, type=str,
                        help="Output list of templates to here")

        parser.add_argument("-tc", "--templatecmd-list", default=None, type=str,
                        help="Output list of templates and templatized commands to here")

        parser.add_argument("-pf", "--position-file", default=None, type=str,
                            help="Output pickle file that contains generated NetworkX graph positional dictionary to here")

        parser.add_argument("-lf", "--labels-file", default=None, type=str,
                            help="Output pickle file that contains dictionary of node labels to here")

        parser.add_argument("-w", "--width", default=12, type=float,
                            help="The width of the plot in inches")

        parser.add_argument("-H", "--height", default=8, type=float,
                            help="The height of the plot in inches")

        parser.add_argument("-fs", "--font-size", default=10, type=int,
                            help="The font size of the node labels") 

        parser.add_argument("-ns", "--node-size", default=300, type=int,
                            help="The node size for each node")

        parser.add_argument("output_file", type=str, nargs=1, 
                            help="Pass path and filename for output network graph")

        args = parser.parse_args()

        return args

    def check_fileArgs(self, file_args, output_names):
        """ Checks input file arguments with output name arguments to make sure correct arguments are provided. If not, raise exception
        Input:
            file_args (list) - list of lists (each list is a list of arguments for input file)
            output_names (list) - list of names to use for output distance graph and edge and cluster lists
        """
        output_node = output_names[0]
        output_source = output_names[1]
        output_id = output_names[2]

        if output_node == '':
            raise Exception("Missing name for output node")

        for i in range(len(file_args)):
            file_num = i+1
            inputfile_args = file_args[i]

            inputfile = inputfile_args[0]
            input_node = inputfile_args[1]
            input_source = inputfile_args[2]
            input_id = inputfile_args[3]

            if inputfile == '':
                raise Exception("Missing input file for input file {}".format(file_num))
            
            if input_node == '':
                raise Exception("Missing node column name for input file {}".format(file_num))
            
            if (output_source != '') and (input_source == ''):
                raise Exception("Source column name for input file {} not provided, but provided in output names.".format(file_num))
            
            if (output_source == '') and (input_source != ''):
                raise Exception("Source column name for input file {} is provided, but not provided in output names.".format(file_num))

            if (output_id != '') and (input_id == ''):
                raise Exception("Identifier column name for input file {} not provided, but provided in output names.".format(file_num))
            
            if (output_id == '') and (input_id != ''):
                raise Exception("Identifier column name for input file {} is provided, but not provided in output names.".format(file_num))

    def get_outputNames(self):
        """ Parse each output name argument and return node name, label name, and identifier name to be used for final output and graph
        Input:
            output_names (list) - list of output names to use for node, label, and identifier
        Output:
            node_name, label_name, identifier_name (str) - return node, label, and identifier for output
        """
        self.node_name = self.output_names[0]
        self.label_name = self.output_names[1]
        self.id_name = self.output_names[2]
        # return node_name, label_name, identifier_name
    

class FileArguments():
    def __init__(self, file_args):
        self.filename = file_args[0]
        self.node_name = file_args[1]
        self.label_name = file_args[2]
        self.id_name = file_args[3]
        # self.map_output_names()

    def map_output_names(self, file_args, output_names):
        """ Maps input file column name to output name. Returns dictionary with 
        Input:
            file_args (list) - list of arguments user provided for input file
            output_names (list) - list of output names to use for node, label, and identifier
        Output:
            mapNameDic (dict) - key: input column name (str) / value: output column name (str)
        """
        mapNameDic = {}
        input_names = file_args[1:]
        if input_names[1] != '':
            input_names[1] = 'label'

        for i in range(len(input_names)):
            if output_names[i] != '':
                mapNameDic[input_names[i]] = output_names[i]

        return mapNameDic