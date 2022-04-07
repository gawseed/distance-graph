from tools.templatizer import *
import pyfsdb
import tqdm

class Templates():
    def __init__(self, stopwords, file_args, temporal):
        self.stopwords = stopwords
        self.get_cmd2template(file_args, temporal)

    def get_cmd2template(self, file_args,temporal):
        """ Given data file with commands, return list of command templates dict. If performing temporal analysis, then get templates for period 1 and period 2
        Input:
            file_args (list) - list of input FSDB files
            temporal (None/list) - list of file numbers (period 2) for temporal analysis
        Output:
            cmd2template (list) - list of two cmd2template dicts. cmd2template - maps templatizable commands to highest degree template
        """
        cmds,cmds2 = self.get_commandCounts(file_args,temporal)
        # cmds,cmds2 = get_commandCounts2(file_args,temporal)
        cmd2template = self.templatize_cmds(cmds)

        if temporal:
            print("Finding period 2 templates")
            cmd2template2 = self.templatize_cmds(cmds2)
        else:
            cmd2template2 = {}

        self.templates = [cmd2template,cmd2template2]

    def templatize_cmds(self, cmds):
        cmd_graph = CommandGraph()
        for cmd in tqdm.tqdm(cmds.keys()):
        #  print("==== CMD IS====\n%s" % cmd) 
            cmd_graph.add(CommandNodeEval(cmd, self.stopwords))

        cmd_graph.finalize_degrees()
        sorted_by_degree = sorted(cmd_graph.template2degree.items(), key=lambda k: -k[1])
        cmd2template = cmd_graph.cmd_to_template()

        # print("Got templates. Done.")
        return cmd2template

    def get_commandCounts(self, file_args, temporal):
        """ Counts number of commands run in the dataset and returns dict with command and respective counts
        Input:
            file_args (list) - list of file argument lists with format [filename, node name, label name, identifier name]
        Output:
            cmdCount (dict) - maps command to number of times the cmd appears in the data
        """
        cmdCount = {}
        cmdCount2 = {}

        file_num = 1
        for input_file in file_args:
            filename = input_file[0]
            node_name = input_file[1]

            db = pyfsdb.Fsdb(filename)
            node_index = db.get_column_number(node_name)

            for row in db:
                node = row[node_index]
                
                if node[0] != "[":
                    node = str([node])

                if temporal:
                    if file_num in temporal: ## if file is in period 2, add nodes to cmdCount2
                        if node not in cmdCount2:
                            cmdCount2[node] = 1
                        else:
                            cmdCount2[node] += 1
                    else:
                        if node not in cmdCount:
                            cmdCount[node] = 1
                        else:
                            cmdCount[node] += 1
                else:
                    if node not in cmdCount:
                        cmdCount[node] = 1
                    else:
                        cmdCount[node] += 1
            
            db.close()
            file_num += 1

        return cmdCount,cmdCount2