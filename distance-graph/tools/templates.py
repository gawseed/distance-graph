from tools.templatizer import *
import pyfsdb
import tqdm

class Templates():
    def __init__(self, stopwords, file_args, temporal):
        self.stopwords = stopwords
        self.templates = []
        self.cmd2template = {}
        self.template2cmd = {}
        self.old_templates = []
        self.new_templates = []
        self.template_counts = {}
        self.cmd2template_count = {}

        self.templatize(file_args, temporal)
        self.build_cmd2template()
        self._init_template2cmd()
        self.find_new_old_templates(temporal)

    def templatize(self, file_args,temporal):
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
                        # if cmdCount2 == {}: ## second period of commands contains period 1
                        #     cmdCount2 = cmdCount.copy()
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
    
    def build_cmd2template(self):
        templates = []
        cmd2template2 = {}

        for cmd2template in self.templates:
            template2cmd = {}
            if cmd2template: ## cmd2template is not None
                for cmd,basename in cmd2template.items():
                    template = basename[2]
                    cmd2template2[cmd[2:-2]] = template
                    if template not in template2cmd:
                        template2cmd[template] = [cmd]
                    else:
                        template2cmd[template] = template2cmd[template] + [cmd]
                templates.append(template2cmd)
            else:
                templates.append({})
        
        self.templates = templates
        self.cmd2template = cmd2template2
    
    def find_new_old_templates(self, temporal):
        if temporal:
            templates1 = self.templates[0].keys()
            templates2 = self.templates[1].keys()
            new_templates = [template for template in templates2 if template not in templates1]
            old_templates = [template for template in templates1 if template not in templates2]

            self.new_templates = new_templates
            self.old_templates = old_templates
    
    def _init_template2cmd(self):
        templateDic = {}
        for template2cmd in self.templates:
            for template,cmds in template2cmd.items():
                if template not in templateDic:
                    templateDic[template] = cmds
                else:
                    templateDic[template] = templateDic[template] + cmds

        templateDic = {template:sorted(set(cmds)) for template,cmds in templateDic.items()}
        self.template2cmd = templateDic
    
    def calculate_template_counts(self, df, node_name, unique_cmds):
        for template,cmds in self.template2cmd.items():
            count = df[node_name].isin(cmds).sum()
            self.template_counts[template] = count
        
        unique_cmds = [cmd[2:-2] for cmd in unique_cmds]
        self.cmd2template_count = {cmd:self.template_counts[self.cmd2template[cmd]] for cmd in unique_cmds}