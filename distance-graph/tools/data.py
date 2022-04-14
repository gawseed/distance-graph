import pandas as pd
import pyfsdb
from tools.arguments import FileArguments

class Data():
    def __init__(self, args):
        self.df = pd.DataFrame()
        self.loggedInOnly = []
        self.unique_cmds = []
        self.cmdIPsDic = {}
        self.labelDic = {}
        self.sourceDic = {}
        self.weightDic = {}

        self.__init_dataframe(args)
        self.find_unique_commands(args)
    
    def __init_dataframe(self, args):
        df = pd.DataFrame()

        for file_arg in args.file_args:
            inputfile_args = FileArguments(file_arg)
            map_input2output_names = inputfile_args.map_output_names(file_arg, args.output_names)

            db = pyfsdb.Fsdb(inputfile_args.filename)
            data = db.get_pandas(data_has_comment_chars=True)

            if inputfile_args.label_name != '':
                data['label'] = inputfile_args.label_name

            data = data.rename(columns=map_input2output_names)
            df = pd.concat([df,data]).reset_index(drop=True)
            db.close()
        
        df[args.node_name] = df[args.node_name].apply(lambda x: str([x]) if x[0]!="[" else x)
        df = self.remove_logged_in_only(df, args)
        df = df[df[args.node_name]!='[]'] ## remove rows where no commands run
        self.df = df

    def remove_logged_in_only(self, df, args):
        if args.id_name != '':
            loggedIn = df[df[args.node_name]=='[]'][args.id_name].unique()
            loggedInOnly = []
            labels = df[args.label].unique()

            for ip in loggedIn:
                cmdsRun = []
                for label_name in labels:
                    cmdsRun = cmdsRun + list(df[(df[args.id_name]==ip) & (df[args.label]==label_name)][args.node_name].unique())
                if cmdsRun == ['[]']:
                    loggedInOnly.append(ip)
            
            self.loggedInOnly = loggedInOnly
            df = df.copy()[~df[args.id_name].isin(loggedInOnly)]
            return df
        else:
            return df
    
    def find_unique_commands(self, args):
        self.unique_cmds = list(self.df[args.node_name].unique())

    def get_cmdIPsDic(self, args, loggedInOnly,id_name,login_index,temporal):
        """ Returns dict that contains IP addresses that ran the command and from what source
        Input:
            file_args (list) - list of file argument lists with format [filename, node name, label name, identifier name]
            loggedInOnly (list) - list of IPs that only logged in
            id_name (str) - column name of identifier column (eg. "ip")
            login_index (False/int) - index of login successful
            temporal (None/list) - list of file numbers (period 2) for temporal analysis
        Output:
            cmdIPsDic (dict) - key: command (str) / value: dictionary with key: source (str) & value: IPs that ran command (list)
        """
        cmdIPsDic = {}
        labelIPDic = {}

        seenNodes = []

        file_num = 1
        for file_arg in args.file_args:
            inputfile_args = FileArguments(file_arg)
            # filename, input_node_name, input_label_name, input_id_name = get_inputFile_args(inputfile_args)
            db = pyfsdb.Fsdb(inputfile_args.filename)
            seenNodes = list(set(seenNodes))

            id_index = db.get_column_number(inputfile_args.id_name)
            node_index = db.get_column_number(inputfile_args.node_name)

            for row in db:
                ident = row[id_index] ## identifier (IP address)
                
                if ident in loggedInOnly: ## if IP only logged in, do not record
                    continue

                ## check if login_index is provided. Skip over data where login_successful is false
                if (login_index != False) and (row[login_index] == 'False'):
                    continue
                
                node = row[node_index]
                label = inputfile_args.label_name
                ident_label = label

                if temporal: ## if doing temporal analysis
                    if (file_num in temporal) and (node not in seenNodes): ## if looking at period 2 input file and node not seen
                        label = "new_"+label
                    elif node in seenNodes: ## if node has been seen, continue
                        continue
                    else:
                        seenNodes.append(node)
                
                if node[0]!="[":
                    node = str([node])
                
                if node not in cmdIPsDic:
                    cmdIPsDic[node] = {label: [ident]}
                else:
                    if (label in cmdIPsDic[node]) and (ident not in cmdIPsDic[node][label]):
                            cmdIPsDic[node][label].append(ident)
                    else:
                        cmdIPsDic[node][label] = [ident]

                if ident not in labelIPDic:
                    labelIPDic[ident] = [ident_label]
                elif ident in labelIPDic and ident_label not in labelIPDic[ident]:
                    labelIPDic[ident] = labelIPDic[ident] + [ident_label]

            db.close()
            file_num += 1

        sourceDic = {ip:"+".join(labelIPDic[ip])+"_"+id_name for ip in labelIPDic.keys()}

        return cmdIPsDic,sourceDic