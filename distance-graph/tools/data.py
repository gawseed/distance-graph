import pandas as pd
import pickle
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
        self.cmd_to_old_label = {}
        self.got_unique_cmds = False

        self.__init_dataframe(args)
        self.find_unique_commands(args)
        self.build_labelDic(args)
    
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
    
    def build_labelDic(self, args):
        seen_nodes = set()
        file_num = 1

        for file_arg in args.file_args:
            inputfile_args = FileArguments(file_arg)

            db = pyfsdb.Fsdb(inputfile_args.filename)
            node_index = db.get_column_number(inputfile_args.node_name)

            ## if id given
            if args.id_name != '':
                id_index = db.get_column_number(inputfile_args.id_name)
                ident_label = inputfile_args.label_name
            
            for row in db:
                node = row[node_index]
                label = inputfile_args.label_name

                if args.args.temporal:
                    if (file_num in args.args.temporal) and (node not in seen_nodes): ## if looking at period 2 input file and node not seen
                        label = "new_"+label
                    elif node in seen_nodes: ## if node has been seen, continue
                        continue
                    else:
                        seen_nodes.add(node)
                
                if node[0]!="[":
                    node = str([node])
                
                if args.id_name != '':
                    ident = row[id_index]
                    if ident in self.loggedInOnly:
                        continue
                    self.update_labelDic_with_IPs(node, ident, label, ident_label)
                else:
                    self.update_labelDic(node, label)

    def update_labelDic_with_IPs(self, node, ident, label, ident_label):
        ## update labelDic
        if node not in self.labelDic:
            self.labelDic[node] = {label: [ident]}
        else:
            if (label in self.labelDic[node]) and (ident not in self.labelDic[node][label]):
                self.labelDic[node][label].append(ident)
            else:
                self.labelDic[node][label] = [ident]

        ## update sourceDic with labels for each identifier (IP)
        if ident not in self.sourceDic:
            self.sourceDic[ident] = [ident_label]
        elif ident in self.sourceDic and ident_label not in self.sourceDic[ident]:
            self.sourceDic[ident] = self.sourceDic[ident] + [ident_label]
    
    def update_labelDic(self, node, label):
        if node not in self.labelDic:
            self.labelDic[node] = [label]
        else:
            if label not in self.labelDic[node]:
                self.labelDic[node].append(label)

    def find_unique_templatized_cmds(self, args, templates, temporal):
        cmdTemplateDic = {}
        first_cmds = []
        templatized_cmds = []

        if temporal:
            to_remove = []
            to_add = []

        # use one command as representative
        for template,cmds in templates.template2cmd.items():
            all_cmds = cmds
            first_cmd = cmds[0]
            cmds = cmds[1:]
            
            if args.id_name != '': ## if cmd not found
                if first_cmd not in self.labelDic:
                    for i in range(len(cmds)):
                        if cmds[i] in self.labelDic:
                            first_cmd = cmds[i]
                            cmds = cmds[i+1:]
                            break

            cmdTemplateDic[first_cmd] = cmds

            first_cmds = first_cmds + [first_cmd]
            templatized_cmds = templatized_cmds + [cmds]

            ## if doing temporal analysis and template is a new template, need to update label and add 'new'
            if temporal:
                labels2cmds = self.find_labels2cmds(all_cmds, args)
                # if cmdIPsDic:
                #     labels2cmds = find_labelCmds(cmdIPsDic, all_cmds, 'ip')
                # else:
                #     labels2cmds = find_labelCmds(labelDic, all_cmds, 'label')
                for label,cmds in labels2cmds.items():
                    if 'new_' not in label and template in templates.new_templates:
                        new_label = 'new_'+label ## add new to label to indicate new template
                        to_remove.append((cmds,label))
                        if args.id_name != '':
                            ips = [ips for cmd in cmds for ips in self.labelDic[cmd][label]]
                        else:
                            ips = None
                        to_add.append((cmds, new_label, ips))
                    elif 'new_' in label and template not in templates.new_templates: ## if cmd new, but template is old >> remove 'new_' from label
                        new_label = label.replace('new_','')
                        to_remove.append((cmds,label))
                        if args.id_name != '':
                            ips = [ips for cmd in cmds for ips in self.labelDic[cmd][label]]
                        else:

                            ips = None
                        to_add.append((cmds, new_label, ips))

        if temporal:
            self.remove_add_labels(to_remove, to_add, args)

        # only keep 1st command of templatized commands as an example
        templatized_cmds = [cmd for cmd in templatized_cmds if cmd not in first_cmds]
        unique_cmds = [x for x in self.unique_cmds if x not in templatized_cmds]

        for cmd_key,cmds in cmdTemplateDic.items():
            if cmd_key not in self.labelDic:
                print(cmd_key)
            for cmd in cmds:
                if cmd not in self.labelDic:
                ## if (cmdIPsDic and cmd not in cmdIPsDic) or (labelDic and cmd not in labelDic):
                    cmdTemplateDic[cmd_key].remove(cmd)
        
        self.update_labelDic_templatized_cmds(cmdTemplateDic, args)
        self.unique_cmds = list(self.labelDic.keys())
    
    def find_labels2cmds(self, cmds, args):
        ## return {label1: [cmd1, cmd2, cmd3], label2: [cmd4, cmd5, cmd6]}
        labels2cmds = {}

        for cmd in cmds:
            if args.id_name != '':
                labels = list(self.labelDic[cmd].keys())
            else:
                labels = self.labelDic[cmd]
            for label in labels:
                if label not in labels2cmds:
                    labels2cmds[label] = [cmd]
                else:
                    labels2cmds[label] = labels2cmds[label] + [cmd]
        
        return labels2cmds
    
    def remove_add_labels(self, to_remove, to_add, args):
        for cmds,label in to_remove:
            for cmd in cmds:
                if args.id_name != '':
                    self.labelDic[cmd].pop(label)
                else:
                    self.labelDic[cmd].remove(label)
        for cmds,label,value in to_add:
            for cmd in cmds:
                if args.id_name != '':
                    self.labelDic[cmd][label] = value
                else:
                    self.labelDic[cmd] = self.labelDic[cmd]+[label]
    
    def update_labelDic_templatized_cmds(self, cmdTemplateDic, args):
        template_labelDic = {}
        
        for cmd in cmdTemplateDic: ## for every template
            if args.id_name != '':
                labels = self.labelDic[cmd].keys()
                IPsDic = {}

                for label in labels:
                    IPs = [self.labelDic[cmd][label]] + [self.labelDic[cmds][label] for cmds in cmdTemplateDic[cmd] if label in self.labelDic[cmds]]
                    IPs = [ip for lst in IPs for ip in lst]
                    IPsDic[label] = IPs

                template_labelDic[cmd] = IPsDic
            else:
                labels = [self.labelDic[cmd]] + [self.labelDic[cmds] for cmds in cmdTemplateDic[cmd]]
                labels = list(set([label for lst in labels for label in lst]))
                template_labelDic[cmd] = labels
        
        updated_labelDic = self.labelDic.copy()
        for cmd in template_labelDic:
            updated_labelDic[cmd] = template_labelDic[cmd]
            for cmds in cmdTemplateDic[cmd]:
                if cmds in updated_labelDic:
                    del updated_labelDic[cmds] ## only keep first representative command
    
        self.labelDic = updated_labelDic

    def get_template_nodes(self, templates, args):
        unique_cmds = []
        for cmd in self.unique_cmds:
            if (cmd in [cmd for lst in templates.template2cmd.values() for cmd in lst]):
                unique_cmds.append(cmd)
        self.unique_cmds = unique_cmds

        ## if label file given
        if args.args.labels:
            labels = pickle.load(open(args.args.labels,"rb"))
            self.update_representative_cmd(labels, templates)
    
    def update_representative_cmd(self, labels, templates):
        labeled_cmds = labels.keys()
        unique_cmds2 = [cmd[2:-2] for cmd in self.unique_cmds]
        change_cmds = {}

        i = 0
        for cmd in unique_cmds2:
            template = templates.cmd2template[cmd]
            temp_cmds = [temp_cmd[2:-2] for temp_cmd in templates.template2cmd[template]]

            for labeled_cmd in labeled_cmds:
                if labeled_cmd in temp_cmds and cmd != labeled_cmd:
                    change_cmds[cmd] = labeled_cmd
                    break
            
            i += 1

        self.unique_cmds = [str([change_cmds[cmd[2:-2]]]) if cmd[2:-2] in change_cmds else cmd for cmd in self.unique_cmds]
        self.cmd_to_old_label = change_cmds
        self.labelDic = self.remap_dic(self.labelDic, self.cmd_to_old_label)
        self.got_unique_cmds = True

    def remap_dic(self, dic, cmd_to_old_label, keys='array'):
        if keys == 'array':
            for cmd,old_label in cmd_to_old_label.items():
                cmd = str([cmd])
                old_label = str([old_label])
                dic[old_label] = dic[cmd]
                dic.pop(cmd)
        else:
            for cmd,old_label in cmd_to_old_label.items():
                dic[old_label] = dic[cmd]
                dic.pop(cmd)
        return dic