# from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType
from tools.arguments import *
from tools.templatizer import *
from tools.templates import Templates
from tools.data import Data
import pyfsdb
import pandas as pd
# import os
# import json
# import collections
import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import Levenshtein
import itertools
import re
import pickle
import numpy as np

# TOKENIZE_PATTERN = re.compile('[\s"=;|&><]')
# _NIX_COMMANDS = None

def get_commandCounts2(file_args, temporal):
    """ Counts number of commands run in the dataset and returns dict with command and respective counts
        If looking at temporal periods, then return two dictionaries with period 1 commands and period 1 & 2 commands respectively
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
                    if cmdCount2 == {}:
                        cmdCount2 = cmdCount.copy()
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

def get_inputFile_args(inputfile_args):
    """ Parse each input file arg and return filename, node column name, label column name, and identifier column name
    Input:
        inputfile_args (list) - list of input arguments user provided for input file
    Output:
        filename, node_name, label_name, identifier_name (str) - return input arguments
    """
    filename = inputfile_args[0]
    node_name = inputfile_args[1]
    label_name = inputfile_args[2]
    identifier_name = inputfile_args[3]

    return filename, node_name, label_name, identifier_name

def get_outputNames(output_names):
    """ Parse each output name argument and return node name, label name, and identifier name to be used for final output and graph
    Input:
        output_names (list) - list of output names to use for node, label, and identifier
    Output:
        node_name, label_name, identifier_name (str) - return node, label, and identifier for output
    """
    node_name = output_names[0]
    label_name = output_names[1]
    identifier_name = output_names[2]

    return node_name, label_name, identifier_name

def map_output_names(file_args, output_names):
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

def get_info2(args, cmd2template):
    """ Cleans and transforms input data according to user arguments. Finds weights and labels of nodes.
        Return four dictionaries: (1) weights between commands, (2) IPs that ran commands, (3) sources for each command,
        and (4) command to array style string
    Input:
        file_args (list) - list of file argument lists with format [filename, node name, label name, identifier name]
        output_names (list) - list of output names to use for node, label, and identifier
        cmd2template (dict) - maps templatizable commands to highest degree template
        args (class) - ArgumentParser class with input arguments
    Output:
        weightDic (dict) - key: pair of commands (tuple) / value: weight (float)
        cmdIPsDic (dict) - key: command (str) / value: dictionary with key: source (str) & value: IPs that ran command (list)
        sourceDic (dict) - key: command (str) / value: source label (str)
        cmdToArray (dict) - key: command (str) / value: array style command (str)
        cmd2template (dict) - key: command (str) / value: template (tuple)
    """
    template_nodes = args.args.template_nodes
    templates_class = cmd2template
    temporal = args.args.temporal
    templates = {}
    cmd2templateCount = {}

    data = Data(args)
    login_index = False

    if args.id_name != '':
        cmdIPsDic = data.labelDic
        sourceDic = data.sourceDic
    else:
        labelDic = data.labelDic
        cmdIPsDic = None

    got_unique_cmds = False

    if cmd2template:
        data.find_unique_templatized_cmds(args, cmd2template, temporal)
        unique_cmds = data.unique_cmds
        # cmdIPsDic = None
        labelDic = data.labelDic
        templates = cmd2template.template2cmd
        old_templates = cmd2template.old_templates

        # templates = cmd2template.templates
        cmd2template = cmd2template.cmd2template
        # if cmdIPsDic:
        #     unique_cmds,cmdIPsDic,templates,old_templates = get_uniqueCmds(data.unique_cmds,cmdIPsDic,{},templates,temporal)
        # else:
        #     unique_cmds,labelDic,templates,old_templates = get_uniqueCmds(data.unique_cmds,cmdIPsDic,labelDic,templates,temporal)

        if template_nodes:
            # unique_cmds2 = []
            # for cmd in unique_cmds:
            #     if (cmd in [cmd for lst in templates.values() for cmd in lst]):
            #     # if (cmd in [cmd for lst in templates[0].values() for cmd in lst]) or (cmd in [cmd for lst in templates[1].values() for cmd in lst]):
            #         unique_cmds2.append(cmd)
            
            # unique_cmds = unique_cmds2
            data.get_template_nodes(templates_class)
            unique_cmds = data.unique_cmds

            templates_class.calculate_template_counts(data.df, args.node_name, unique_cmds)
            # templateCounts = templates_class.template_counts
            cmd2templateCount = templates_class.cmd2template_count
            
            # templateCounts = calc_templateCount(templates,data.df,args.node_name)
            # cmd2templateCount = map_cmd2templateCount(cmd2template,templateCounts,unique_cmds)

            if args.args.labels:
                labels = pickle.load(open(args.args.labels,"rb"))
                unique_cmds,cmd_to_old_label = update_representativeCmd(unique_cmds,labels,cmd2template,templates)
                cmdToArray = {cmd[2:-2]:cmd for cmd in unique_cmds}
                unique_cmds = [cmd[2:-2] for cmd in unique_cmds]

                if cmdIPsDic:
                    cmdIPsDic = remap_dic(cmdIPsDic,cmd_to_old_label)
                elif labelDic:
                    labelDic = remap_dic(labelDic,cmd_to_old_label)

                cmd2templateCount = remap_dic(cmd2templateCount,cmd_to_old_label,'cmd')
                
                got_unique_cmds = True
    
    if got_unique_cmds == False:
        cmdToArray = {cmd[2:-2]:cmd for cmd in unique_cmds}
        unique_cmds = [cmd[2:-2] for cmd in unique_cmds]

    # cmdToArray = {cmd[2:-2]:cmd for cmd in unique_cmds}
    # unique_cmds = [cmd[2:-2] for cmd in unique_cmds]
    
    distDic = get_distances(unique_cmds)
    weightDic = get_weights(distDic)

    if cmdIPsDic:
        sourceDic.update({cmd:"+".join(list(cmdIPsDic[cmdToArray[cmd]].keys()))+"_"+args.node_name for cmd in unique_cmds})
    else:
        sourceDic = {cmd:"+".join(labelDic[cmdToArray[cmd]])+"_"+args.node_name for cmd in unique_cmds}

    return weightDic,cmdIPsDic,sourceDic,cmdToArray,cmd2template,templates,cmd2templateCount,old_templates


def get_info(file_args, output_names, cmd2template, args):
    """ Cleans and transforms input data according to user arguments. Finds weights and labels of nodes.
        Return four dictionaries: (1) weights between commands, (2) IPs that ran commands, (3) sources for each command,
        and (4) command to array style string
    Input:
        file_args (list) - list of file argument lists with format [filename, node name, label name, identifier name]
        output_names (list) - list of output names to use for node, label, and identifier
        cmd2template (dict) - maps templatizable commands to highest degree template
        args (class) - ArgumentParser class with input arguments
    Output:
        weightDic (dict) - key: pair of commands (tuple) / value: weight (float)
        cmdIPsDic (dict) - key: command (str) / value: dictionary with key: source (str) & value: IPs that ran command (list)
        sourceDic (dict) - key: command (str) / value: source label (str)
        cmdToArray (dict) - key: command (str) / value: array style command (str)
        cmd2template (dict) - key: command (str) / value: template (tuple)
    """
    template_nodes = args.template_nodes
    temporal = args.temporal
    templates = {}
    cmd2templateCount = {}

    df = pd.DataFrame()

    node_name, label_name, id_name = get_outputNames(output_names)

    for inputfile_args in file_args:
        filename = inputfile_args[0]
        map_input2output_names = map_output_names(inputfile_args, output_names)

        db = pyfsdb.Fsdb(filename)

        login_index = False

        data = db.get_pandas(data_has_comment_chars=True)

        if inputfile_args[2] != '':
            data['label'] = inputfile_args[2]

        data = data.rename(columns=map_input2output_names)
        
        df = pd.concat([df,data]).reset_index(drop=True)

        db.close()

    df[node_name] = df[node_name].apply(lambda x: str([x]) if x[0]!="[" else x)

    if id_name != '':
        loggedInOnly = get_loggedInOnly(df,node_name,label_name,id_name)

        df2 = df.copy()[~df[id_name].isin(loggedInOnly)]
        df2 = df2[df2[node_name]!='[]']
        unique_cmds = list(df2[node_name].unique())

        cmdIPsDic,sourceDic = get_cmdIPsDic(file_args,loggedInOnly,id_name,login_index, temporal)
    else:
        df2 = df.copy()
        df2 = df2[df2[node_name]!='[]']
        unique_cmds = list(df2[node_name].unique())

        labelDic = get_labelDic(file_args,login_index,temporal)
        cmdIPsDic = None

    got_unique_cmds = False

    if cmd2template:
        templates,cmd2template = get_templates(cmd2template)
        if cmdIPsDic:
            unique_cmds,cmdIPsDic,templates,old_templates = get_uniqueCmds(unique_cmds,cmdIPsDic,{},templates,temporal)
        else:
            unique_cmds,labelDic,templates,old_templates = get_uniqueCmds(unique_cmds,cmdIPsDic,labelDic,templates,temporal)

        if template_nodes:
            unique_cmds2 = []
            for cmd in unique_cmds:
                if (cmd in [cmd for lst in templates.values() for cmd in lst]):
                # if (cmd in [cmd for lst in templates[0].values() for cmd in lst]) or (cmd in [cmd for lst in templates[1].values() for cmd in lst]):
                    unique_cmds2.append(cmd)
            
            unique_cmds = unique_cmds2
            templateCounts = calc_templateCount(templates,df,node_name)
            cmd2templateCount = map_cmd2templateCount(cmd2template,templateCounts,unique_cmds)

            if args.labels:
                labels = pickle.load(open(args.labels,"rb"))
                unique_cmds,cmd_to_old_label = update_representativeCmd(unique_cmds,labels,cmd2template,templates)
                cmdToArray = {cmd[2:-2]:cmd for cmd in unique_cmds}
                unique_cmds = [cmd[2:-2] for cmd in unique_cmds]

                if cmdIPsDic:
                    cmdIPsDic = remap_dic(cmdIPsDic,cmd_to_old_label)
                elif labelDic:
                    labelDic = remap_dic(labelDic,cmd_to_old_label)

                cmd2templateCount = remap_dic(cmd2templateCount,cmd_to_old_label,'cmd')
                
                got_unique_cmds = True
    
    if got_unique_cmds == False:
        cmdToArray = {cmd[2:-2]:cmd for cmd in unique_cmds}
        unique_cmds = [cmd[2:-2] for cmd in unique_cmds]

    # cmdToArray = {cmd[2:-2]:cmd for cmd in unique_cmds}
    # unique_cmds = [cmd[2:-2] for cmd in unique_cmds]
    
    distDic = get_distances(unique_cmds)
    weightDic = get_weights(distDic)

    if cmdIPsDic:
        sourceDic.update({cmd:"+".join(list(cmdIPsDic[cmdToArray[cmd]].keys()))+"_"+node_name for cmd in unique_cmds})
    else:
        sourceDic = {cmd:"+".join(labelDic[cmdToArray[cmd]])+"_"+node_name for cmd in unique_cmds}

    return weightDic,cmdIPsDic,sourceDic,cmdToArray,cmd2template,templates,cmd2templateCount,old_templates

def remap_dic(dic, cmd_to_old_label, keys='array'):
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

def get_loggedInOnly(df,node_name,label,id_name):
    """ Returns list of IP addresses that only logged in and did not run any commands 
    Input:
        df (Pandas DataFrame) - dataframe with info on IPs, commands
        node_name (str) - column name of node (eg. "command")
        label (str) - column name of label (eg. "source")
        id_name (str) - column name of identifier column (eg. "ip")
    Output:
        loggedInOnly (list) - IPs that only logged in
    """
    loggedIn = df[df[node_name]=='[]'][id_name].unique()
    loggedInOnly = []
    labels = df[label].unique()

    for ip in loggedIn:
        cmdsRun = []
        for label_name in labels:
            cmdsRun = cmdsRun + list(df[(df[id_name]==ip) & (df[label]==label_name)][node_name].unique())
        if cmdsRun == ['[]']:
            loggedInOnly.append(ip)

    return loggedInOnly

def get_cmdIPsDic(file_args,loggedInOnly,id_name,login_index,temporal):
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
    for inputfile_args in file_args:
        filename, input_node_name, input_label_name, input_id_name = get_inputFile_args(inputfile_args)
        db = pyfsdb.Fsdb(filename)
        seenNodes = list(set(seenNodes))

        id_index = db.get_column_number(input_id_name)
        node_index = db.get_column_number(input_node_name)

        for row in db:
            ident = row[id_index] ## identifier (IP address)
            
            if ident in loggedInOnly: ## if IP only logged in, do not record
                continue

            ## check if login_index is provided. Skip over data where login_successful is false
            if (login_index != False) and (row[login_index] == 'False'):
                continue
            
            node = row[node_index]
            label = input_label_name
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

def get_labelDic(file_args, login_index, temporal):
    """ Returns dict that maps node to list of labels node has
    Input:
        input_file (str) - FSDB input file
        node_name (str): column name of node (eg. "command")
        label_name (str): column name of label (eg. "source")
        temporal (None/list) - list of file numbers (period 2) for temporal analysis
    Output:
        labelDic (dict) - key: node (str) / value: list of labels
    """
    labelDic = {}

    seenNodes = []
    file_num = 1

    for inputfile_args in file_args:
        filename, input_node_name, input_label_name, input_id_name = get_inputFile_args(inputfile_args)
        db = pyfsdb.Fsdb(filename)
        seenNodes = list(set(seenNodes))

        node_index = db.get_column_number(input_node_name)

        for row in db:
            node = row[node_index]
            label = input_label_name

            if (login_index != False) and (row[login_index] == 'False'):
                continue

            if temporal:
                if (file_num in temporal) and (node not in seenNodes):
                    label = "new_"+label
                elif node in seenNodes:
                    continue
                else:
                    seenNodes.append(node)
            
            if node[0]!="[":
                node = str([node])
            
            if node not in labelDic:
                labelDic[node] = [label]
            else:
                if label not in labelDic[node]:
                    labelDic[node].append(label)
        
        db.close()
        file_num += 1

    return labelDic

def get_templates(cmd2templates):
    """ Gets list of commands that belong to each template and returns a dictionary
    Input:
        cmd2template (dict) - key: command (str) / value: template (tuple)
    Output:
        template2cmd (dict) - key: template (tuple) / value: commands that belong to the template (list)
    """
    templates = []
    cmd2template2 = {}

    for cmd2template in cmd2templates.templates:
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

    return templates,cmd2template2

def calc_templateCount(template2cmd,df,node_name):
    """ Counts how many of the templatized commands were run in the data and returns dict
    Input:
        template2cmd (dict) - key: template (tuple) / value: list of commands that match the template (list)
        df (pandas DataFrame) - DataFrame consisting of commands run in the full data
        node_name - name of node column
    Output:
        templateCounts (dict) - key: template (tuple) / value: number of commands that match template (int)
    """
    templateCounts = {}
    for template,cmds in template2cmd.items():
        count = df[node_name].isin(cmds).sum()
        templateCounts[template] = count

    return templateCounts

def map_cmd2templateCount(cmd2template,templateCounts,unique_cmds):
    unique_cmds = [cmd[2:-2] for cmd in unique_cmds]
    cmd2templateCount = {cmd:templateCounts[cmd2template[cmd]] for cmd in unique_cmds}
    # cmd2templateCount = {cmd:int(math.sqrt(5*templateCounts[cmd2template[cmd]])) for cmd in unique_cmds}
    return cmd2templateCount

def get_uniqueCmds(cmds,cmdIPsDic,labelDic,templates,temporal):
    """ Returns list of unique commands, dict that maps command to a dict that has source and IPs that ran command
    Input:
        cmds (list) - list of commands,
        cmdIPsDic (dict) - dict that maps command to a dictionary that has source and IPs that ran the command
        templates (list) - list of template2cmd dicts
        template2cmd (dict) - key: template (tuple) / value: commands that belong to the template (list)
        temporal (None/list) - list of file numbers (period 2) for temporal analysis
    Output:
        unique_cmds (list) - list of unique commands
        cmdIPsDic (dict) - maps command to a dict that has source and IPs that ran the command
    """
    unique_cmds = cmds
    cmdTemplateDic = {}
    first_cmds = []
    templatized_cmds = []

    if temporal:
        new_templates = find_new_templates(templates)
        old_templates = find_old_templates(templates)
        to_remove = []
        to_add = []
    else:
        new_templates = templates
        old_templates = templates

    template2cmd = combine_templates(templates)
    # for template2cmd in templates:
    for template,cmds in template2cmd.items():
        all_cmds = cmds
        first_cmd = cmds[0]
        cmds = cmds[1:]
        
        if cmdIPsDic:
            if first_cmd not in cmdIPsDic:
                for i in range(len(cmds)):
                    if cmds[i] in cmdIPsDic:
                        first_cmd = cmds[i]
                        cmds = cmds[i+1:]
                        break

        cmdTemplateDic[first_cmd] = cmds

        first_cmds = first_cmds + [first_cmd]
        templatized_cmds = templatized_cmds + [cmds]

        ## if doing temporal analysis and template is a new template
        if temporal:
            if cmdIPsDic:
                labels2cmds = find_labelCmds(cmdIPsDic, all_cmds, 'ip')
            else:
                labels2cmds = find_labelCmds(labelDic, all_cmds, 'label')
            for label,cmds in labels2cmds.items():
                if 'new_' not in label and template in new_templates:
                    new_label = 'new_'+label ## add new to label to indicate new template
                    to_remove.append((cmds,label))
                    if cmdIPsDic:
                        ips = [ips for cmd in cmds for ips in cmdIPsDic[cmd][label]]
                    else:
                        ips = None
                    to_add.append((cmds, new_label, ips))
                elif 'new_' in label and template not in new_templates: ## if cmd new, but template is old >> remote 'new_' from label
                    new_label = label.replace('new_','')
                    to_remove.append((cmds,label))
                    if cmdIPsDic:
                        ips = [ips for cmd in cmds for ips in cmdIPsDic[cmd][label]]
                    else:
                        ips = None
                    to_add.append((cmds, new_label, ips))

    if temporal:
        if cmdIPsDic:
            for cmds,label in to_remove:
                for cmd in cmds:
                    cmdIPsDic[cmd].pop(label)
            for cmds,label,value in to_add:
                for cmd in cmds:
                    cmdIPsDic[cmd][label] = value
        else:
            for cmds,label in to_remove:
                for cmd in cmds:
                    labelDic[cmd].remove(label)
            for cmds,label,value in to_add:
                for cmd in cmds:
                    labelDic[cmd] = labelDic[cmd]+[label]

    # only keep 1st command of templatized commands as an example
    templatized_cmds = [cmd for cmd in templatized_cmds if cmd not in first_cmds]
    unique_cmds = [x for x in unique_cmds if x not in templatized_cmds]

    for cmd_key,cmds in cmdTemplateDic.items():
        for cmd in cmds:
            if (cmdIPsDic and cmd not in cmdIPsDic) or (labelDic and cmd not in labelDic):
                cmdTemplateDic[cmd_key].remove(cmd)

    if cmdIPsDic:       
        cmdIPsDic = update_cmdIPsDic(cmdIPsDic,cmdTemplateDic)
        unique_cmds = list(cmdIPsDic.keys())
        # print("Finished with cmdIPsDic")
        return unique_cmds,cmdIPsDic,template2cmd,old_templates
    else:
        labelDic = update_labelDic(labelDic, cmdTemplateDic)
        unique_cmds = list(labelDic.keys())
        # print("Finished with labelDic")
        return unique_cmds,labelDic,template2cmd,old_templates

def find_new_templates(templates):
    """ Compares templates from period 1 to period 2 and returns list of templates from period 2 not found in period 1
    Input:
        templates (list) - list containing two cmd2template dictionaries from period 1 and period 2
    Output:
        new_templates (list) - list of new templates
    """
    templates1 = templates[0].keys()
    templates2 = templates[1].keys()
    new_templates = [template for template in templates2 if template not in templates1]

    return new_templates

def find_old_templates(templates):
    """ Compares templates from period 1 to period 2 and returns list of templates from period 2 not found in period 1
    Input:
        templates (list) - list containing two cmd2template dictionaries from period 1 and period 2
    Output:
        new_templates (list) - list of new templates
    """
    templates1 = templates[0].keys()
    templates2 = templates[1].keys()
    old_templates = [template for template in templates1 if template not in templates2]

    return old_templates

def combine_templates(templates):
    """ Combines list of template2cmd dicts into one dictionary
    Input:
        templates (list) - list containing two cmd2template dictionaries from period 1 and period 2
    Output:
        templateDic (dict) - key: template (tuple) / value: commands that templatize to the template (list)
    """
    templateDic = {}
    for template2cmd in templates:
        for template,cmds in template2cmd.items():
            if template not in templateDic:
                templateDic[template] = cmds
            else:
                templateDic[template] = templateDic[template] + cmds

    templateDic = {template:sorted(set(cmds)) for template,cmds in templateDic.items()}
    
    return templateDic

def find_labelIPs(cmdIPsDic, cmds):
    labels2ips = {}

    for cmd in cmds:
        ipsDic = cmdIPsDic[cmd]
        labels = list(cmdIPsDic[cmd].keys())
        for label in labels:
            if label not in labels2ips:
                labels2ips[label] = ipsDic[label]
            else:
                labels2ips[label] = labels2ips[label] + ipsDic[label]
    
    return labels2ips

def find_labelCmds(cmdIPsDic, cmds, type):
    labels2cmds = {}

    for cmd in cmds:
        if type == 'ip':
            labels = list(cmdIPsDic[cmd].keys())
        else:
            labels = cmdIPsDic[cmd]
        for label in labels:
            if label not in labels2cmds:
                labels2cmds[label] = [cmd]
            else:
                labels2cmds[label] = labels2cmds[label] + [cmd]
    
    return labels2cmds

def get_allLabels(cmdIPsDic,cmds):
    all_labels = []
    for cmd in cmds:
        labels = list(cmdIPsDic[cmd].keys())
        all_labels = all_labels + labels

    all_labels = set(all_labels)

    return all_labels

def update_cmdIPsDic(cmdIPsDic,cmdTemplateDic):
    """ Returns updated cmdIPsDic dict so templatized command IPs include all IPs that ran cmd with templatized cmd
    Input:
        cmdIPsDic (dict) - maps command to a dictionary that has source and IPs that ran the command,
        cmdTemplateDic (dict) - maps first command of template to list of all other commands of same template
    Output: 
        cmdIPs (dict) - maps command to dict that contains source and IP addresses
    """
    template_cmdIPsDic = {}
    for cmd in cmdTemplateDic: ## for every template
        labels = cmdIPsDic[cmd].keys()
        IPsDic = {}

        for label in labels:
            IPs = [cmdIPsDic[cmd][label]] + [cmdIPsDic[cmds][label] for cmds in cmdTemplateDic[cmd] if label in cmdIPsDic[cmds]]
            IPs = [ip for lst in IPs for ip in lst]
            IPsDic[label] = IPs

        template_cmdIPsDic[cmd] = IPsDic
    
    cmdIPs = cmdIPsDic.copy()

    for cmd in template_cmdIPsDic:
        cmdIPs[cmd] = template_cmdIPsDic[cmd]
        for cmds in cmdTemplateDic[cmd]:
            if cmds in cmdIPs:
                del cmdIPs[cmds]
    
    return cmdIPs

def update_labelDic(labelDic, cmdTemplateDic):
    updated_labelDic = labelDic.copy()
    template_labelDic = {}

    for cmd in cmdTemplateDic:
        labels = [labelDic[cmd]] + [labelDic[cmds] for cmds in cmdTemplateDic[cmd]]
        labels = [label for lst in labels for label in lst]
        labels = list(set(labels))
        template_labelDic[cmd] = labels

    for cmd in template_labelDic:
        updated_labelDic[cmd] = template_labelDic[cmd]
        for cmds in cmdTemplateDic[cmd]:
            if cmds in updated_labelDic:
                del updated_labelDic[cmds]
        #updated_labelDic[cmd] = template_labelDic[cmd]
    
    return updated_labelDic

def update_representativeCmd(unique_cmds,labels,cmd2templates,templates):
    labeled_cmds = labels.keys()
    unique_cmds2 = [cmd[2:-2] for cmd in unique_cmds]
    # templatized_cmds = [cmd[2:-2] for lst in templates.values() for cmd in lst]
    change_cmds = {}

    i = 0
    for cmd in unique_cmds2:
        template = cmd2templates[cmd]
        temp_cmds = [temp_cmd[2:-2] for temp_cmd in templates[template]]

        for labeled_cmd in labeled_cmds:
            if labeled_cmd in temp_cmds and cmd != labeled_cmd:
                change_cmds[cmd] = labeled_cmd
                # unique_cmds.append(labeled_cmd)
                # remove_cmds.append(cmd)
                break
        
        i += 1
        # for arrayCmd in temp_cmds:
        #     temp_cmd = arrayCmd[2:-2]
        #     if temp_cmd in labels:
        #         print(temp_cmd)
        #         unique_cmds.append(temp_cmd)
        #         remove_cmds.append(cmd)
        #         break

    unique_cmds = [str([change_cmds[cmd[2:-2]]]) if cmd[2:-2] in change_cmds else cmd for cmd in unique_cmds]
    # unique_cmds = [str([cmd]) for cmd in unique_cmds]
    
    return unique_cmds, change_cmds

def get_distances(cmds):
    """ Returns dict that maps every pair of commands with their calculated distance
    Input:
        cmds (list) - list of commands
    Output:
        distDic (dict) - key: pair of commands (tuple) / value: distance between commands (float)
    """
    cmdCombos = list(itertools.combinations(cmds,2))

    distDic = {}

    for combo in cmdCombos:
        cmd1, cmd2 = combo
        length = len(cmd1)+len(cmd2)
        distDic[combo] = Levenshtein.distance(cmd1, cmd2)/length

    return distDic

def get_weights(distDic):
    """ Returns dict with weights of every pair of commands. Weights are inversely proportional to distance
    Input:
        distDic (dict) - dict with distances of every pair of commands
    Output:
        weightDic (dict) - key: pair of commands (tuple) / value: weight between commands (float)
    """
    distances = sorted(list(set(distDic.values())))

    weights = {}
    maxWeight = max(distances)

    for i in range(len(distances)):
        if i==0:
            weights[distances[i]] = maxWeight
            weight = maxWeight
        else:
            diff = distances[i]-distances[i-1]
            weights[distances[i]] = weight-diff
            weight = weight-diff
    
    weightDic = {}
    for pair,dist in distDic.items():
        weightDic[pair] = weights[dist]

    return weightDic

def draw_networkx(args,output_names,weightDic,cmdIPsDic,sourceDic,cmdToArray,cmd2templates,cmd2templateCount,old_templates):
    """ Finds the weighted edges and plots the NetworkX graph. Obtains labels for nodes and cluster IDs for connected nodes
    Input:
        args (argument parser) - parser of command line arguments,
        weightDic (dict) - weights for each pair of commands, cmdIPsDic (dict) - maps command to a dictionary that has source and IPs that ran the command,
        cmdIPsDic (dict) - maps command to dict that contains source and IP addresses
        sourceDic (dict) - maps command to source label, cmdToArray (dict) - maps command to array style command,
        cmdToArray (dict) - maps command (str) to array style command (str)
    Output:
        G (NetworkX graph) - labeled graph with weighted edges,
        weighted_edges (list) - list of tuples containing command pair and weight,
        labels (dict) - maps command node to integer,
        clusters (dict) - key: command node (str) / value: cluster ID (int)
    """
    threshold = args.args.edge_weight
    pos = args.args.position
    output_file = args.args.output_file[0]
    figsize = tuple([args.args.width,args.args.height])
    node_name, label_name, id_name = get_outputNames(output_names)

    edgeweight = [tuple(list(k)+[v]) for k,v in weightDic.items()]

    if args.args.top_k:
        k = args.args.top_k
        weighted_edges = get_topK_edges(k, edgeweight, k_edges=args.args.top_k_edges)
    else:
        weighted_edges = [x for x in edgeweight if x[2] > threshold]

    weighted_edges = sorted(weighted_edges)
    G = nx.Graph()
    G.add_weighted_edges_from(sorted(weighted_edges))

    if (args.args.labels):
        labels = pickle.load(open(args.args.labels,"rb"))
        labels = add_newLabels(G,labels,cmd2templates)
    else:
        labels = get_numberNodes(G)

    clusters = get_clusters(G)

    if cmdIPsDic:
        add_IPnodes(G,cmdToArray,cmdIPsDic)

    nodeTypeDic,colorslist = set_nodeColors(G,sourceDic,id_name)

    if (args.args.temporal):
        pos = plot_temporal_networkx(G,pos,output_file,labels,colorslist,nodeTypeDic,id_name,cmd2templateCount,cmd2templates,old_templates,figsize=figsize,font_size=args.args.font_size,node_size=args.args.node_size)
    else:
        pos = plot_networkx(G,pos,output_file,labels,colorslist,nodeTypeDic,id_name,cmd2templateCount,figsize=figsize,font_size=args.args.font_size,node_size=args.args.node_size)

    return G,weighted_edges,labels,clusters,pos

def get_topK_edges(k,edgeweight,k_edges=False):
    """ Finds top k highest weight edges and returns list of top k edges
    Input:
        k (int): number of edges to return
        edgeweight (list): list of edgeweights where each element is (node1, node2, weight)
    Output:
        topK_edges (list) - list of top K edges
    """
    topK = {}
    edgeweight = sorted(edgeweight, key=lambda x: x[2], reverse=True)

    if k_edges: ## get k edges for each node
        for i in range(len(edgeweight)):
            edge = edgeweight[i]
            cmd1,cmd2,weight = edge

            if cmd1 in topK:
                if len(topK[cmd1]) < k:
                    topK[cmd1] = topK[cmd1] + [edge]
            else:
                topK[cmd1] = [edge]
            
            if cmd2 in topK:
                if len(topK[cmd2]) < k:
                    topK[cmd2] = topK[cmd2] + [edge]
            else:
                topK[cmd2] = [edge]
    else: ## get top k nodes
        for i in range(len(edgeweight)):
            edge = edgeweight[i]
            cmd1,cmd2,weight = edge

            if len(topK) < k:
                if cmd1 in topK:
                    topK[cmd1] = topK[cmd1] + [edge]
                else:
                    topK[cmd1] = [edge]
                
                if cmd2 in topK:
                    topK[cmd2] = topK[cmd2] + [edge]
                else:
                    topK[cmd2] = [edge]
                    
    topK_edges = list(set([tups for lst in list(topK.values()) for tups in lst]))
    
    return topK_edges

def add_newLabels(G,labels,cmd2templates):
    nodes = G.nodes()
    num_labels = [labels[cmd]['label'] for cmd in labels]
    template2label = {labels[cmd]['template']:labels[cmd]['label'] for cmd in labels}
    # num_labels = list(labels.values())
    new_labels = {}
    
    i=max(num_labels)+1
    for node in nodes:
        if node in labels:
            new_labels[node] = labels[node]['label']
        else:
            if cmd2templates != {}:
                template = cmd2templates[node]
                if template in template2label:
                    label = template2label[template]
                    new_labels[node] = label
                else:
                    new_labels[node] = i
                i+=1
            else:
                new_labels[node] = i
                i+=1
    
    return new_labels

def get_numberNodes(G):
    """ Returns dict that has nodes mapped to a unique integer label
    Input:
        G (NetworkX graph) - graph with IP and command nodes
        sourceDic (dict) - maps command nodes to source label
    Output:
        labels (dict) - maps node to labeled number
    """
    nodes = sorted(G.nodes())
    labels = {}

    i=0
    for node in nodes:
        labels[node] = i
        i += 1
    
    return labels

def add_IPnodes(G,cmdToArray,cmdIPsDic):
    """  Adds IP edges to command nodes
    Input:
        G (NetworkX graph) - graph to add IP edges to
        cmdToArray (dict) - maps command to array style command
        cmdIPsDic (dict) - maps command to dict that contains source and IP addresses
    """
    nodes = list(G.nodes())

    for node in nodes:
        cmd = cmdToArray[node]
        ips = get_IPs(cmd,cmdIPsDic)
        edges = [(node,ip) for ip in ips]
        G.add_edges_from(edges)

def get_IPs(cmd,dic):
    """ Finds and returns all IP addressses that ran a command
    Input:
        cmd (str) - command in array style
        dic (dict) - dict with commands mapped to source IPs
    Output:
        (list) unique IPs that ran command
    """
    ips = []

    for label,label_ips in dic[cmd].items():
        ips = ips + label_ips
    
    return list(set(ips))

def set_nodeColors(G,sourceDic,id_name):
    """ Sets source as an attribute for each command node, returns nodeTypeDic and color list
    Input:
        G (NetworkX graph) - graph with command nodes
        sourceDic (dict) - maps command nodes to source label
        id_name (str) - column name of identifier column
    Output:
        nodeTypeDic (dict) - maps node type with nodes
        colorslist (list) - list of color for node types
    """
    nx.set_node_attributes(G,name="source",values=sourceDic)
    sources = set(sorted(nx.get_node_attributes(G,"source").values()))

    if id_name != '':
        id_labels = [label for label in sources if id_name in label]
    else:
        id_labels = []

    source_labels = [label for label in sources if label not in id_labels]
    types = sorted(source_labels) + sorted(id_labels)

    colorslist = get_colors(types)

    nodeTypeDic = get_nodeTypeDic(types,G.nodes(),sourceDic,id_name)
    return nodeTypeDic,colorslist

def get_colors(types):
    """ Given all types of nodes to graph, return list of colors to use for each node type. Nodes that come from the same sources will have the same colors
    Input:
        types (list): list of node types to graph (will appear in legend)
    Output:
        colors_to_use (list): list of colors to use for each node
    """
    sourceToColor = {}
    colors_to_use = []
    colorslist = ["b","c","r","tab:orange","y","lime","tab:pink","g","tab:brown","tab:purple"]

    for type in types:
        source = type.split('_')[0]

        if source not in sourceToColor:
            if 'new' in source: ## make new nodes red color
                if "r" in colorslist:
                    sourceToColor[source] = 'r'
                    colorslist.pop(colorslist.index('r'))
                else:
                    sourceToColor[source] = 'lime'
                    colorslist.pop(colorslist.index('lime'))
            else:
                sourceToColor[source] = colorslist[0]
                colorslist = colorslist[1:]
        
        colors_to_use.append(sourceToColor[source])
    
    return colors_to_use

def get_nodeTypeDic(types,nodes,sourceDic,id_name):
    """ Maps node label to list of nodes and returns dict
    Input:
        types (list) - list of node types/labels
        nodes (NetworkX nodes) - nodes of NetworkX graph
        sourceDic (dict) - maps command nodes to source label
        id_name (str) - column name of identifier column
    Output:
        nodeTypeDic (dict) - maps node type to list of nodes with that node type
    """
    nodeTypeDic = {nodetype:[] for nodetype in types}
    
    for node in nodes:
        source = sourceDic[node]
        nodeTypeDic[source] = nodeTypeDic[source]+[node]
            
    return nodeTypeDic

def plot_networkx(G,pos,output_file,labels,colorslist,nodeTypeDic,id_name,cmd2templateCount,figsize=(12,8),font_size=10,node_size=350,ip_alpha=0.2,cmd_alpha=0.2,edge_alpha=0.2):
    """ Plots NetworkX graph and saves image to output file
    Input:
        G (NetworkX graph) - graph with IP and command nodes to graph
        output_file (str) - filename for network graph image
        labels (dict) - maps node to integer label
        colorslist (list) - list of node colors
        nodeTypeDic (dict) - maps node type to list of nodes
        id_name (str) - column name of identifier column (eg. "ip")
    """  
    fig,ax = plt.subplots(1,figsize=figsize)

    if not pos:
        pos=nx.spring_layout(G)
    else:
        pos=pickle.load(open(pos,"rb"))
        fixed_nodes = pos.keys()
        pos=nx.spring_layout(G,pos=pos,fixed=fixed_nodes)
        
    i=0
    handles = []
    for nodetype in nodeTypeDic:
        nodelist = nodeTypeDic[nodetype]
        color = colorslist[i]
        i+=1

        if id_name and id_name in nodetype:
            alpha=ip_alpha
            nx.draw_networkx_nodes(G,pos=pos,nodelist=nodelist,ax=ax,\
                        label=nodetype,alpha=alpha,node_size=node_size,node_shape="^",node_color=color)
        else:
            alpha=cmd_alpha
            
            if cmd2templateCount != {}:
                node_sizes = [cmd2templateCount[node] for node in nodelist]
                node_size_template = [int((5*node)**0.5) for node in node_sizes]
                alpha = 0.25
                # alpha = [alpha if cmd2template[node] not in old_templates else 0.085 for node in nodelist]
                points = nx.draw_networkx_nodes(G,pos=pos,nodelist=nodelist,ax=ax,\
                        label=nodetype,alpha=alpha,node_size=node_size_template,node_color=color)
                handles.append(points.legend_elements("sizes", num=4))
            else:
                nx.draw_networkx_nodes(G,pos=pos,nodelist=nodelist,ax=ax,\
                            label=nodetype,alpha=alpha,node_size=node_size,node_color=color)

    nx.draw_networkx_edges(G,pos=pos,alpha=edge_alpha)
    nx.draw_networkx_labels(G,pos=pos,labels=labels,font_size=font_size)

    legend = ax.legend(scatterpoints=1, markerscale=0.75)
    for leg in legend.legendHandles:
        leg._sizes = [250]
    plt.gca().add_artist(legend)

    if handles != []:
        legend_handles, legend_labels = get_size_legend(handles)
        ax.legend(handles=legend_handles,labels=legend_labels,bbox_to_anchor=(1,1.15), title='command count')

    ## remove black border
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.savefig(output_file, dpi=300)

    return pos

def plot_temporal_networkx(G,pos,output_file,labels,colorslist,nodeTypeDic,id_name,cmd2templateCount,cmd2template,old_templates,figsize=(12,8),font_size=10,node_size=350,ip_alpha=0.1,cmd_alpha=0.25,edge_alpha=0.1):
    """ Plots NetworkX graph and saves image to output file
    Input:
        G (NetworkX graph) - graph with IP and command nodes to graph
        output_file (str) - filename for network graph image
        labels (dict) - maps node to integer label
        colorslist (list) - list of node colors
        nodeTypeDic (dict) - maps node type to list of nodes
        id_name (str) - column name of identifier column (eg. "ip")
    """  
    fig,ax = plt.subplots(1,figsize=figsize)

    if not pos:
        pos=nx.spring_layout(G)
    else:
        pos=pickle.load(open(pos,"rb"))
        fixed_nodes = pos.keys()
        pos=nx.spring_layout(G,pos=pos,fixed=fixed_nodes,k=0.3)
    
    handles = []
    i = 0
    for nodetype in nodeTypeDic:
        nodelist = nodeTypeDic[nodetype]
        color = colorslist[i]
        i += 1

        if id_name and id_name in nodetype:
            alpha=ip_alpha
            nx.draw_networkx_nodes(G,pos=pos,nodelist=nodelist,ax=ax,\
                        label=nodetype,alpha=alpha,node_size=node_size,node_shape="^",node_color=color)
        else:
            if 'new' in nodetype:
                alpha=0.4
            else:
                alpha=cmd_alpha
                alpha = [alpha if cmd2template[node] not in old_templates else 0.085 for node in nodelist]
            
            if cmd2templateCount != {}:
                node_sizes = [cmd2templateCount[node] for node in nodelist]
                node_size_template = [int((5*node)**0.5) for node in node_sizes]
                points = nx.draw_networkx_nodes(G,pos=pos,nodelist=nodelist,ax=ax,\
                        label=nodetype,alpha=alpha,node_size=node_size_template,node_color=color)
                handles.append(points.legend_elements("sizes", num=4))
            else:
                nx.draw_networkx_nodes(G,pos=pos,nodelist=nodelist,ax=ax,\
                            label=nodetype,alpha=alpha,node_size=node_size,node_color=color)

    nx.draw_networkx_edges(G,pos=pos,alpha=edge_alpha)
    nx.draw_networkx_labels(G,pos=pos,labels=labels,font_size=font_size)

    legend = ax.legend(scatterpoints=1, markerscale=0.75)
    for leg in legend.legendHandles:
        leg.set_alpha(0.5)
        leg._sizes = [250]
    plt.gca().add_artist(legend)

    if handles != []:
        legend_handles, legend_labels = get_size_legend(handles)
        legend2 = ax.legend(handles=legend_handles,labels=legend_labels,bbox_to_anchor = (1,1.15), title='command count')
        for leg in legend2.legendHandles:
            leg.set_alpha(0.3)

    ## remove black border
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.savefig(output_file, dpi=300)

    return pos

def get_size_legend(handles):
    handle_points = []
    handle_labels = []
    regex = r'(\d+)'
    
    for handle in handles:
        handle_points += handle[0]
        handle_labels += handle[1]

    handle_labels = [int(re.search(regex, handle).group(1)) for handle in handle_labels]
    all_handles = sorted([(handle_points[i],handle_labels[i]) for i in range(len(handle_points))], key=lambda x: x[1])
    to_keep = [min(handle_labels), max(handle_labels), all_handles[int(len(all_handles)/2)][1]]
    all_handles = [handle for handle in all_handles if handle[1] in to_keep]

    legend_handles = []
    legend_labels = []
    for handle in all_handles:
        if handle[1] not in legend_labels:
            legend_handles.append(handle[0])
            legend_labels.append(handle[1])
    
    legend_labels = [int((label**2)/5) for label in legend_labels]
    legend_labels = [f'{label:,}' for label in legend_labels]
    return legend_handles, legend_labels

def get_clusters(G):
    """ Finds clusters of commands in NetworkX graph. A cluster is considered to be nodes that are connected by an edge
    Input:
        G (NetworkX graph) - graph to find clusters
    Output:
        cmdToCluster (dict) - key: command node (str) / value: cluster ID (int)
    """
    ip_regex = r'^\d+\.\d+\.\d+\.\d+$'
    components = sorted([sorted(list(comp)) for comp in list(nx.connected_components(G))])

    clusters = {}
    i=0
    for comp in components:
        cluster = [node for node in comp if not re.search(ip_regex,node)]
        clusters[i] = cluster
        i+=1

    cmdToCluster = {}
    for cluster,commands in clusters.items():
        ids = {cmd:cluster for cmd in commands}
        cmdToCluster.update(ids)
        
    return cmdToCluster

def main():
    args = Arguments()
    # args = parse_args()

    # ## get list of file args and remove None input files args that were not specified
    # file_args = list(filter(None,[args.input_file1, args.input_file2, args.input_file3, args.input_file4]))
    # output_names = args.output_names

    # check_fileArgs(file_args, output_names)

    if args.args.templatize:
        # temporal = args.temporal
        # stopwords = args.stopwords
        if args.args.stopwords == None:
            raise Exception('Stopwords file is missing. The stopwords file is required for templatization.')

        # global _NIX_COMMANDS
        # if _NIX_COMMANDS is None:
        #     _NIX_COMMANDS = set(open(stopwords).read().split('\n'))

        cmd2template = Templates(args.args.stopwords, args.file_args, args.args.temporal)
        # cmd2template = get_cmd2template(file_args,temporal)
    else:
        cmd2template = None

    weightDic,cmdIPsDic,sourceDic,cmdToArray,cmd2template,templates,cmd2templateCount,old_templates = get_info2(args, cmd2template)
    # weightDic,cmdIPsDic,sourceDic,cmdToArray,cmd2template,templates,cmd2templateCount,old_templates = get_info(file_args, output_names, cmd2template, args)
    G,weighted_edges,labels,clusters,pos = draw_networkx(args,args.output_names,weightDic,cmdIPsDic,sourceDic,cmdToArray,cmd2template,cmd2templateCount,old_templates)

    ## save NetworkX graph position file to pickle file
    if (args.args.position_file):
        pickle.dump(pos, open(args.args.position_file, "wb" ))

    ## save labels dict to pickle file
    if (args.args.labels_file):
        if (args.args.template_nodes):
            output_labels = {cmd:{'label':label, 'template':cmd2template[cmd]} for cmd,label in labels.items()}
            pickle.dump(output_labels, open(args.args.labels_file, "wb"))
        else:
            output_labels = {cmd:{'label':label, 'template':"N/A"} for cmd,label in labels.items()}
            # pickle.dump(labels, open(args.labels_file, "wb" ))

    ## create edge list to FSDB file
    if (args.args.edge_list):
        if (args.args.template_nodes): ## if template nodes are being graphed, produce edge list that includes templates
            print("Graphing template nodes...")
            outh = pyfsdb.Fsdb(out_file=args.args.edge_list)
            outh.out_column_names=['cluster_id', 'weight', 'node1_id', 'node2_id', 'node1', 'node2', 'template1', 'template2']
            for cmd1,cmd2,weight in weighted_edges:
                cluster_id = clusters[cmd1]
                num1 = labels[cmd1]
                num2 = labels[cmd2]
                template1 = ' '.join(cmd2template[cmd1])
                template2 = ' '.join(cmd2template[cmd2])
                outh.append([cluster_id,round(weight,3),num1,num2,cmd1,cmd2,template1,template2])
            outh.close()
        else:
            outh = pyfsdb.Fsdb(out_file=args.args.edge_list)
            outh.out_column_names=['cluster_id', 'node1_id', 'node2_id', 'node1', 'node2', 'weight']
            for cmd1,cmd2,weight in weighted_edges:
                cluster_id = clusters[cmd1]
                num1 = labels[cmd1]
                num2 = labels[cmd2]
                outh.append([cluster_id,num1,num2,cmd1,cmd2,round(weight,3)])
            outh.close()

    ## create cluster list to FSDB file
    if (args.args.cluster_list):
        if (args.args.template_nodes): ## if template nodes are being graphed, produce edge list that includes templates
            outh = pyfsdb.Fsdb(out_file=args.args.cluster_list)
            outh.out_column_names=['cluster_id','command','template']
            for cmd,cluster_id in clusters.items():
                template = ' '.join(cmd2template[cmd])
                outh.append([cluster_id,cmd,template])
            outh.close()
        else:
            outh = pyfsdb.Fsdb(out_file=args.cluster_list)
            outh.out_column_names=['cluster_id','command']
            for cmd,cluster_id in clusters.items():
                outh.append([cluster_id,cmd])
            outh.close()

    ## create template list to FSDB file
    if (args.args.template_list):
        if (args.args.template_nodes): ## if template nodes are being graphed, produce template list that contains template, example command, and label
            outh = pyfsdb.Fsdb(out_file=args.args.template_list)
            outh.out_column_names=['template','command','node','label']
            template_list = []
            for cmd in clusters.keys():
                template = ' '.join(cmd2template[cmd])
                node = labels[cmd]
                if sourceDic != {}:
                    label = sourceDic[cmd]
                else:
                    label = ''
                template_list.append([template,cmd,node,label])

            template_list = sorted(template_list)
            for line in template_list:
                outh.append(line)
            outh.close()
        else:
            print("Template nodes were not graphed. No template list to output.")
    
    if (args.args.templatecmd_list):
        if (args.args.template_nodes):
            outh = pyfsdb.Fsdb(out_file=args.args.templatecmd_list)
            outh.out_column_names=['template','command']
            tc_list = []
            for temp,cmds in templates.items():
                template = ' '.join(temp)
                cmds = sorted(set(cmds))
                cmds = [cmd[2:-2] for cmd in cmds]
                for cmd in cmds:
                    tc_list.append([template,cmd])
            tc_list = sorted(tc_list)
            for line in tc_list:
                outh.append(line)
            outh.close()
        else:
            print("Template nodes were not graphed. No template command list to output.")

if __name__ == "__main__":
    main()