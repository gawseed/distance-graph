from tools.arguments import *
from tools.templatizer import *
from tools.templates import Templates
from tools.data import Data
from tools.graph import NetworkGraph
import pyfsdb
import pickle


def get_info(args, templates):
    """
    Extracts the data from the input files and gets necessary data for distance graph

    Parameters ---
    args : Arguments class
        Class with user arguments
    templates : Templates class
        Class with template information

    Returns ---
    data : Data class
        Class with data information
    templates : Templates class
        Class with updated template information
    """
    data = Data(args)

    if templates:
        data.find_unique_templatized_cmds(args, templates, args.args.temporal)

        if args.args.template_nodes:
            data.get_template_nodes(templates, args)
            templates.calculate_template_counts(data.df, args.node_name, data.unique_cmds)
            
            if args.args.labels:
                labels = pickle.load(open(args.args.labels,"rb"))
                data.update_representative_cmd(labels,templates)

                templates.cmd2template_count = data.remap_dic(templates.cmd2template_count, data.cmd_to_old_label, 'cmd')
    
    data.get_unique_cmds()
    data.calculate_weights()
    data.update_source_dic(args)

    return data,templates

def draw_networkx(args,data,templates):
    """
    Creates the NetworkX distance graph

    Parameters ---
    args : Arguments class
        Class with user arguments
    data : Data class
        Class with data information
    templates : Templates class
        Class with template information

    Returns ---
    networkgraph : NetworkGraph class
        Class with distance graph information--labels, clusters, edgelist
    """
    networkgraph = NetworkGraph(args,data,templates)
    networkgraph.plot_networkx(args,templates)

    return networkgraph

def main():
    args = Arguments()

    if args.args.templatize:
        if args.args.stopwords == None:
            raise Exception('Stopwords file is missing. The stopwords file is required for templatization.')

        templates = Templates(args.args.stopwords, args.file_args, args.args.temporal)
    else:
        templates = None

    data,templates = get_info(args, templates)
    networkgraph = draw_networkx(args,data,templates)

    ## save NetworkX graph position file to pickle file
    if (args.args.position_file):
        pickle.dump(networkgraph.pos, open(args.args.position_file, "wb" ))

    ## save labels dict to pickle file
    if (args.args.labels_file):
        if (args.args.template_nodes):
            output_labels = {cmd:{'label':label, 'template':templates.cmd2template[cmd]} for cmd,label in networkgraph.labels.items()}
            pickle.dump(output_labels, open(args.args.labels_file, "wb"))
        else:
            output_labels = {cmd:{'label':label, 'template':"N/A"} for cmd,label in networkgraph.labels.items()}

    ## create edge list to FSDB file
    if (args.args.edge_list):
        if (args.args.template_nodes): ## if template nodes are being graphed, produce edge list that includes templates
            print("Graphing template nodes...")
            outh = pyfsdb.Fsdb(out_file=args.args.edge_list)
            outh.out_column_names=['cluster_id', 'weight', 'node1_id', 'node2_id', 'node1', 'node2', 'template1', 'template2']
            for cmd1,cmd2,weight in networkgraph.weighted_edges:
                cluster_id = networkgraph.clusters[cmd1]
                num1 = networkgraph.labels[cmd1]
                num2 = networkgraph.labels[cmd2]
                template1 = ' '.join(templates.cmd2template[cmd1])
                template2 = ' '.join(templates.cmd2template[cmd2])
                outh.append([cluster_id,round(weight,3),num1,num2,cmd1,cmd2,template1,template2])
            outh.close()
        else:
            outh = pyfsdb.Fsdb(out_file=args.args.edge_list)
            outh.out_column_names=['cluster_id', 'node1_id', 'node2_id', 'node1', 'node2', 'weight']
            for cmd1,cmd2,weight in networkgraph.weighted_edges:
                cluster_id = networkgraph.clusters[cmd1]
                num1 = networkgraph.labels[cmd1]
                num2 = networkgraph.labels[cmd2]
                outh.append([cluster_id,num1,num2,cmd1,cmd2,round(weight,3)])
            outh.close()

    ## create cluster list to FSDB file
    if (args.args.cluster_list):
        if (args.args.template_nodes): ## if template nodes are being graphed, produce edge list that includes templates
            outh = pyfsdb.Fsdb(out_file=args.args.cluster_list)
            outh.out_column_names=['cluster_id','command','template']
            for cmd,cluster_id in networkgraph.clusters.items():
                template = ' '.join(templates.cmd2template[cmd])
                outh.append([cluster_id,cmd,template])
            outh.close()
        else:
            outh = pyfsdb.Fsdb(out_file=args.args.cluster_list)
            outh.out_column_names=['cluster_id','command']
            for cmd,cluster_id in networkgraph.clusters.items():
                outh.append([cluster_id,cmd])
            outh.close()

    ## create template list to FSDB file
    if (args.args.template_list):
        if (args.args.template_nodes): ## if template nodes are being graphed, produce template list that contains template, example command, and label
            outh = pyfsdb.Fsdb(out_file=args.args.template_list)
            outh.out_column_names=['template','command','node','label']
            template_list = []
            for cmd in networkgraph.clusters.keys():
                template = ' '.join(templates.cmd2template[cmd])
                node = networkgraph.labels[cmd]
                if data.sourceDic != {}:
                    label = data.sourceDic[cmd]
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
            for temp,cmds in templates.template2cmd.items():
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