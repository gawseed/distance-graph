from tools.arguments import *
from tools.templatizer import *
from tools.templates import Templates
from tools.data import Data
from tools.graph import NetworkGraph
import pyfsdb
import pickle


def get_info2(args, cmd2template):
    templates_class = cmd2template
    data = Data(args)

    if cmd2template:
        data.find_unique_templatized_cmds(args, cmd2template, args.args.temporal)

        cmd2template = cmd2template.cmd2template

        if args.args.template_nodes:
            data.get_template_nodes(templates_class, args)
            templates_class.calculate_template_counts(data.df, args.node_name, data.unique_cmds)
            
            if args.args.labels:
                labels = pickle.load(open(args.args.labels,"rb"))
                data.update_representative_cmd(labels,templates_class)

                templates_class.cmd2template_count = data.remap_dic(templates_class.cmd2template_count, data.cmd_to_old_label, 'cmd')
    
    data.get_unique_cmds()
    data.calculate_weights()
    data.update_source_dic(args)

    return data,templates_class

def draw_networkx2(args,data,templates):
    networkgraph = NetworkGraph(args,data,templates)

    labels = networkgraph.labels
    clusters = networkgraph.clusters

    G = networkgraph.G
    networkgraph.plot_networkx(args, templates)
    pos = networkgraph.pos

    return G,networkgraph.weighted_edges,labels,clusters,pos

def main():
    args = Arguments()

    if args.args.templatize:
        if args.args.stopwords == None:
            raise Exception('Stopwords file is missing. The stopwords file is required for templatization.')

        cmd2template = Templates(args.args.stopwords, args.file_args, args.args.temporal)
    else:
        cmd2template = None

    data,templates_class = get_info2(args, cmd2template)
    G,weighted_edges,labels,clusters,pos = draw_networkx2(args,data,templates_class)

    ## save NetworkX graph position file to pickle file
    if (args.args.position_file):
        pickle.dump(pos, open(args.args.position_file, "wb" ))

    ## save labels dict to pickle file
    if (args.args.labels_file):
        if (args.args.template_nodes):
            output_labels = {cmd:{'label':label, 'template':templates_class.cmd2template[cmd]} for cmd,label in labels.items()}
            pickle.dump(output_labels, open(args.args.labels_file, "wb"))
        else:
            output_labels = {cmd:{'label':label, 'template':"N/A"} for cmd,label in labels.items()}

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
                template1 = ' '.join(templates_class.cmd2template[cmd1])
                template2 = ' '.join(templates_class.cmd2template[cmd2])
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
                template = ' '.join(templates_class.cmd2template[cmd])
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
                template = ' '.join(templates_class.cmd2template[cmd])
                node = labels[cmd]
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
            for temp,cmds in templates_class.template2cmd.items():
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