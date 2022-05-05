# distance-graph

Calculates the Levenshtein distance between nodes and plots a weighted network graph.

## Example Usage
`python distance-graph/run.py -1 example/data1.fsdb command data1 ip -2 example/data2.fsdb commands data2 ip -3 example/data3.fsdb cmd data3 id -on command source ip --templatize -stop example/nix_commands.txt -e 0.05 -w 10 -H 8 -fs 10 -ns 250 -E edgelist.fsdb -c clusterlist.fsdb distance-graph.png`
