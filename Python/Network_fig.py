# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 07:17:34 2021

@author: linde
"""

import networkx as nx
import matplotlib.pyplot as plt
import graphviz


## Graph Params

hidden_layers = 3
layer_nodes = 1

## Initialize Nodes

dot = graphviz.Digraph(comment = "A Neural Network")


s = graphviz.Digraph('sub')


s.attr(rank= 'same')


for n in range(hidden_layers):
    s.node("Gene "+ str(n+1))
s.node("Cooperation", shape = 'box')



## Initialize Edges 

for n in range(hidden_layers):
    s.edge("Input", "Gene " + str(n+1))
    s.edge("Gene " +str(n+1), "Cooperation")
    for n2 in range(hidden_layers):
        if n2 > n:
            s.edge("Gene "+ str(n+1), "Gene " +str(n2+1))

s.edge("Input", "Cooperation")



## Render Properties

dot.subgraph(s)
dot.edge_attr.update(arrowhead='vee')

## Render Call
dot.graph_attr['rankdir'] = 'LR'
# dot.graph_attr['constraint'] = "false"

dot.render("output.gv", view =True, format = "png")




# %%

# %%
