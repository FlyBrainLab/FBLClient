import json
import networkx as nx
import numpy as np


def diagram_generator(
    X, log, prog="dot", method=None, splines="line", rename=False, styles=None, **kwargs
):

    name = X["name"]
    G = X["graph"]

    if "method" in X.keys():
        method = X["method"]
    if "splines" in X.keys():
        splines = X["splines"]
    if "prog" in X.keys():
        prog = X["prog"]
    if "styles" in X.keys():
        styles = X["styles"]
    if "rename" in X.keys():
        rename = X["rename"]

    with open("_temp_model.gexf", "w") as file:
        file.write(G)

    G = nx.read_gexf("_temp_model.gexf")
    log.info("Circuit loaded for visualization...")
    log.info("Name: " + str(name))
    log.info("Number of Nodes: " + str(nx.number_of_nodes(G)))
    log.info("Number of Edges: " + str(nx.number_of_edges(G)))

    log.info("method: " + str(method))
    log.info("splines: " + str(splines))
    log.info("prog: " + str(prog))
    log.info("styles: " + str(styles))
    log.info("rename: " + str(rename))

    if method == None:
        if nx.number_of_edges(G) < 200:
            method = "sugiyama"
        else:
            method = "force_directed"
    if styles is None:
        if method == "sugiyama":
            size_mult = str(int(np.round(128.0 * np.sqrt(nx.number_of_edges(G)))))
            styles = {
                "graph": {
                    "label": name,
                    "fontsize": "16",
                    "fontcolor": "black",
                    "outputorder": "edgesfirst",
                    "splines": splines,
                    "model": "dot",
                    "size": size_mult + "," + size_mult,
                    "overlap": "false",
                    "bgcolor": "transparent",
                },
                "nodes": {
                    "shape": "box",
                    "fontcolor": "black",
                    "color": "black",
                    "style": "filled",
                    "fillcolor": "white",
                },
                "edges": {
                    "style": "solid",
                    "color": "black",
                    "arrowhead": "open",
                    "arrowsize": "0.5",
                    "fontname": "Courier",
                    "fontsize": "12",
                    "fontcolor": "black",
                    "splines": "ortho",
                    "concentrate": "false",
                },
            }
            log.info("Performing Sugiyama layouting.")
        if method == "force_directed":
            prog = "sfdp"
            size_mult = str(int(np.round(128.0 * np.sqrt(nx.number_of_edges(G)))))
            log.info("Diagram Size (Expected): " + size_mult + "," + size_mult)
            styles = {
                "graph": {
                    "label": name,
                    "fontsize": "12",
                    "fontcolor": "black",
                    "outputorder": "edgesfirst",
                    "splines": splines,
                    "model": "neato",
                    "size": size_mult + "," + size_mult,
                    "overlap": "scale",
                    "bgcolor": "transparent",
                },
                "nodes": {
                    "shape": "circle",
                    "fontcolor": "black",
                    "color": "black",
                    "style": "filled",
                    "fillcolor": "white",
                },
                "edges": {
                    "style": "solid",
                    "color": "black",
                    "arrowhead": "open",
                    "arrowsize": "0.5",
                    "fontname": "Courier",
                    "fontsize": "12",
                },
            }
            log.info("Performing SFDP layouting.")
    else:
        log.info("Performing custom layouting.")
    # log.info('Circuit loaded for visualization...')
    # G = self.G
    # G.remove_nodes_from(list(nx.isolates(G)))
    if rename == True:
        mapping = {}
        node_types = set()
        for n, d in G.nodes_iter(data=True):
            node_types.add(d["name"].rstrip("1234567890"))
        node_nos = dict.fromkeys(node_types, 1)
        for n, d in G.nodes_iter(data=True):
            node_type = d["name"].rstrip("1234567890")
            mapping[n] = d["name"].rstrip("1234567890") + str(node_nos[node_type])
            node_nos[node_type] += 1
        G = nx.relabel_nodes(G, mapping)
    # nx.write_dot(G,"grid.dot")
    A = nx.drawing.nx_agraph.to_agraph(G)
    A.graph_attr.update(styles["graph"])
    A.write("file.dot")
    for i in A.edges():
        e = A.get_edge(i[0], i[1])
        # e.attr['splines'] = 'ortho'
        e.attr.update(styles["edges"])
        if i[0][:-1] == "Repressor":
            e.attr["arrowhead"] = "tee"
    for i in A.nodes():
        n = A.get_node(i)
        # print(n)
        # n.attr['shape'] = 'box'
        n.attr.update(styles["nodes"])
    A.layout(prog=prog)
    A.draw("_temp_circuit.svg")
    A.draw("_temp_circuit.eps")
    A.draw("_temp_circuit.dot")
    A.draw("../web/data/" + name + "_visual.svg")
    A.draw("../web/data/" + name + "_visual.eps")
    A.draw("../web/data/" + name + "_visual.dot")
    log.info("Diagram visualization complete.")
    output = {}
    output["success"] = True
    with open("_temp_circuit.dot", "r") as file:
        output["dot"] = file.read()
    with open("_temp_circuit.svg", "r") as file:
        output["svg"] = file.read()
    return output
