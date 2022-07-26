import sng_parser
import dgl
import networkx as nx
import matplotlib.pyplot as plt


# Function that takes a sentence and produce the dgl graph of it
def sentence_to_dgl(sentence, visualize=True):
    # Create the graph of the sentence
    graph = sng_parser.parse(sentence)
    
    
    # Create the networkx graph to visualize it
    G = nx.Graph()
    for rel in graph['relations']:
        # Add the nodes (Just two because in each relation there is only one object and one subject)
        G.add_node(graph['entities'][rel['subject']]['head'])
        G.add_node(graph['entities'][rel['object']]['head'])
        # Add the relation between the two
        G.add_edge(graph['entities'][rel['subject']]['head'],graph['entities'][rel['object']]['head'], relation=rel['relation'])
        if visualize:
            pos = nx.spring_layout(G)
            # Add the edge label
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels={(graph['entities'][rel['subject']]['head'], graph['entities'][rel['object']]['head']): rel['relation'] },
                font_color='red'
                )
    if visualize:
        nx.draw(G, with_labels= True)
        plt.show()
    # Return the dgl network
    return dgl.from_networkx(G)

############# TEST CODE #######################

# Define the sentence
sentence = 'a river goes through this area with a forest on one bank and a road on the other side'

# Check the table to see the resulting graph for the sentence
print("Right relations: ")
# Parse the sentence
graph = sng_parser.parse(sentence)
sng_parser.tprint(graph)

result = sentence_to_dgl(sentence, visualize=True)
