# Provone

Files in the repo:
    <li> <code>graph_utils.py</code> that contains the final function that will be used to pre-process the captions</li>
    <li> <code>produce_graph_libs.py</code> that is used as demo to understand graphs from senteces</li>
    <li> <code>extract_triplets.py</code> file used to extract triplets from captions</li>
    <li> <code>train.py</code> file containing the training procedures</li>
    <li> <code>main.py</code> file containing the main (for now only for the first part of the model)</li>
    <li> <code>models.py</code> contains the model definition for triplets classification and the second part of the task(for now)</li>
    <li> <code>gnn.py</code> contains the model definition for the Graph neural network (for now)</li>
    <li> <code>dataset.py</code> contains the definition for the dataset class for both tasks</li>

To Fix:
    <li> Fix the dataset usage for both tasks (mainly the contructor)</li>
    <li> Create a <code>collate_fn</code> to be used in both tasks</li>
    <li> Try the GNN network and decoder for a batched input</li>
