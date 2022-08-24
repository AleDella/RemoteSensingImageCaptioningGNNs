# Provone

Basic usage of the project:
<code> python run.py<code>

Possible arguments:
<li> <code> --dataset </code>: name of the dataset used for the run (currently "ucm" or "rsicd")</li>
<li> <code> --task </code>: name of the desired task (currently "tripl2caption" or "img2tripl")</li>
<li> <code> --e </code>: number of training epochs</li>
<li> <code> --lr </code>: learning rate</li>
<li> <code> --bs </code>: batch size</li>
<li> <code> --decoder </code>: type of decoder used for GNNs (currently "linear" or "lstm")</li>
<li> <code> --name </code>: name of the file of the network</li>
<li> <code> --es </code>: allow usage of early stopping</li>
<li> <code> --thresh </code>: threshold of early stopping</li>
<li> <code> --test </code>: do the run in test mode (currently implemented only with "tripl2caption")</li>
<li> <code> --o </code>: name of the file with the results of the testing</li>

To Do (ordered in priority):
<li> implement and test the addition of image features to the gnn</li>
<li> implement and test the MLAP model for the gnn part</li>
<li> do the evaluation for the classification part (Maybe)</li>
<li> do the train/eval for the whole pipeline (Maybe) </li>