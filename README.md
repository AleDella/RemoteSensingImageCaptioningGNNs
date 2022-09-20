# Provone

## Basic usage of the project:

<code> python run.py<code>

## Possible arguments:

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

# TODO LIST 
- [X] do the dictionary of the captions for UCM and RSICD (key: image_id, value: caption, list of tokens)
- [X] do the complete pipeline (from image to caption) train and validation 
- [X] performance of the triplet classifier
- [ ] find a method to improve classifier
- [X] Begin the presentation (introduction to GNN)


