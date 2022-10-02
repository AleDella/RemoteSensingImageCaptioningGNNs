import json
import sng_parser
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Extract triplets from dataset's captions.")
parser.add_argument('--dataset', default='ucm', required=True, help='name of the dataset of which you want to create the triplets.')

def extract_ent(sentences):
    final_input = []
    for sentence in sentences:
        g = sng_parser.parse(sentence['raw'])
        # Get the tokenization like in the graph
        for rel in g['relations']:
            final_input.append((g['entities'][rel['subject']]['lemma_span'], rel['relation'], g['entities'][rel['object']]['lemma_span']))
    return final_input

def extract_triplets(sentence):
    
    sentence = ' '.join(sentence)
    # Extract triplets from one sentence
    final_input = []
    g = sng_parser.parse(sentence)
    
    # Get the tokenization like in the graph
    if g['relations'] == []:
        for ent in g['entities']:
            final_input.append(ent['lemma_span'])
    else:
        for rel in g['relations']:
            final_input.append((g['entities'][rel['subject']]['lemma_span'], rel['relation'], g['entities'][rel['object']]['lemma_span']))
        
    
    return final_input

def readfile(path):
    with open(path,'r') as file:
        text = file.readlines()
    return text


# FOR RSICD

def rsicd(path_rsicd, json_name):
    f = open(path_rsicd)
    anno = json.load(f)
    f.close()
    anno = anno['images']
    final_file = {'train': {}, 'test': {}, 'val': {}, 'discarded_images': []}

    for img in anno:
        tripl = list(set(extract_ent(img['sentences'])))
        if tripl!=[]:
            try:
                final_file[img['split']][img['imgid']].append(tripl)
            except:
                final_file[img['split']][img['imgid']] = tripl
        else:
            final_file['discarded_images'].append(img['filename'])

    with open(json_name, "w") as outfile:
        json.dump(final_file, outfile)




# # FOR UCM
def ucm(path_annots, train, test, val, json_name):
    path_annots = "D:/Alessio/Provone/dataset/UCM_dataset/filenames/descriptions_UCM.txt"
    train = "D:/Alessio/Provone/dataset/UCM_dataset/filenames/filenames_train.txt"
    test  = "D:/Alessio/Provone/dataset/UCM_dataset/filenames/filenames_test.txt"
    val = "D:/Alessio/Provone/dataset/UCM_dataset/filenames/filenames_val.txt"

    vocabulary = dict()

    annotations = readfile(path_annots)
    train = readfile(train)
    test = readfile(test)
    val = readfile(val)

    train_list = []
    test_list = []
    val_list = []

    for img in train:
        train_list.append(img.split(".")[0])

    for img in test:
        test_list.append(img.split(".")[0])

    for img in val:
        val_list.append(img.split(".")[0])

    # Create vocabulary
    for annot in annotations:
        tokens = annot.split(" ")[1:]
        for token in tokens:
            try:
                vocabulary[token]+=1
            except:
                vocabulary[token]=1

    # Filter words with occurrence less than 5 
    words_to_keep = [key for key in vocabulary.keys() if vocabulary[key]>=1]
    words_to_discard = [key for key in vocabulary.keys() if key not in words_to_keep]

    # Create the triplets and store them in the dictionary
    final_file = {'train': {}, 'test': {}, 'val': {}}
    all_triplets = []
    for annot in tqdm(annotations):
        image = annot.split(" ")[0]
        triplets = extract_triplets(annot.split(" ")[1:])
        if(image in train_list):
            try:
                final_file["train"][image].append(triplets)
            except:
                final_file["train"][image] = [triplets]
        elif(image in test_list):
            try:
                final_file["test"][image].append(triplets)
            except:
                final_file["test"][image] = [triplets]
        elif(image in val_list):
            try:
                final_file["val"][image].append(triplets)
            except:
                final_file["val"][image] = [triplets]
        else:
            raise Exception("Something went wrong, find an image not assigned to any of train test or val splits")     
        
        for triplet in triplets:
            all_triplets.append(triplet)
        
    all_triplets = set(all_triplets)


    print("Total number of unclean triplets found: ", len(all_triplets))

    triplet_to_idx = {}
    for i, triplet in enumerate(all_triplets):
        triplet_to_idx[str(triplet)] = i  # cast to string to make it become dictionary key

    final_file['Triplet_to_idx'] = triplet_to_idx

    # Cleaning triplets
    clean_triplets = []
    for triplet in all_triplets:
        discard = False
        for word in triplet:
            if word in words_to_discard:
                discard = True
                break
        if(not discard):
            clean_triplets.append(triplet)

    print("Total number of cleaned triplets found: ", len(clean_triplets))


    with open(json_name, "w") as outfile:
        json.dump(final_file, outfile)
        
        
        
if __name__== "__main__":
    args = parser.parse_args()
    if args.dataset == 'ucm':
        path_ucm = "dataset/UCM_dataset/filenames/descriptions_UCM.txt"
        train_ucm = "dataset/UCM_dataset/filenames/filenames_train.txt"
        test_ucm  = "dataset/UCM_dataset/filenames/filenames_test.txt"
        val_ucm = "dataset/UCM_dataset/filenames/filenames_val.txt"
        json_ucm = 'triplets_ucm.json'
        ucm(path_ucm, train_ucm, test_ucm, val_ucm, json_ucm)
    elif args.dataset == 'rsicd':
        path_rsicd = "dataset/RSICD_dataset/dataset_rsicd.json"
        json_rsicd = 'triplets_rsicd.json'
        rsicd(path_rsicd, json_rsicd)
    else:
        print("Dataset not implemented.")