import json
import sng_parser
from tqdm import tqdm

def extract_ent(sentences):
    final_input = []
    for sentence in sentences:
        g = sng_parser.parse(sentence['raw'])
        # Get the tokenization like in the graph
        for rel in g['relations']:
            final_input.append((g['entities'][rel['subject']]['head'], rel['relation'], g['entities'][rel['object']]['head']))
    return final_input

def extract_triplets(sentence):
    # Extract triplets from one sentence
    final_input = []
    g = sng_parser.parse(sentence)
    # Get the tokenization like in the graph
    for rel in g['relations']:
        final_input.append((g['entities'][rel['subject']]['head'], rel['relation'], g['entities'][rel['object']]['head']))
    return final_input

# FOR RSICD

# anno_path = "Provone/dataset/dataset_rsicd.json"
# f = open(anno_path)
# anno = json.load(f)
# f.close()
# anno = anno['images']
# final_file = {'annotations': []}
# final_labels = []

# for img in anno:
#     tripl = list(set(extract_ent(img['sentences'])))
#     if tripl not in final_labels:
#         final_labels.append(tripl)
#     x = {'imgid': img['imgid'], 'triplets': tripl}
#     final_file['annotations'].append(x)

# with open("Provone/triplets.json", "w") as outfile:
#     json.dump(final_file, outfile)

# print("Total number of unique triplets found: ", len(final_labels))
def readfile(path):
    with open(path,'r') as file:
        text = file.readlines()
    return text


# FOR UCM
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
    triplets = extract_triplets(annot)
    if(image in train_list):
        try:
            final_file["train"][image].append(triplets)
        except:
            final_file["train"][image] = triplets
    elif(image in test_list):
        try:
            final_file["test"][image].append(triplets)
        except:
            final_file["test"][image] = triplets
    elif(image in val_list):
        try:
            final_file["val"][image].append(triplets)
        except:
            final_file["val"][image] = triplets
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


with open("triplets.json", "w") as outfile:
    json.dump(final_file, outfile)