from graph_utils import load_json, tripl2list
from transformers import BertModel, BertTokenizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import json


data = load_json('triplets_ucm.json')
triplets = list(data['Triplet_to_idx'].keys())
model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print("Total number of initial triplets: ", len(triplets))

idx2tripl = {v: k for k, v in data['Triplet_to_idx'].items()}

tripl_feats = []
for tripl in triplets:
    # Extract features from triplets
    tripl = tripl.replace('(', '')
    tripl = tripl.replace(')', '')
    tripl = tripl.replace("'", '')
    tripl = tripl.replace(",", '')
    tripl = tripl.strip()
    encoded_input = tokenizer(tripl, return_tensors='pt', add_special_tokens=True, padding=True)
    output = model(**encoded_input)
    tripl_feats.append(output.pooler_output[0].detach().numpy())


print("Initial feature size: ", (len(tripl_feats), len(tripl_feats[0]))) # (number_triplets, feature_shape)
pca = PCA(n_components=100, random_state=22)
pca.fit(tripl_feats)
x = pca.transform(tripl_feats)
print(f"Components before PCA: {len(tripl_feats[0])}")
print(f"Components after PCA: {pca.n_components}")

clusters = 100
clustering = KMeans(n_clusters=clusters)
clustering.fit(x)

distances = pairwise_distances_argmin_min(clustering.cluster_centers_, x, metric='euclidean')
print("Nearest triplets: ", distances[0])
# holds the cluster id and the images { id: [images] }
groups = {}
for file, cluster in zip(triplets, clustering.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(tripl2list(file))
    else:
        groups[cluster].append(tripl2list(file))

centroid2tripl = {}
for id in groups:
    centroid2tripl[idx2tripl[distances[0][id]]] = groups[id]

for k,v in centroid2tripl.items():
    print("{}: {}".format(k, v))

with open('centroid2triplet.json', 'w') as f:
    json.dump(centroid2tripl, f)