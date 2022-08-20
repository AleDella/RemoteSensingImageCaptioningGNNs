####
# This file is created due to the heavyness of the computations for creating a large dataset using graphs
####
import json
import torch
import time
from transformers import BertTokenizer, BertModel

f = open('dataset/RSICD_dataset/triplets_rsicd.json', 'r')
all_triplets = json.load(f)
f.close()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

total_time = time.time()

for split in all_triplets.keys():
    if str(split) != 'discarded_images':
        split_time = time.time()
        triplets = all_triplets[split]
        node_feats = {}
        num_nodes = {}
        src_ids = {}
        dst_ids = {}
        # Here to check what happen when split is not passed
        for id in triplets:
            f_tripl = []
            tmp_dict = {}
            tmp_id = 0
            tmp_src_ids = []
            tmp_dst_ids = []
            tmp_node_feats = []
            # Extract features from triplets
            for _, tripl in enumerate(triplets[id]):
                encoded_input = tokenizer(tripl, return_tensors='pt', add_special_tokens=False, padding=True)
                output = model(**encoded_input)
                f_tripl.append(output.pooler_output)
                if tripl[0] not in list(tmp_dict.keys()):
                    tmp_dict[tripl[0]]=tmp_id
                    tmp_id+=1
                    tmp_node_feats.append(list(output.pooler_output[0]))
                if tripl[1] not in list(tmp_dict.keys()):
                    tmp_dict[tripl[1]]=tmp_id
                    tmp_id+=1
                    tmp_node_feats.append(list(output.pooler_output[1]))
                if tripl[2] not in list(tmp_dict.keys()):
                    tmp_dict[tripl[2]]=tmp_id
                    tmp_id+=1
                    tmp_node_feats.append(list(output.pooler_output[2]))
                
                # Create source and destination lists
                tmp_src_ids.append(tmp_dict[tripl[0]])
                tmp_dst_ids.append(tmp_dict[tripl[1]])
                tmp_src_ids.append(tmp_dict[tripl[1]])
                tmp_dst_ids.append(tmp_dict[tripl[2]])
            src_ids[id] = tmp_src_ids
            dst_ids[id] = tmp_dst_ids
            node_feats[id] = torch.Tensor(tmp_node_feats).numpy().tolist()
            num_nodes[id] = len(tmp_node_feats)

        # Write onto files
        with open('src_ids_' + str(split) + '.json', 'w') as f:
            json.dump(src_ids, f)
        with open('dst_ids_' + str(split) + '.json', 'w') as f:
            json.dump(dst_ids, f)
        with open('node_feats_' + str(split) + '.json', 'w') as f:
            json.dump(node_feats, f)
        with open('num_nodes_' + str(split) + '.json', 'w') as f:
            json.dump(num_nodes, f)
        print("{} split done! Total time: {}".format(str(split), (time.time()-split_time)))

print("Done everything! Total time: {}".format((time.time()-total_time)))
