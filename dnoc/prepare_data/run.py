import json
import os
with open('mscoco/noc_coco_cap.json', 'r') as fin:
  data = json.load(fin)
del data['train']
del data['test']
data['test'] = []
for files in os.listdir('mscoco/extracted_cnn_feature/'):
  data['test'].append({'vname':files[:-4]})
 
with open('mscoco/noc_coco_cap.json', 'w') as fp:
    json.dump(data, fp)
