import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str)
args = parser.parse_args()

captions = pd.read_csv('Captions.csv')
metrics = pd.read_csv('Meta-data.csv')

test_image = args.image
index_to_image = list(captions.index[captions['ImageFiles']==test_image])[0]

SAT = captions['Show Attend And Tell'][index_to_image]
NOC = captions['Novel Object Captioner'][index_to_image]
DNOC = captions['Decoupled Novel Object Captioner'][index_to_image]

index_to_metric = list(metrics.index[metrics['Images']==test_image])[0]

SAT_SF = metrics['SF_11_SAT'][index_to_metric]
NOC_SF = metrics['SF_11_NOC'][index_to_metric]
DNOC_SF = metrics['SF_11_DNOC'][index_to_metric]

captions = [SAT,NOC,DNOC]
SF = [SAT_SF,NOC_SF,DNOC_SF]

final_index = sorted(range(len(SF)), key=lambda i:SF[i], reverse=True)

print('TOP-3 Captions for the image - ',test_image,'\n')
for i,index in enumerate(final_index):
    print(str(i+1)+'.\t',captions[index],'\t','SF = ',round(SF[index],2),'\n')
