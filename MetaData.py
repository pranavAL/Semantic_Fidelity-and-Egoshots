
import pandas as pd
from textblob import TextBlob
import spacy
nlp = spacy.load('en_core_web_lg')


captions = pd.read_csv('Captions.csv')
objects = pd.read_csv('Objects.csv')


Egoshots_metadata = {}


def detect_noun(sentence):
    
    blob = TextBlob(str(sentence))
    list = []
    
    for word, tag in blob.tags:
        if tag=='NN' and word!='['and word!=']':
            list.append(word)
       
    return list  


def cosine_similarity(nouns,objects):
    
    listNouns = ' '.join(nouns[i] for i in range(len(nouns)))
    countNouns = len(nouns)
    
    obj = [item.replace('[','').replace(']','').replace("'",'') for item in list(objects.split(','))]
    listObjects = ' '.join(obj[i] for i in range(len(obj)))
    countObjects = len(obj)
    
    noun_emb = nlp(listNouns)
    obj_emb = nlp(listObjects)
    
    sim = noun_emb.similarity(obj_emb)
        
    return sim, countNouns, countObjects


def Semantic_Fidelity(cosine, countNouns, countObjects):
    
    metric1 = cosine ** (countObjects / countNouns)
    metric2 = cosine ** ((countObjects / countNouns)/countObjects**(0.5)+1)
    metric3 = cosine * ((countObjects / countNouns)/countObjects+1)
    metric4 = cosine * ((countObjects / countNouns)/countObjects**(0.5)+1)
    
    return metric1, metric2, metric3, metric4
    

def create_meta_data(captions, objects):
    
    meta_data = pd.DataFrame(columns = ['Images','Nouns_SAT','Nouns_NOC','Nouns_DNOC','Objects_Y9','SF_1_SAT',
                                                 'SF_2_SAT','SF_3_SAT','SF_4_SAT','SF_1_NOC',
                                                 'SF_2_NOC','SF_3_NOC','SF_4_NOC','SF_1_DNOC',
                                                 'SF_2_DNOC','SF_3_DNOC','SF_4_DNOC'])
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    list7 = []
    list8 = []
    list9 = []
    list10 = []
    list11 = []
    list12 = []
    list13 = []
    list14 = []
    list15 = []
    list16 = []
    list17 = []
    
    
    for index,image in enumerate(objects['ImageFiles']):
        
        index_of_caption = captions.index[captions['ImageFiles'] == image]
        
        list1.append(image)
        
        list2.append(detect_noun(list(captions['Show Attend And Tell'][index_of_caption])))
        list3.append(detect_noun(list(captions['Novel Object Captioner'][index_of_caption])))
        list4.append(detect_noun(list(captions['Decoupled Novel Object Captioner'][index_of_caption])))
        
        list5.append(objects['YOLO9000'][index])
        
        similarity, countNouns, countObjects = cosine_similarity(list2[index], objects['YOLO9000'][index])
        
        SF_1, SF_2, SF_3, SF_4 = Semantic_Fidelity(similarity, countNouns, countObjects)
        if SF_3 > 1.0:
            SF_3 = 1.0
        
        if SF_4 > 1.0:
            SF_4 = 1.0
                     
        list6.append(SF_1)
        list7.append(SF_2)
        list8.append(SF_3)
        list9.append(SF_4)
        
        similarity, countNouns, countObjects = cosine_similarity(list3[index], objects['YOLO9000'][index])
        
        SF_1, SF_2, SF_3, SF_4 = Semantic_Fidelity(similarity, countNouns, countObjects)
        if SF_3 > 1.0:
            SF_3 = 1.0
            
        if SF_4 > 1.0:
            SF_4 = 1.0
                     
        list10.append(SF_1)
        list11.append(SF_2)
        list12.append(SF_3)
        list13.append(SF_4)
        
        similarity, countNouns, countObjects = cosine_similarity(list4[index], objects['YOLO9000'][index])
        
        SF_1, SF_2, SF_3, SF_4 = Semantic_Fidelity(similarity, countNouns, countObjects)
        if SF_3 > 1.0:
            SF_3 = 1.0
        if SF_4 > 1.0:
            SF_4 = 1.0
                     
        list14.append(SF_1)
        list15.append(SF_2)
        list16.append(SF_3)
        list17.append(SF_4)
        
    
    meta_data['Images'] = list1
        
    meta_data['Nouns_SAT'] = list2
    meta_data['Nouns_NOC'] = list3
    meta_data['Nouns_DNOC'] = list4
        
    meta_data['Objects_Y9'] = list5
      
    meta_data['SF_1_SAT'] = list6
    meta_data['SF_2_SAT'] = list7
    meta_data['SF_3_SAT'] = list8
    meta_data['SF_4_SAT'] = list9
    
    meta_data['SF_1_NOC'] = list10
    meta_data['SF_2_NOC'] = list11
    meta_data['SF_3_NOC'] = list12
    meta_data['SF_4_NOC'] = list13
    
    meta_data['SF_1_DNOC'] = list14
    meta_data['SF_2_DNOC'] = list15
    meta_data['SF_3_DNOC'] = list16
    meta_data['SF_4_DNOC'] = list17
        
    return meta_data
        

meta_data = create_meta_data(captions,objects)

HSF_COCO = pd.read_csv('COCO_HSF.csv')

for index in range(len(HSF_COCO['Images'])):
    
    nouns = detect_noun(HSF_COCO['Captions'][index])
    coco_objects = HSF_COCO['Objects'][index]
    
    similarity, countNouns, countObjects = cosine_similarity(nouns, coco_objects)
    
    SF_1, SF_2, SF_3, SF_4 = Semantic_Fidelity(similarity, countNouns, countObjects)
    if SF_3 > 1.0:
            SF_3 = 1.0
            
    if SF_4 > 1.0:
            SF_4 = 1.0
    
    HSF_COCO['SF_1'][index] = SF_1
    HSF_COCO['SF_2'][index] = SF_2
    HSF_COCO['SF_3'][index] = SF_3
    HSF_COCO['SF_4'][index] = SF_4

meta_data.to_csv('Meta-data.csv')
HSF_COCO.to_csv('SF_HSF_COCO.csv')

