
import pandas as pd
from textblob import TextBlob
import spacy
import numpy as np
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
    x = countNouns / countObjects
    metric1 = cosine ** (1 - x)
    metric2 = cosine ** ((1 - x)/(max(countObjects,countNouns)+1))
    metric3 = cosine ** ((1 - x)/(max(countObjects,countNouns)**(0.5)+1))
    metric4 = cosine * x
    metric5 = cosine * (x /(max(countObjects,countNouns)+1))
    metric6 = cosine * (x/(max(countObjects,countNouns)**(0.5)+1))
    metric7 = cosine * (1 - 1 / (1 + x))
    metric8 = (1 - np.exp(-cosine ** (1-x)))
    metric9 = cosine * (1 - np.exp(-x))
    metric10 = (1 - np.exp(-cosine * x))

    return metric1, metric2, metric3, metric4, metric5, metric6, metric7, metric8, metric9, metric10


def create_meta_data(captions, objects):

    meta_data = pd.DataFrame(columns = ['Images','Nouns_SAT','Nouns_NOC','Nouns_DNOC','Objects_Y9',
                            'SF_1_SAT','SF_2_SAT','SF_3_SAT','SF_4_SAT','SF_5_SAT','SF_6_SAT','SF_7_SAT','SF_8_SAT',
                            'SF_1_NOC','SF_2_NOC','SF_3_NOC','SF_4_NOC','SF_5_NOC','SF_6_NOC','SF_7_NOC','SF_8_NOC',
                            'SF_1_DNOC','SF_2_DNOC','SF_3_DNOC','SF_4_DNOC','SF_5_DNOC','SF_6_DNOC','SF_7_DNOC','SF_8_DNOC'])
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
    list18 = []
    list19 = []
    list20 = []
    list21 = []
    list22 = []
    list23 = []
    list24 = []
    list25 = []
    list26 = []
    list27 = []
    list28 = []
    list29 = []

    for index,image in enumerate(objects['ImageFiles']):

        index_of_caption = captions.index[captions['ImageFiles'] == image]

        list1.append(image)

        list2.append(detect_noun(list(captions['Show Attend And Tell'][index_of_caption])))
        list3.append(detect_noun(list(captions['Novel Object Captioner'][index_of_caption])))
        list4.append(detect_noun(list(captions['Decoupled Novel Object Captioner'][index_of_caption])))

        list5.append(objects['YOLO9000'][index])

        similarity, countNouns, countObjects = cosine_similarity(list2[index], objects['YOLO9000'][index])

        SF_1, SF_2, SF_3, SF_4, SF_5, SF_6, SF_7, SF_8 = Semantic_Fidelity(similarity, countNouns, countObjects)

        list6.append(SF_1)
        list7.append(SF_2)
        list8.append(SF_3)
        list9.append(SF_4)
        list10.append(SF_5)
        list11.append(SF_6)
        list12.append(SF_7)
        list13.append(SF_8)

        similarity, countNouns, countObjects = cosine_similarity(list3[index], objects['YOLO9000'][index])

        SF_1, SF_2, SF_3, SF_4, SF_5, SF_6, SF_7, SF_8 = Semantic_Fidelity(similarity, countNouns, countObjects)

        list14.append(SF_1)
        list15.append(SF_2)
        list16.append(SF_3)
        list17.append(SF_4)
        list18.append(SF_5)
        list19.append(SF_6)
        list20.append(SF_7)
        list21.append(SF_8)

        similarity, countNouns, countObjects = cosine_similarity(list4[index], objects['YOLO9000'][index])

        SF_1, SF_2, SF_3, SF_4, SF_5, SF_6, SF_7, SF_8 = Semantic_Fidelity(similarity, countNouns, countObjects)

        list22.append(SF_1)
        list23.append(SF_2)
        list24.append(SF_3)
        list25.append(SF_4)
        list26.append(SF_5)
        list27.append(SF_6)
        list28.append(SF_7)
        list29.append(SF_8)


    meta_data['Images'] = list1

    meta_data['Nouns_SAT'] = list2
    meta_data['Nouns_NOC'] = list3
    meta_data['Nouns_DNOC'] = list4

    meta_data['Objects_Y9'] = list5

    meta_data['SF_1_SAT'] = list6
    meta_data['SF_2_SAT'] = list7
    meta_data['SF_3_SAT'] = list8
    meta_data['SF_4_SAT'] = list9
    meta_data['SF_5_SAT'] = list10
    meta_data['SF_6_SAT'] = list11
    meta_data['SF_7_SAT'] = list12
    meta_data['SF_8_SAT'] = list13

    meta_data['SF_1_NOC'] = list14
    meta_data['SF_2_NOC'] = list15
    meta_data['SF_3_NOC'] = list16
    meta_data['SF_4_NOC'] = list17
    meta_data['SF_5_NOC'] = list18
    meta_data['SF_6_NOC'] = list19
    meta_data['SF_7_NOC'] = list20
    meta_data['SF_8_NOC'] = list21

    meta_data['SF_1_DNOC'] = list22
    meta_data['SF_2_DNOC'] = list23
    meta_data['SF_3_DNOC'] = list24
    meta_data['SF_4_DNOC'] = list25
    meta_data['SF_5_DNOC'] = list26
    meta_data['SF_6_DNOC'] = list27
    meta_data['SF_7_DNOC'] = list28
    meta_data['SF_8_DNOC'] = list29

    return meta_data


#meta_data = create_meta_data(captions,objects)

HSF_COCO = pd.read_csv('SF_HSF_COCO.csv')

for index in range(len(HSF_COCO['Images'])):

    nouns = detect_noun(HSF_COCO['Captions'][index])
    coco_objects = HSF_COCO['Objects'][index]

    similarity, countNouns, countObjects = cosine_similarity(nouns, coco_objects)

    SF_1, SF_2, SF_3, SF_4, SF_5, SF_6, SF_7, SF_8, SF_9, SF_10 = Semantic_Fidelity(similarity, countNouns, countObjects)

    #if SF_3 > 1.0 or SF_4 > 1.0:
        #print(nouns,coco_objects)
        #similarity, countNouns, countObjects

    HSF_COCO['SF_1'][index] = SF_1
    HSF_COCO['SF_2'][index] = SF_2
    HSF_COCO['SF_3'][index] = SF_3
    HSF_COCO['SF_4'][index] = SF_4
    HSF_COCO['SF_5'][index] = SF_5
    HSF_COCO['SF_6'][index] = SF_6
    HSF_COCO['SF_7'][index] = SF_7
    HSF_COCO['SF_8'][index] = SF_8
    HSF_COCO['SF_9'][index] = SF_9
    HSF_COCO['SF_10'][index] = SF_10

#meta_data.to_csv('Meta-data.csv')
HSF_COCO.to_csv('SF_HSF_COCO.csv')
