
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
    y = countObjects / countNouns

    if countNouns > countObjects:
        x = 1
        y = 1

    metric1 = cosine ** y
    metric2 = cosine ** (y/(max(countObjects,countNouns)+1))
    metric3 = cosine ** (y/(max(countObjects,countNouns)**(0.5)+1))
    metric4 = (cosine ** y)/(max(countObjects,countNouns)+1)
    metric5 = (cosine ** y)/(max(countObjects,countNouns)**(0.5)+1)

    metric6 = cosine ** (1-x)
    metric7 = cosine ** ((1-x)/(max(countObjects,countNouns)+1))
    metric8 = cosine ** ((1-x)/(max(countObjects,countNouns)**(0.5)+1))
    metric9 = (cosine ** (1-x))/(max(countObjects,countNouns)+1)
    metric10 = (cosine ** (1-x))/(max(countObjects,countNouns)**(0.5)+1)

    metric11 = cosine * x
    metric12 = cosine * (x /(max(countObjects,countNouns)+1))
    metric13 = cosine * (x/(max(countObjects,countNouns)**(0.5)+1))

    metric14 = cosine * (1 - 1 / (1 + x))
    metric15 = (1 - np.exp(-cosine ** (1-x)))
    metric16 = cosine * (1 - np.exp(-x))
    metric17 = (1 - np.exp(-cosine * x))
    metric18 = 1 - (cosine ** x)
    metric19 = 1 - ((cosine ** x)/(max(countObjects,countNouns)+1))
    metric20 = 1 - ((cosine ** x)/(max(countObjects,countNouns)**(0.5)+1))

    return metric1, metric2, metric3, metric4, metric5, metric6, metric7, metric8, metric9, metric10, metric11, metric12, metric13, metric14, metric15, metric16, metric17, metric18, metric19, metric20


def create_meta_data(captions, objects):

    meta_data = pd.DataFrame(columns = ['Images','Nouns_SAT','Nouns_NOC','Nouns_DNOC','Objects_Y9',
                            'SF_1_SAT','SF_2_SAT','SF_3_SAT','SF_4_SAT','SF_5_SAT','SF_6_SAT','SF_7_SAT','SF_8_SAT','SF_9_SAT','SF_10_SAT',
                            'SF_11_SAT','SF_12_SAT','SF_13_SAT','SF_14_SAT','SF_15_SAT','SF_16_SAT','SF_17_SAT','SF_18_SAT','SF_19_SAT','SF_20_SAT',
                            'SF_1_NOC','SF_2_NOC','SF_3_NOC','SF_4_NOC','SF_5_NOC','SF_6_NOC','SF_7_NOC','SF_8_NOC','SF_9_NOC','SF_10_NOC',
                            'SF_11_NOC','SF_12_NOC','SF_13_NOC','SF_14_NOC','SF_15_NOC','SF_16_NOC','SF_17_NOC','SF_18_NOC','SF_19_NOC','SF_20_NOC',
                            'SF_1_DNOC','SF_2_DNOC','SF_3_DNOC','SF_4_DNOC','SF_5_DNOC','SF_6_DNOC','SF_7_DNOC','SF_8_DNOC','SF_9_DNOC','SF_10_DNOC',
                            'SF_11_DNOC','SF_12_DNOC','SF_13_DNOC','SF_14_DNOC','SF_15_DNOC','SF_16_DNOC','SF_17_DNOC','SF_18_DNOC','SF_19_DNOC','SF_20_DNOC'])
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
    list30 = []
    list31 = []
    list32 = []
    list33 = []
    list34 = []
    list35 = []
    list36 = []
    list37 = []
    list38 = []
    list39 = []
    list40 = []
    list41 = []
    list42 = []
    list43 = []
    list44 = []
    list45 = []
    list46 = []
    list47 = []
    list48 = []
    list49 = []
    list50 = []
    list51 = []
    list52 = []
    list53 = []
    list54 = []
    list55 = []
    list56 = []
    list57 = []
    list58 = []
    list59 = []
    list60 = []
    list61 = []
    list62 = []
    list63 = []
    list64 = []
    list65 = []

    for index,image in enumerate(objects['ImageFiles']):

        index_of_caption = captions.index[captions['ImageFiles'] == image]

        list1.append(image)

        list2.append(detect_noun(list(captions['Show Attend And Tell'][index_of_caption])))
        list3.append(detect_noun(list(captions['Novel Object Captioner'][index_of_caption])))
        list4.append(detect_noun(list(captions['Decoupled Novel Object Captioner'][index_of_caption])))

        list5.append(objects['YOLO9000'][index])

        similarity, countNouns, countObjects = cosine_similarity(list2[index], objects['YOLO9000'][index])

        SF_1, SF_2, SF_3, SF_4, SF_5, SF_6, SF_7, SF_8, SF_9, SF_10, SF_11, SF_12, SF_13, SF_14, SF_15, SF_16, SF_17, SF_18, SF_19, SF_20 = Semantic_Fidelity(similarity, countNouns, countObjects)

        list6.append(SF_1)
        list7.append(SF_2)
        list8.append(SF_3)
        list9.append(SF_4)
        list10.append(SF_5)
        list11.append(SF_6)
        list12.append(SF_7)
        list13.append(SF_8)
        list14.append(SF_9)
        list15.append(SF_10)
        list16.append(SF_11)
        list17.append(SF_12)
        list18.append(SF_13)
        list19.append(SF_14)
        list20.append(SF_15)
        list21.append(SF_16)
        list22.append(SF_17)
        list23.append(SF_18)
        list24.append(SF_19)
        list25.append(SF_20)

        similarity, countNouns, countObjects = cosine_similarity(list3[index], objects['YOLO9000'][index])

        SF_1, SF_2, SF_3, SF_4, SF_5, SF_6, SF_7, SF_8, SF_9, SF_10, SF_11, SF_12, SF_13, SF_14, SF_15, SF_16, SF_17, SF_18, SF_19, SF_20 = Semantic_Fidelity(similarity, countNouns, countObjects)

        list26.append(SF_1)
        list27.append(SF_2)
        list28.append(SF_3)
        list29.append(SF_4)
        list30.append(SF_5)
        list31.append(SF_6)
        list32.append(SF_7)
        list33.append(SF_8)
        list34.append(SF_9)
        list35.append(SF_10)
        list36.append(SF_11)
        list37.append(SF_12)
        list38.append(SF_13)
        list39.append(SF_14)
        list40.append(SF_15)
        list41.append(SF_16)
        list42.append(SF_17)
        list43.append(SF_18)
        list44.append(SF_19)
        list45.append(SF_20)

        similarity, countNouns, countObjects = cosine_similarity(list4[index], objects['YOLO9000'][index])

        SF_1, SF_2, SF_3, SF_4, SF_5, SF_6, SF_7, SF_8, SF_9, SF_10, SF_11, SF_12, SF_13, SF_14, SF_15, SF_16, SF_17, SF_18, SF_19, SF_20 = Semantic_Fidelity(similarity, countNouns, countObjects)

        list46.append(SF_1)
        list47.append(SF_2)
        list48.append(SF_3)
        list49.append(SF_4)
        list50.append(SF_5)
        list51.append(SF_6)
        list52.append(SF_7)
        list53.append(SF_8)
        list54.append(SF_9)
        list55.append(SF_10)
        list56.append(SF_11)
        list57.append(SF_12)
        list58.append(SF_13)
        list59.append(SF_14)
        list60.append(SF_15)
        list61.append(SF_16)
        list62.append(SF_17)
        list63.append(SF_18)
        list64.append(SF_19)
        list65.append(SF_20)

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
    meta_data['SF_9_SAT'] = list14
    meta_data['SF_10_SAT'] = list15
    meta_data['SF_11_SAT'] = list16
    meta_data['SF_12_SAT'] = list17
    meta_data['SF_13_SAT'] = list18
    meta_data['SF_14_SAT'] = list19
    meta_data['SF_15_SAT'] = list20
    meta_data['SF_16_SAT'] = list21
    meta_data['SF_17_SAT'] = list22
    meta_data['SF_18_SAT'] = list23
    meta_data['SF_19_SAT'] = list24
    meta_data['SF_20_SAT'] = list25

    meta_data['SF_1_NOC'] = list26
    meta_data['SF_2_NOC'] = list27
    meta_data['SF_3_NOC'] = list28
    meta_data['SF_4_NOC'] = list29
    meta_data['SF_5_NOC'] = list30
    meta_data['SF_6_NOC'] = list31
    meta_data['SF_7_NOC'] = list32
    meta_data['SF_8_NOC'] = list33
    meta_data['SF_9_NOC'] = list34
    meta_data['SF_10_NOC'] = list35
    meta_data['SF_11_NOC'] = list36
    meta_data['SF_12_NOC'] = list37
    meta_data['SF_13_NOC'] = list38
    meta_data['SF_14_NOC'] = list39
    meta_data['SF_15_NOC'] = list40
    meta_data['SF_16_NOC'] = list41
    meta_data['SF_17_NOC'] = list42
    meta_data['SF_18_NOC'] = list43
    meta_data['SF_19_NOC'] = list44
    meta_data['SF_20_NOC'] = list45

    meta_data['SF_1_DNOC'] = list46
    meta_data['SF_2_DNOC'] = list47
    meta_data['SF_3_DNOC'] = list48
    meta_data['SF_4_DNOC'] = list49
    meta_data['SF_5_DNOC'] = list50
    meta_data['SF_6_DNOC'] = list51
    meta_data['SF_7_DNOC'] = list52
    meta_data['SF_8_DNOC'] = list53
    meta_data['SF_9_DNOC'] = list54
    meta_data['SF_10_DNOC'] = list55
    meta_data['SF_11_DNOC'] = list56
    meta_data['SF_12_DNOC'] = list57
    meta_data['SF_13_DNOC'] = list58
    meta_data['SF_14_DNOC'] = list59
    meta_data['SF_15_DNOC'] = list60
    meta_data['SF_16_DNOC'] = list61
    meta_data['SF_17_DNOC'] = list62
    meta_data['SF_18_DNOC'] = list63
    meta_data['SF_19_DNOC'] = list64
    meta_data['SF_20_DNOC'] = list65

    return meta_data


meta_data = create_meta_data(captions,objects)

HSF_COCO = pd.read_csv('SF_HSF_COCO.csv')

for index in range(len(HSF_COCO['Images'])):

    nouns = detect_noun(HSF_COCO['Captions'][index])
    coco_objects = HSF_COCO['Objects'][index]

    similarity, countNouns, countObjects = cosine_similarity(nouns, coco_objects)

    SF_1, SF_2, SF_3, SF_4, SF_5, SF_6, SF_7, SF_8, SF_9, SF_10, SF_11, SF_12, SF_13, SF_14, SF_15, SF_16, SF_17, SF_18, SF_19, SF_20 = Semantic_Fidelity(similarity, countNouns, countObjects)

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
    HSF_COCO['SF_11'][index] = SF_11
    HSF_COCO['SF_12'][index] = SF_12
    HSF_COCO['SF_13'][index] = SF_13
    HSF_COCO['SF_14'][index] = SF_14
    HSF_COCO['SF_15'][index] = SF_15
    HSF_COCO['SF_16'][index] = SF_16
    HSF_COCO['SF_17'][index] = SF_17
    HSF_COCO['SF_18'][index] = SF_18
    HSF_COCO['SF_19'][index] = SF_19
    HSF_COCO['SF_20'][index] = SF_20

meta_data.to_csv('Meta-data.csv')
HSF_COCO.to_csv('SF_HSF_COCO.csv')
