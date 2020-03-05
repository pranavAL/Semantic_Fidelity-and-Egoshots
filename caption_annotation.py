import pandas as pd

file1 = pd.read_csv('ShowAttendAndTell/test/results.csv')
file1 = file1[['image_files','caption']]

file2 = pd.read_csv('dnoc/dnoc_ego.txt',header=None)

row = []
caption = []
for index, image in file2.iterrows():
    if index%2==0:
        row.append(image[0])
    else:
        caption.append(image[0])


caption2 = pd.DataFrame(row, columns=['image_files'])

caption2['caption']= caption

file3 = pd.read_csv('nocaps/results/output.imgnetcoco_3loss_voc72klabel_inglove_prelm75k_sgd_lr4e5_iter_80000.caffemodel.h5_beam_size_1.txt',header=None)

row = []
caption = []
for index, image in file3.iterrows():
    if index%2==0:
        row.append(image[0])
    else:
        caption.append(image[0])

caption3 = pd.DataFrame(row, columns=['image_files'])

caption3['caption']= caption

caption1 = file1
caption2 = caption2
caption3 = caption3

df = pd.DataFrame(columns=['ImageFiles', 'Show Attend And Tell', 'Novel Object Captioner', 'Decoupled Novel Object Captioner'])

count = 0
image = []
file1 = []
file2 = []
file3 = []
for image_name in caption3['image_files']:
    try:
        index_cap3 = caption3['image_files'].index[caption3['image_files']==image_name][0]
        index_cap2 = caption2['image_files'].index[caption2['image_files']==image_name][0]
        index_cap1 = caption1['image_files'].index[caption1['image_files']==image_name][0]
        cap3 = caption3['caption'][index_cap3]
        cap2 = caption2['caption'][index_cap2]
        cap1 = caption1['caption'][index_cap1]
        image.append(image_name)
        file1.append(cap1)
        file2.append(cap2)
        file3.append(cap3)
    except Exception as e:
        count+=1
        print(count)
        pass

df['ImageFiles'] = image
df['Show Attend And Tell'] = file1
df['Novel Object Captioner'] = file3
df['Decoupled Novel Object Captioner'] = file2

df.to_csv("Captions.csv", index=False)

