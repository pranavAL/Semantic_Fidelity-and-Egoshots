import pandas as pd

filePath = 'detected_object.txt'
detected_objects = {}
with open(filePath) as fp:
    line = fp.readline()
    while line:
        if 'data/EgoShots/' in line.strip():
            basename = (line.strip()).split('/')[2].split(':')[0]
            key = basename
            print(basename)
        else:
            dete_obje = line.split(':')[0]
            if basename in detected_objects:
                detected_objects[basename].append(dete_obje)
            else:
                detected_objects[basename] = [dete_obje]
        line = fp.readline()


df = pd.DataFrame(list(detected_objects.items()), columns=['image', 'Objects'])    
df.to_csv('yolo9000_objects.csv')
