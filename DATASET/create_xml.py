import pandas as pd

trainval = open('./trainval.txt', 'a')
test = open('./test.txt', 'a')

for i in range(900):
    file_id = str(i).zfill(5)
    annotation_file = open('./Annotations/' + file_id + '.xml', 'a')
    annotation_file.write('<annotation>\n')
    annotation_file.close()
    if i < 600:
        trainval.write(file_id + '\n')
    else:
        test.write(file_id + '\n')
        
trainval.close()
test.close()

descriptions = ('speed limit (20)', 
                'speed limit (30)', 
                'speed limit (50)', 
                'speed limit (60)', 
                'speed limit (70)', 
                'speed limit (80)', 
                'restriction ends 80', 
                'speed limit (100)',
                'speed limit (120)', 
                'no overtaking',
                'no overtaking (trucks)',
                'priority at next intersection',
                'priority road',
                'give way',
                'stop',
                'no traffic both ways',
                'no trucks',
                'no entry',
                'danger',
                'bend left',
                'bend right',
                'bend',
                'uneven road',
                'slippery road',
                'road narrows',
                'construction',
                'traffic signal',
                'pedestrian crossing',
                'school crossing',
                'cycles crossing',
                'snow',
                'animals',
                'restriction ends',
                'go right',
                'go left',
                'go straight',
                'go right or straight',
                'go left or straight',
                'keep right',
                'keep left',
                'roundabout',
                'restriction ends (overtaking)',
                'restriction ends (overtaking (trucks))')

objects = pd.read_csv('./gt.txt', sep=';', header=None).to_numpy()

for obj in objects:
    file_id = obj[0][0:5]
    annotation_file = open('./Annotations/' + file_id + '.xml', 'a')
    annotation_file.write('\t<object>\n')
    annotation_file.write('\t\t<name>' + descriptions[int(obj[5])] + '</name>\n')
    annotation_file.write('\t\t<difficult>0</difficult>\n')
    annotation_file.write('\t\t<bndbox>\n')
    annotation_file.write('\t\t\t<xmin>' + str(obj[1]) + '</xmin>\n')
    annotation_file.write('\t\t\t<ymin>' + str(obj[2]) + '</ymin>\n')
    annotation_file.write('\t\t\t<xmax>' + str(obj[3]) + '</xmax>\n')
    annotation_file.write('\t\t\t<ymax>' + str(obj[4]) + '</ymax>\n')
    annotation_file.write('\t\t</bndbox>\n')
    annotation_file.write('\t</object>\n')
    annotation_file.close()

objects = pd.read_csv('./gt.txt', sep=';', header=None).to_numpy()

for obj in objects:
    file_id = obj[0][0:5]
    annotation_file = open('./Annotations/' + file_id + '.xml', 'a')
    annotation_file.write('\t<object>\n')
    annotation_file.write('\t\t<name>' + descriptions[int(obj[5])] + '</name>\n')
    annotation_file.write('\t\t<difficult>0</difficult>\n')
    annotation_file.write('\t\t<bndbox>\n')
    annotation_file.write('\t\t\t<xmin>' + str(obj[1]) + '</xmin>\n')
    annotation_file.write('\t\t\t<ymin>' + str(obj[2]) + '</ymin>\n')
    annotation_file.write('\t\t\t<xmax>' + str(obj[3]) + '</xmax>\n')
    annotation_file.write('\t\t\t<ymax>' + str(obj[4]) + '</ymax>\n')
    annotation_file.write('\t\t</bndbox>\n')
    annotation_file.write('\t</object>\n')
    annotation_file.close()
