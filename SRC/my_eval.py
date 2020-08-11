from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
from datasets import GTSDBDataset
from tqdm import tqdm
from pprint import PrettyPrinter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()


# Label map
voc_labels = ('speed limit (20)', 
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
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping


# Load model checkpoint
checkpoint = '..\\RESOURCES\\trained.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((600,600))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
all_results = {}
results = {}
for label in voc_labels:
    results[label] = {}
    results[label]['true_positive'] = 0
    results[label]['false_positive'] = 0
    results[label]['false_negative'] = 0

all_results['true_positive'] = 0
all_results['false_positive'] = 0
all_results['false_negative'] = 0

def box_match(box_1, label_1, box_2, label_2, max_overlap=0.3):
    """
    Determines whether two boxes overlap themselves and have the same label
    :param box_1: the first box
    :param label_1: the label of the first box
    :param box_2: the second box
    :param label_2: the label of the second box
    :param max_overlap: threshold value for determining whether two boxes overlap themselves
    :return: boolean value indicating whether the boxes overlap themselves and have the same label
    """
    return find_jaccard_overlap(box_1.unsqueeze(0), box_2.unsqueeze(0)) > max_overlap and label_1 == label_2

def my_evaluate(original_image, img_id, annotations, min_score=0.45, max_overlap=0.3, top_k=200, annotate_image=True):
    """
    Detect objects in an image with a trained SSD600, and visualize the results.

    :param original_image: image, a PIL Image
    :param img_id: the identifier of the image, used as file name
    :param annotations: ground truth information on the traffic signs in the image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param annotate_image: boolean variable indicating whether annotated images shall be written to a file
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    det_scores = det_scores[0].to('cpu').tolist()
    
    # Annotate
    annotated_image = original_image

    if det_labels != ['background']:
        for i in range(det_boxes.size(0)):
            
            # Create an image showing the detected traffic signs, if requested
            if annotate_image:
                draw = ImageDraw.Draw(annotated_image)
                font = ImageFont.truetype("..\\RESOURCES\\Helvetica.ttf", 16)

                # Boxes
                box_location = det_boxes[i].tolist()
                draw.rectangle(xy=box_location, outline='#ff0000')
                draw.rectangle(xy=[l + 1. for l in box_location], outline='#ff0000')  # a second rectangle at an offset of 1 pixel to increase line thickness

                # Text
                text = det_labels[i].upper() + ' ' + str(det_scores[i])
                text_size = font.getsize(text)
                text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
                textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                                    box_location[1]]
                draw.rectangle(xy=textbox_location, fill='#ff0000')
                draw.text(xy=text_location, text=text, fill='white', font=font)

                del draw
            
            # For every detection, see whether it matches a ground truth
            match = False
            for j in range(len(annotations['boxes'])):
                if annotations['labels'][j] == -1: # this is being set when a ground truth already matched
                    continue
                match = box_match(det_boxes[i], 
                                  det_labels[i], 
                                  torch.Tensor(annotations['boxes'][j]), 
                                  rev_label_map[annotations['labels'][j]])
                if match:
                    annotations['labels'][j] = -1
                    break
            
            if match: # true positive if the detection is correct and matched a ground truth
                all_results['true_positive'] += 1
                results[det_labels[i]]['true_positive'] += 1
            else: # false positive if the detection did not match a ground truth
                all_results['false_positive'] += 1
                results[det_labels[i]]['false_positive'] += 1
        
        if annotate_image:
            annotated_image.save('..\\RESULTS\\' + img_id + '.png')


    # After all detections were checked, let us see whether the detector missed something
    for label in annotations['labels']: 
        if label == -1: # This is set after a detection matched this ground truth
            continue
        
        # false negative if we reach this line, since the ground truth object was not found
        results[rev_label_map[label]]['false_negative'] += 1 
        all_results['false_negative'] += 1
    

    
if __name__ == '__main__':
    path = '..\\DATASET\\'
    
    # Find IDs of images in the test data
    with open(os.path.join(path, 'test.txt')) as f:
        ids = f.read().splitlines()
    
    # Evaluate and annotate
    for img_id in tqdm(ids):
        annotations = parse_annotation(path + 'Annotations\\' + img_id + '.xml')
        original_image = Image.open(path + img_id + '.ppm', mode='r')
        original_image = original_image.convert('RGB')
        my_evaluate(original_image, img_id, annotations, annotate_image=True)
        
    # Calculate precisio and recall
    precision = {}
    recall = {}
    precision['ALL'] = all_results['true_positive'] / (all_results['true_positive'] + all_results['false_positive'])       
    recall['ALL'] = all_results['true_positive'] / (all_results['true_positive'] + all_results['false_negative'])
    
    for label in voc_labels:
        # Precision
        if results[label]['true_positive'] + results[label]['false_positive'] > 0: # check for detections
            precision[label] = results[label]['true_positive'] / (results[label]['true_positive'] + results[label]['false_positive'])
        else:
            precision[label] = 'No detections'
            if results[label]['false_negative'] == 0:
                precision[label] = 'No detections, but also no signs in test set'
                
        # Recall
        if results[label]['true_positive'] + results[label]['false_negative'] > 0: # check for ground truth objects
            recall[label] = results[label]['true_positive'] / (results[label]['true_positive'] + results[label]['false_negative'])
        else:
            recall[label] = 'No signs in test set'
    
    # Print results
    print('PRECISION')
    pp.pprint(precision)
    print()
    print('RECALL')
    pp.pprint(recall)
