from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SSD600(nn.Module):
    """
    The SSD600 network - containing the base VGG network, auxiliary, and prediction convolutions.
    """

    def __init__(self, n_classes):
        super(SSD600, self).__init__()

        self.n_classes = n_classes

        # Standard convolutional layers in VGG16
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # ceiling (not floor) here for even dims

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # retains size because stride is 1 (and padding)

        # Replacements for FC6 and FC7 in VGG16
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # atrous convolution

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        # Load pretrained layers
        self.load_pretrained_layers()

        # Since lower level features (conv3_3_feats) have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        self.rescale_factors3_3 = nn.Parameter(torch.FloatTensor(1, 256, 1, 1))  # there are 256 channels in conv3_3_feats
        nn.init.constant_(self.rescale_factors3_3, 20)
        self.rescale_factors4_3 = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors4_3, 20)

        
        # Auxiliary/additional convolutions on top of the VGG base
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=3, padding=1)  # stride = 1, by default
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        # Initialize convolutions' parameters
        nn.init.xavier_uniform_(self.conv8_1.weight)
        nn.init.constant_(self.conv8_1.bias, 0.)
        nn.init.xavier_uniform_(self.conv8_2.weight)
        nn.init.constant_(self.conv8_2.bias, 0.)

        
        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'conv3_3': 4,
                   'conv4_3': 4,
                   'conv7': 4,
                   'conv8_2': 4}

        # 4 prior boxes implies we use 4 different aspect ratios, etc.

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_conv3_3 = nn.Conv2d(256, n_boxes['conv3_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv3_3 = nn.Conv2d(256, n_boxes['conv3_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)

        # Initialize convolutions' parameters
        nn.init.xavier_uniform_(self.loc_conv3_3.weight)
        nn.init.constant_(self.loc_conv3_3.bias, 0.)
        nn.init.xavier_uniform_(self.loc_conv4_3.weight)
        nn.init.constant_(self.loc_conv4_3.bias, 0.)
        nn.init.xavier_uniform_(self.loc_conv7.weight)
        nn.init.constant_(self.loc_conv7.bias, 0.)
        nn.init.xavier_uniform_(self.loc_conv8_2.weight)
        nn.init.constant_(self.loc_conv8_2.bias, 0.)

        nn.init.xavier_uniform_(self.cl_conv3_3.weight)
        nn.init.constant_(self.cl_conv3_3.bias, 0.)
        nn.init.xavier_uniform_(self.cl_conv4_3.weight)
        nn.init.constant_(self.cl_conv4_3.bias, 0.)
        nn.init.xavier_uniform_(self.cl_conv7.weight)
        nn.init.constant_(self.cl_conv7.bias, 0.)
        nn.init.xavier_uniform_(self.cl_conv8_2.weight)
        nn.init.constant_(self.cl_conv8_2.bias, 0.)


        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def load_pretrained_layers(self):
        """
        We use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())
        
        # Pretrained VGG base
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:-4]):  # excluding conv6 and conv7 parameters
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        # Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of conv6 and conv7
        # fc6
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

        # Note: an FC layer of size (K) operating on a flattened version (C*H*W) of a 2D image of size (C, H, W)...
        # ...is equivalent to a convolutional layer with kernel size (H, W), input channels C, output channels K...
        # ...operating on the 2D image of size (C, H, W) without padding

        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")


    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 600, 600)
        :return: 119720 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        out = F.relu(self.conv1_1(image))  # (N, 64, 600, 600)
        out = F.relu(self.conv1_2(out))  # (N, 64, 600, 600)
        out = self.pool1(out)  # (N, 64, 300, 300)

        out = F.relu(self.conv2_1(out))  # (N, 128, 300, 300)
        out = F.relu(self.conv2_2(out))  # (N, 128, 300, 300)
        out = self.pool2(out)  # (N, 128, 150, 150)

        out = F.relu(self.conv3_1(out))  # (N, 256, 150, 150)
        out = F.relu(self.conv3_2(out))  # (N, 256, 150, 150)
        out = F.relu(self.conv3_3(out))  # (N, 256, 150, 150)
        conv3_3_feats = out  # (N, 512, 150, 150)
        out = self.pool3(out)  # (N, 256, 75, 75)

        out = F.relu(self.conv4_1(out))  # (N, 512, 75, 75)
        out = F.relu(self.conv4_2(out))  # (N, 512, 75, 75)
        out = F.relu(self.conv4_3(out))  # (N, 512, 75, 75)
        conv4_3_feats = out  # (N, 512, 75, 75)
        out = self.pool4(out)  # (N, 512, 38, 38), it would have been 37 if not for ceil_mode = True

        out = F.relu(self.conv5_1(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv5_2(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv5_3(out))  # (N, 512, 38, 38)
        out = self.pool5(out)  # (N, 512, 38, 38), pool5 does not reduce dimensions

        out = F.relu(self.conv6(out))  # (N, 1024, 38, 38)

        out = F.relu(self.conv7(out))  # (N, 1024, 38, 38)
        conv7_feats = out
        
        # Rescale conv3_3 after L2 norm
        norm = conv3_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 150, 150)
        conv3_3_feats = conv3_3_feats / norm  # (N, 512, 150, 150)
        conv3_3_feats = conv3_3_feats * self.rescale_factors3_3  # (N, 512, 150, 150)
        # (PyTorch autobroadcasts singleton dimensions during arithmetic)

        # Rescale conv4_3 after L2 norm
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 75, 75)
        conv4_3_feats = conv4_3_feats / norm  # (N, 512, 75, 75)
        conv4_3_feats = conv4_3_feats * self.rescale_factors4_3  # (N, 512, 75, 75)

        
        # Run auxiliary convolutions (higher level feature map generators)
        out = F.relu(self.conv8_1(conv7_feats))  # (N, 256, 38, 38)
        out = F.relu(self.conv8_2(out))  # (N, 512, 19, 19)
        conv8_2_feats = out  # (N, 512, 19, 19)

        
        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        batch_size = conv3_3_feats.size(0)
        
        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv3_3 = self.loc_conv3_3(conv3_3_feats)  # (N, 16, 150, 150)
        l_conv3_3 = l_conv3_3.permute(0, 2, 3, 1).contiguous()  # (N, 150, 150, 16), to match prior-box order (after .view())
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        l_conv3_3 = l_conv3_3.view(batch_size, -1, 4)  # (N, 90000, 4), there are a total 90000 boxes on this feature map

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 16, 75, 75)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()  # (N, 75, 75, 16), to match prior-box order (after .view())
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # (N, 22500, 4), there are a total 22500 boxes on this feature map

        l_conv7 = self.loc_conv7(conv7_feats)  # (N, 16, 38, 38)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 16)
        l_conv7 = l_conv7.view(batch_size, -1, 4)  # (N, 5776, 4), there are a total 5776 boxes on this feature map

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 16, 19, 19)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 16)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # (N, 1444, 4), there are a total 1444 boxes on this feature map


        # Predict classes in localization boxes
        c_conv3_3 = self.cl_conv3_3(conv3_3_feats)  # (N, 4 * n_classes, 150, 150)
        c_conv3_3 = c_conv3_3.permute(0, 2, 3,
                                      1).contiguous()  # (N, 150, 150, 4 * n_classes), to match prior-box order (after .view())
        c_conv3_3 = c_conv3_3.view(batch_size, -1,
                                   self.n_classes)  # (N, 90000, n_classes), there are a total 90000 boxes on this feature map

        # Predict classes in localization boxes
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, 4 * n_classes, 75, 75)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3,
                                      1).contiguous()  # (N, 75, 75, 4 * n_classes), to match prior-box order (after .view())
        c_conv4_3 = c_conv4_3.view(batch_size, -1,
                                   self.n_classes)  # (N, 22500, n_classes), there are a total 22500 boxes on this feature map

        c_conv7 = self.cl_conv7(conv7_feats)  # (N, 4 * n_classes, 38, 38)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 4 * n_classes)
        c_conv7 = c_conv7.view(batch_size, -1,
                               self.n_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  # (N, 4 * n_classes, 19, 19)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 4 * n_classes)
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)  # (N, 1444, n_classes), there are a total 1444 boxes on this feature map


        # A total of 119720 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_conv3_3, l_conv4_3, l_conv7, l_conv8_2 ], dim=1) # (N, 119720, 4)
        classes_scores = torch.cat([c_conv3_3, c_conv4_3, c_conv7, c_conv8_2 ], dim=1)  # (N, 119720, n_classes)

        return locs, classes_scores

    def create_prior_boxes(self):
        """
        Create the 119720 prior (default) boxes for the SSD600.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (119720, 4)
        """
        fmap_dims = {'conv3_3': 150,
                     'conv4_3': 75,
                     'conv7': 38,
                     'conv8_2': 19}

        # small objects only
        obj_scales = {'conv3_3': 0.04, 
                      'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375}

        # near-square aspect ratios only
        aspect_ratios = {'conv3_3': [0.5, 0.6, 0.7],
                         'conv4_3': [0.5, 0.6, 0.7],
                         'conv7': [0.5, 0.6, 0.7],
                         'conv8_2': [0.5, 0.6, 0.7]}


        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # For an aspect ratio of 0.6, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        if ratio == 0.6:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (119720, 4)
        prior_boxes.clamp_(0, 1)  # (119720, 4)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 119720 locations and class scores (output of ths SSD600) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 119720 prior boxes, a tensor of dimensions (N, 119720, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 119720, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 119720, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (119720, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (119720)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (119720)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 119720
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.bool).to(device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == True:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = False

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[~suppress])
                image_labels.append(torch.LongTensor((~suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[~suppress])
            
            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # remove multiple class suggestions for the same object    
            image_boxes, image_scores, image_labels = self.remove_duplicates(image_boxes, image_scores, image_labels, max_overlap)
            
            
            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size
    
    def remove_duplicates(self, image_boxes, image_scores, image_labels, max_overlap):
        """
        :param image_boxes: the boxes in an image
        :param image_scores: the scores of the boxes
        :param image_labels: the labels of the boxes
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed here
        :return: the unsuppressed boxes with scores and labels
        """
        suppress = torch.zeros((image_scores.shape[0]), dtype=torch.bool).to(device)
        for i in range(image_scores.shape[0]):
            for j in range(image_scores.shape[0]):
                if i == j:
                    continue
                if find_jaccard_overlap(image_boxes[i].unsqueeze(0), image_boxes[j].unsqueeze(0)) > max_overlap:
                    suppress[i if image_scores[i] < image_scores[j] else j] = True
                    return self.remove_duplicates(image_boxes[~suppress], image_scores[~suppress], image_labels[~suppress], max_overlap)
                    
        return image_boxes, image_scores, image_labels

class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 119720 prior boxes, a tensor of dimensions (N, 119720, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 119720, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
        
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 119720, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 119720)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 119720)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (119720)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (119720)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (119720)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (119720, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 119720)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 119720)
        # So, if predicted_locs has the shape (N, 119720, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 119720)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 119720)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 119720)
        conf_loss_neg[positive_priors] = 0.  # (N, 119720), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 119720), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 119720)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 119720)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS
                
        return conf_loss + self.alpha * loc_loss

