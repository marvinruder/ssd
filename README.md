Contents
========

This repository is organized as follows:

```
┤   LICENCE                        the MIT License file
│   README.md                      this README file
│   REPORT.pdf                     the project report
│
├───DATASET
│   │   create_xml.py              a script creating XML annotation files from the
│   │                              original 'gt.txt' annotation file
│   │   gt.txt                     ground truth information containing locations
│   │                              and classes of all traffic signs in the dataset
│   │   ReadMe.txt                 the dataset's README file
│   │   test.txt                   a list of identifiers of all test images
│   │   train.txt                  a list of identifiers of all test images
│   │
│   └───Annotations
│           00000.xml – 00899.xml  ground truth information containing locations
│                                  and classes of all traffic signs in each image
│
├───RESOURCES
│       Helvetica.ttf              a font neccessary to create annotations in
│                                  images, cf. image files in 'RESULTS'
│       label_map.json             a JSON file containing the traffic sign class
│                                  numbers and corresponding names
│       TEST_images.json           a JSON file containing the absolute paths of
│                                  all test image files
│       TEST_objects.json          a JSON file containing ground truth information
│                                  containing locations and classes of all traffic
│                                  signs in the test images
│       train.log                  a log file created during training, listing
│                                  epochs and loss
│       trained.pth.tar            the weights of the model trained as described
│                                  in the report
│       TRAIN_images.json          a JSON file containing the absolute paths of
│                                  all training image files
│       TRAIN_objects.json         a JSON file containing ground truth information
│                                  containing locations and classes of all traffic
│                                  signs in the training images
│
├───RESULTS
│       00601.png – 00899.png      images annotated by the detector with a
│                                  bounding box, the class name and the detection
│                                  score per detection. Images without detections
│                                  are skipped.
│
└───SRC
        area_under_curve.py        a code file to calculate precision and recall
                                   for multiple threshold values to create a
                                   precision-recall curve
        create_data_lists.py       a code file to (re)create the JSON files to be
                                   processed by the PyTorch Dataset from the XML
                                   annotation files
        datasets.py                a code file containing a PyTorch Dataset class
                                   for the GTSDB dataset
        model.py                   a code file containing the SSD model
        my_eval.py                 a code file to calculate precision and recall
                                   for one default threshold value and create the
                                   annotated image files located in 'RESULTS'
        train.py                   a code file to train the model
        utils.py                   a code file containing several utility functions
```

Requirements
============

Please download the full GTSDB dataset (available via https://doi.org/10.17894/ucph.358970eb-0474-4d8f-90b5-3f124d9f9bc6) to your computer and place all `*****.ppm` image files from its root directory into the `DATASET` folder. All other files neccessary for detection, such as the XML annotation files, have been created using the information from the original `gt.txt` file (cf. `create_xml.py`).

Dependencies
------------

* Python (we are using Python 3.7.6)
* PyTorch (we are using PyTorch 1.5.0) with torchvision
* Pillow
* tqdm

Running the code
================

1. Navigate into the `SRC` subdirectory.
2. Run `python create_data_lists.py` to create the JSON files to be processed by the data loader from the XML annotation files.
3. Run `python train.py` to train the model. The weights are saved in the file `RESOURCES\checkpoint.pth.tar`.
4. Run `python my_eval.py` to calculate precision and recall for one default threshold value (`min_value=0.45`, declared in the definition of the `my_evaluate(…)` function) and create the annotated image files located in `RESULTS`. Weights from the file `RESOURCES\trained.pth.tar` are used.
5. Run `python area_under_curve.py` to calculate precision and recall for multiple threshold values to create a precision-recall curve. The area under the curve can then be computed e.g. using a spreadsheet software of your choice.

Acknowledgements
================

Portions of the software in this repository utilize the following copyrighted material, the use of which is hereby acknowledged.

SSD: Single Shot MultiBox Detector | a PyTorch Tutorial to Object Detection
---------------------------------------------------------------------------

MIT License

Copyright (c) 2019 Sagar Vinodababu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

