# Convolutional Neural Network For Mammography Classification
By Tianxiang Chen (ORNL Reseach Assistant)

This project presents how to design a classifier for distinguishing malignant, benign, and normal tumor using Deep Learning Neural Network (CNN). Then compare the performance of regular Neural Network. Also, compare the training efficiency between GPU and CPU.

Required Pathon package: 

    nolearn; skimage; sklearn; Theano; CUDA

Required Hardware:

    GPU: GRID K520, 4GB memory. (You can reduce the batch sizes if you don't have enough memory)

Datasets: 

    The dataset for training contains 3 classes, “benign” (102 images), “cancers” (177 images) and “normal” (399 images). Dataset is unbalanced. All the images are has 256 by 256 pixels and approximately 70 KB, stored in “.png” format. (Private dataset)

Results:
    benign = 0 ; Cancer = 1; Normal = 2

    (a) Deep learning Neural Network. Use GLCM (Dissimilarity, Correlation, ASM, Homogeneity) and Entropy as input features.
    
               Precision    Support                 
            0       0.20     	 24
            1       0.33     	 33
            2       0.82         79
        avg / total   0.60  	136        

    (b) Convolutional Neural Network. Normalized raw pixel values as input.
    
                Precision    support
            0       0.57     	 23
            1       0.69     	 30
            2       0.88         85
        avg / total   0.79  	138

