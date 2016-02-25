===============================
scikit-feature
===============================
Feature selection repository built on scikit-learn (DMML Lab@ASU). 

scikit-feature is an open-source feature selection repository in Python developed at Arizona State University. It is built upon one widely used machine learning package scikit-learn and two scientific computing packages Numpy and Scipy. Scikit-feature contains more than 40 popular feature selection algorithms, including traditional feature selection algorithms and some structural and streaming feature selection algorithms. 

It serves as a platform for facilitating feature selection application, research and comparative study. It is designed to share widely used feature selection algorithms developed in the feature selection research, and offer convenience for researchers and practitioners to perform empirical evaluation in their developing new feature selection algorithms.

##Installing scikit-feature
###Prerequisites:
Python 2.7

NumPy

SciPy

Scikit-learn

###Steps:
After you download scikit-feature-1.0.0.zip from the project website (http://featureselection.asu.edu/scikit-feature/), unzip the file. Then the scikit-feature root directory will contain a setup script setup.py and a file named README.txt.

For Linux, under the scikit-feature root directory, you can build and install the module by the following command:

    python setup.py install

For Windows, you can also run the command prompt window (Start->Accessories) under the scikit-feature root directory:

    setup.py install

##Project website
Instructions of using this package can be found in our project webpage at http://featureselection.asu.edu/scikit-feature/

##Citing

If you find scikit-feature feature selection reposoitory useful in your research, please consider citing the following paper::

    @article{Li-etal16,
       title= {Feature Selection: A Data Perspective},
       author= {J. Li and K. Cheng and S. Wang and F. Morstatter and R. Trevino and J. Tang and H. Liu},
       organization= {Arizona State University},
       year= {2016},
       url= {http://featureselection.asu.edu/scikit-feature},
    }
    
##Contact
Jundong Li
E-mail: jundong.li@asu.edu

Kewei Cheng
E-mail: kcheng18@asu.edu
