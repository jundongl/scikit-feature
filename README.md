===============================
scikit-feast
===============================
Feature selection repository built on scikit-learn (DMML Lab@ASU). 

The feature selection repository is designed to collect some widely used feature selection algorithms that have been developed in the feature selection research to serve as a platform for facilitating their application, comparison and joint study. The feature selection repository also effectively assists researchers to achieve more reliable evaluation in the process of developing new feature selection algorithms. We develop the open source feature selection repository scikit-feast by one of the most popular programming language - python. It contains more than 40 popular feature selection algorithms, including most traditional feature selection algorithms and some structural and streaming feature selection algorithms. It is built upon one widely used machine learning package scikit-learn and two scientific computing packages Numpy and Scipy.

##Installing scikit-feast
###Prerequisites:
Python 2.7

NumPy

SciPy

Scikit-learn

###Steps:
After you download scikit-feast-1.0.0.zip from the project website (http://featureselection.asu.edu/scikit-feast/), unzip the file. Then the scikit-feast root directory will contain a setup script setup.py and a file named README.txt.

For Linux, under the scikit-feast root directory, you can build and install the module by the following command from a terminal：
python setup.py install

For Windows, you can also run the command prompt window (Start->Accessories), under the scikit-feast root directory:
setup.py install

##Project website
Instructions of using this package can be found in our project webpage at http://featureselection.asu.edu/scikit-feast/

##Citing

If you find scikit-feast feature selection reposoitory useful in your research, please consider citing the following paper::

    @article{Li-etal16,
       title= {Feature Selection: A Data Perspective},
       author= {J. Li and K. Cheng and S. Wang and F. Morstatter and R. Trevino and J. Tang and H. Liu},
       organization= {Arizona State University},
       year= {2016},
       url= {http://featureselection.asu.edu/scikit-feast},
    }
    
##Contact
Jundong Li
E-mail: jundong.li@asu.edu

Kewei Cheng
E-mail: kcheng18@asu.edu
