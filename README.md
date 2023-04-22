# LIS-Detection
In this project has been implemented an algorithm to detect the italian sign language (LIS - Lingua Italiana dei Segni) through [Opencv](https://opencv.org/), [Mediapipe Hand Detection](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) and a [K-nn classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).

<img src="readmeFiles/lis_example.gif" />

## :hammer: Technologies

- Python 3.8.13
- Mediapipe 0.9.2.1
- Opencv 4.6.0.66
- Sklearn 1.1.2
- Pickle 4.0

## :books: Install

```
pip install - pip/requirements.txt
```

## :bar_chart: Dataset

Dataset was acquired using a previously created tool: [acqTool](https://github.com/xandrew94x/acqTool)

- **LIS** alphabet letters were recorded (26 classes). 
- **500** vectors for each classes,
- Vectors was normalized using the _MinMaxScaler_ [0,1],
- Each vector is composed by 42 features:
```math

V_{i} = (x_{1}, y_{1}, ... , x_{n}, y_{n})

```
where: 

-  <img src="https://render.githubusercontent.com/render/math?math=x_{n}" align="center" border="0" alt="x_{n} " width="24" height="15" /> and <img src="https://render.githubusercontent.com/render/math?math=y_{n}" align="center" border="0" alt="x_{n} " width="24" height="15" /> are the position in pixels, 
- (<img src="https://render.githubusercontent.com/render/math?math=x_{n}" align="center" border="0" alt="x_{n} " width="24" height="15" />, <img src="https://render.githubusercontent.com/render/math?math=y_{n}" align="center" border="0" alt="x_{n} " width="24" height="15" />) is nth landmark on 2D image.

_Note: only right hand was used during the acquisition._

## :children_crossing: Classifier

The classifier chosen is **K-NN** ( _sklearn_ ).

Confusion Matrix: 

![confusion_matrix_knn](readmeFiles/confusion_matrix_knn.png)

## :rocket: Launch

```
python hand_detection_main.py -c {classifier.pkl}
```

## :page_facing_up: License

This project is released under the [GNU General Public License v3.0](LICENSE)

