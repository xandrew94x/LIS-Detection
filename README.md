# LIS-Detection
In this repository has been implemented an algorithm to identify the italian sign language (LIS - Lingua Italiana dei Segni) through Opencv, Mediapipe Hand Detection and a K-nn classifier.

## :hammer: Technologies

- Python 3.8
- Mediapipe
- Opencv
- Sklearn
- Pickle

## Dataset

Dataset was acquired using a previously created tool: [acqTool](https://github.com/xandrew94x/acqTool)

With _acqTool_ 26 classes were recorded, using alphabet letters of italian sign language. 

All classes are balanced, with up to **500** items each.

_Note: only right hand was used during the acquisition._

## Classifier


## :rocket: Launch

```
python hand_detection_main.py -c {classifier.pkl}
```

## :page_facing_up: License

This project is released under the [GNU General Public License v3.0](LICENSE)

