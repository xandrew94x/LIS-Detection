#!/usr/bin/env python
# coding: utf-8

# In[121]:


import cv2
from PIL import Image, ImageTk
import os
import pickle
import string
import numpy as np
from MediapipeModels import MediapipeHandModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

class GUI:
    def __init__(self, cls_path_name:str):
        
        self.init_mediapipe_components()
        self.knn_cls = self.load_classificator(cls_path_name)
        self.current_pred = ""
        self.current_phrase = ""
        self.info_text = "press 'space' to save char\npress 'd' to delete phrase\npress 'q' to quit"
    
    def init_mediapipe_components(self):
        
        """
        init_mediapipe_components()
        
        Init all mediapipe component witch i need
        """
        self.mp_class = MediapipeHandModel()
        self.mp_model = self.mp_class.return_hand_model()
        self.mp_hands = self.mp_class.return_mp_hands()
        self.mp_drawing = self.mp_class.return_mp_drawing()
        self.mp_drawing_styles = self.mp_class.return_mp_drawing_styles()
      
    
    def load_classificator(self, model_name:str):
        
        """
        load_classificator()
        
        Load the classifier from input path.
        
        Note: the file should be .pkl type.
        
        input:
            - model_name (str): path to the model
            
        output:
            - loaded_model
        """
        
        # Load the model from disk
        loaded_model = pickle.load(open(model_name, 'rb'))
        
        return loaded_model
     
        
    def norm_min_max(self, vec:np.array) -> list:
        
        # Init MinMaxScaler
        scaler = MinMaxScaler()
        
        # Fit MinMaxScaler with vector features
        scaler.fit(vec)
        
        # Normalize vector
        norm = scaler.transform(vec)
        
        # Reshape
        norm = norm.reshape(1, len(norm))[0]
        
        return list(norm)
    
    
    def put_info_text(self, image:np.array, text:str) -> np.array:
        
        """
        put_info_text()
        
        Put info text in cv2 image.
        """
        
        x_pos = image.shape[0] - 50
        y_pos = 10
        for i, line in enumerate(text.split('\n')):
            cv2image = cv2.putText(
                      img = image,
                      text = line,
                      org = (x_pos, y_pos),
                      fontFace = cv2.FONT_HERSHEY_DUPLEX,
                      fontScale = 0.4,
                      color = (58, 255, 255),
                      thickness = 1
                    )
            y_pos += 12
        return cv2image
    
    
    def put_pred_text(self, image:np.array, text:str) -> np.array:
        
        """
        put_pred_text()
        
        Update predicted text in cv2 image.
        """
        cv2image = cv2.putText(
                  img = image,
                  text = text,
                  org = (20, 50),
                  fontFace = cv2.FONT_HERSHEY_DUPLEX,
                  fontScale = 1.5,
                  color = (24, 57, 255),
                  thickness = 3
                )
        return cv2image

    def put_select_text(self, image:np.array, text:str) -> np.array:
        
        """
        put_select_text()
        
        Update selected text in cv2 image.
        """
        
        cv2image = cv2.putText(
                  img = image,
                  text = text,
                  org = (20, 80),
                  fontFace = cv2.FONT_HERSHEY_DUPLEX,
                  fontScale = 1.5,
                  color = (125, 246, 55),
                  thickness = 3
                )
        return cv2image
    
    
    def label_to_str(self, label:int) -> str:
        
        return str(list(string.ascii_lowercase)[label])
    
    
    def get_mediapipe_keypoints(self, landmark, window_w:int, window_h:int) -> list:
    
        """
        get_mediapipe_keypoints()

        This method is used to extract keypoints.
        Mediapipe keypoits are normalized, window_w and window_h are used to remove normalization.

        input: 
            - landmark      : Mediapipe landmarks
            - window_w (int): GUI width
            - window_h (int): GUI height
        output:
            - vector (list): vector of keypoints -> ex: [X1, Y1, ... , Xn, Yn]
        """
        # Init output vector
        vector = []

        # For each landmark
        for markers in landmark:

            # For each marker
            for mark in range(len(markers.landmark)):

                # Get X and Y keypoint not normalized
                vector.append(markers.landmark[mark].x * window_w)
                vector.append(markers.landmark[mark].y * window_h)

        return vector
    
    def show_hand_keypoints(self, cv2image:np.array) -> np.array:
    
        """
        show_hand_keypoints()

        This method is used to print in cv2 image the hand keypoints and the text of predicted label.

        inputs:
            - cv2image (np.array): the cv2 image
        output:
            - image (np.array): the new image create with the keypoint and text printed
        """

        # Proced cv2 image with the Hand Mediapipe model
        results_img = self.mp_model.process(cv2image)

        # Check if there are landmarks 
        if results_img.multi_hand_landmarks:

            # Load keypoints to vectors
            vector = self.get_mediapipe_keypoints(results_img.multi_hand_landmarks,cv2image.shape[1],cv2image.shape[0])

            # Normalize keypoint vector
            vector = self.norm_min_max(np.array(vector).reshape(-1, 1))

            # Check if there is only one hand detected (one hand have 42 features in this case)
            if len(vector) == 42:

                # Predict hand features
                pred = self.knn_cls.predict([vector])[0]

                # Convert label pred (int) to alphabet letter (str) 
                self.current_pred = self.label_to_str(int(pred))

                # Put text on cv2 preview
                self.put_pred_text(cv2image, self.current_pred)
                
            # Show keypoints        
            for hand_landmarks in results_img.multi_hand_landmarks:

                # Mediapipe settings to show landmark
                self.mp_drawing.draw_landmarks(
                    cv2image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

        return cv2image
    
    
    def main_gui(self):
        
        width, height = 360, 360
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        while True:
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:

                    frame = cv2.flip(frame, 1)
                    frame = self.show_hand_keypoints(frame)
                    
                    self.put_select_text(frame, self.current_phrase)
                    
                    self.put_info_text(frame, self.info_text)
                    
                    cv2.imshow('Frame', frame)
                else:
                    print("Frame not captured.")
                
                # On spacebar press
                if cv2.waitKey(1) & 0xFF==ord(' '):
                    
                    # Update current phrase
                    self.current_phrase += self.current_pred
                    
                # On press d (delete)
                if cv2.waitKey(1) & 0xFF==ord('d'):
                    
                    # Delete currente phrase
                    self.current_phrase = "" 
                
                # On press q
                if cv2.waitKey(1) & 0xFF==ord('q'):
                    
                    # Quit from while cycle
                    break
            else:
                
                # Release camera
                cap.release
                
                # Close window
                cv2.destroyAllWindows()
                
                print("Cannot open camera.")
                
        # Release camera
        cap.release

        # Close window
        cv2.destroyAllWindows()
        
        print("Window closed.")

