from kivy.app import App
from kivy.lang import Builder

from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.core.camera import Camera as CoreCamera

from kivy.clock import Clock

from kivy.graphics.texture import Texture

from kivy.properties import NumericProperty, ListProperty, BooleanProperty
from kivy.properties import ObjectProperty, NumericProperty

import cv2

from mediapipe.python.solutions.drawing_utils import draw_landmarks
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

capture_screen = cv2.VideoCapture(0)

"""class MyCamera(Widget):
    def __init__(self,**kwargs):
        super(MyCamera,self).__init__(**kwargs)
        self.build()

    def build(self):
        layout = BoxLayout()
        self.img1 = Image()
        self.capture = capture_screen
        layout.add_widget(self.img1)
        Clock.schedule_interval(self.update, 0.05)
        return layout

    def update(self, dt):
        ret, frame = self.capture.read()
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img1.texture = texture1"""

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

def draw_minimalist_points(image,results):
    mp_drawing.draw_landmarks(image,results.face_landmarks,mp_body_points.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(255,128,0),thickness=1,circle_radius=1) , mp_drawing.DrawingSpec(color=(255,128,0),thickness=1,circle_radius=1))
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_body_points.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(0,128,255),thickness=1,circle_radius=1),mp_drawing.DrawingSpec(color=(0,128,255),thickness=1,circle_radius=1) )
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_body_points.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(121,22,76),thickness=1,circle_radius=1),mp_drawing.DrawingSpec(color=(121,44,250),thickness=1,circle_radius=1) )
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_body_points.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,22,76),thickness=1,circle_radius=1),mp_drawing.DrawingSpec(color=(245,66,230),thickness=1,circle_radius=1) )

mp_body_points = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_body_points.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5)

def draw_the_points(frame):
    image,results = mediapipe_detection(frame,holistic)
    result_test = extract_keypoints(results)
    draw_minimalist_points(image,results)
    return image

"""class MyCamera(Image):
    def __init__(self, **kwargs):
        super(MyCamera, self).__init__(**kwargs)
        #Connect to 0th camera
        self.capture = capture_screen
        #Set drawing interval
        Clock.schedule_interval(self.update, 1.0 / 60)

    #Drawing method to execute at intervals
    def update(self, dt):
        #Load frame
        #holistic = mp_body_points.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5)
        ret, self.frame = self.capture.read()
        #image,results = mediapipe_detection(self.frame,holistic)
        #result_test = extract_keypoints(results)
        #draw_minimalist_points(image,results)
        #Convert to Kivy Texture
        buf = cv2.flip(self.frame, 0).tobytes()
        texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr') 
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        #Change the texture of the instance
        self.texture = texture"""


class MyCamera(Image):
    def __init__(self, **kwargs):
        super(MyCamera, self).__init__(**kwargs)
        #Connect to 0th camera
        self.capture = capture_screen
        self.pass_by_reference = []
        #Set drawing interval
        Clock.schedule_interval(self.update, 1.0 / 60)

    #Drawing method to execute at intervals
    def update(self, dt):
        #Load frame
        self.pass_by_reference.append(self.capture.read())
        """image,results = mediapipe_detection(self.frame,holistic)
        result_test = extract_keypoints(results)
        draw_minimalist_points(image,results)"""
        #Convert to Kivy Texture
        image = draw_the_points(self.pass_by_reference[0][1])
        self.pass_by_reference.pop()
        buf = cv2.flip(image, 0).tobytes()
        texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr') 
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        #Change the texture of the instance
        self.texture = texture


"""core_camera = CoreCamera(index=0, resolution=(640, 480), stopped=True)

class MyCamera(Image):
    play = BooleanProperty(True)

    index = NumericProperty(-1)

    resolution = ListProperty([-1, -1])

    def __init__(self, **kwargs):
        self._camera = None
        super(MyCamera, self).__init__(**kwargs)
        if self.index == -1:
            self.index = 0
        on_index = self._on_index
        fbind = self.fbind
        fbind('index', on_index)
        fbind('resolution', on_index)
        on_index()

    def on_tex(self, *l):
        self.canvas.ask_update()

    def _on_index(self, *largs):
        self._camera = None
        if self.index < 0:
            return
        if self.resolution[0] < 0 or self.resolution[1] < 0:
            return

        self._camera = core_camera # `core_camera` instead of `CoreCamera(index=self.index, resolution=self.resolution, stopped=True)`

        self._camera.bind(on_load=self._camera_loaded)
        if self.play:
            self._camera.start()
            self._camera.bind(on_texture=self.on_tex)

    def _camera_loaded(self, *largs):
        self.texture = self._camera.texture
        self.texture_size = list(self.texture.size)
        self.texture.flip_horizontal()

    def on_play(self, instance, value):
        if not self._camera:
            return
        if value:
            self._camera.start()
        else:
            self._camera.stop()"""

class PreMainWindow(Screen):
    pass

class MainWindow(Screen):
    pass

class SecondWindow(Screen):
    pass

class TestWindow(Screen):
    pass

class CamWindow(Screen):
    pass

class CongratsWindow(Screen):
    pass

class WindowManager(ScreenManager):
    pass


kv = Builder.load_file("kivy_educasenha.kv")

class MyMainApp(App):
    def build(self):
        return kv

if __name__ == "__main__":
    MyMainApp().run()
