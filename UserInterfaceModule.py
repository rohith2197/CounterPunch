from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import (
    Color, RoundedRectangle, Rectangle,
    StencilPush, StencilUse, StencilUnUse, StencilPop, Line
)
from kivy.core.text import Label as CoreLabel
from kivy.core.window import Window

import cv2
from collections import deque

from GuardAnalyzer import image_guard_analysis
from HeadAnalyzer import analyze_frame

# Screen Size Relative Configuration
WINDOW_W = 1200
WINDOW_H = 700
SIDE_PANEL_W = 800
BG_GREY = (0.1, 0.1, 0.1, 1)
CAM_INDEX = 0
RADIUS = 50

CAMERA_RADIUS = 50
ROLLING_PRED_WINDOW = 30
GRAPH_HISTORY_POINTS = 120

LEFT_TITLE_Y_OFFSET = -20000
RIGHT_TITLE_Y_OFFSET = 0
GRAPH_TITLE_Y_OFFSET = 0

# Spacing Constants
RIGHT_PANEL_SPACING = 40
LEFT_PANEL_SPACING = 40

# Guard Square Location/Offsets
SQUARE_OFFSETS = {
    "Jab": (15, -50),
    "Left Hook": (15, -50),
    "Right Hook": (15, -50),
    "Uppercut": (15, -50),
    "Body Shots": (15, -50),
}

# Predictability Graph Height
GRAPH_HEIGHT = 1400

Window.size = (WINDOW_W, WINDOW_H)

# Camera Setup and Position/UI
class CameraWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cap = cv2.VideoCapture(CAM_INDEX)
        self._head_history = deque(maxlen=30)

        Clock.schedule_interval(self.update, 1 / 60)

        with self.canvas:
            StencilPush()
            self.stencil_rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[CAMERA_RADIUS])
            StencilUse()
            Color(1, 1, 1, 1)
            self.rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[CAMERA_RADIUS])
            StencilUnUse()
            StencilPop()

        self.bind(pos=self._sync_rect, size=self._sync_rect)

    def _sync_rect(self, *args):
        self.stencil_rect.pos = self.pos
        self.stencil_rect.size = self.size
        self.rect.pos = self.pos
        self.rect.size = self.size

    def update(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Method used to analyze guard
        drawn_frame, guard_results = image_guard_analysis(frame, draw=True)

        # Method used to predict head movement based on previous positions
        score, self._head_history, drawn_frame = analyze_frame(
            drawn_frame, history=self._head_history
        )

        if hasattr(self, "control_panel"):
            self.control_panel.update_squares_from_results(guard_results)

        if hasattr(self, "stats_panel"):
            self.stats_panel.update_predictability(score >= 0.5)

        frame_rgb = cv2.cvtColor(drawn_frame, cv2.COLOR_BGR2RGB)
        fh, fw, _ = frame_rgb.shape
        scale = min(self.width / fw, self.height / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        frame_rgb = cv2.resize(frame_rgb, (nw, nh))

        texture = Texture.create(size=(nw, nh), colorfmt='rgb')
        texture.blit_buffer(frame_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.rect.texture = texture
        self.rect.size = (nw, nh)
        self.rect.pos = (self.x, self.y + (self.height - nh) / 2)

# Left Panel Code Including Guard Susceptibility
class ControlPanel(BoxLayout):
    def __init__(self, camera, **kwargs):
        super().__init__(
            orientation='vertical',
            size_hint_x=None,
            width=SIDE_PANEL_W,
            padding=40,
            spacing=LEFT_PANEL_SPACING,
            **kwargs
        )

        self.camera = camera
        self.camera.control_panel = self

        with self.canvas.before:
            Color(*BG_GREY)
            self.rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[RADIUS])
        self.bind(pos=self._sync_rect, size=self._sync_rect)

        self.title = Label(
            text="Guard Susceptibility",
            font_size='36sp',
            bold=True,
            size_hint=(1, None),
            height=100,
            halign='center',
            valign='middle'
        )
        self.title.bind(size=lambda s, _: setattr(s, 'text_size', s.size))
        self.add_widget(self.title)

        self.image_container = FloatLayout(size_hint_y=1)
        self.panel_image = Image(
            source='BodyOutline.png',
            allow_stretch=True,
            keep_ratio=True
        )
        self.image_container.add_widget(self.panel_image)
        self.add_widget(self.image_container)

        self.squares = []
        self.square_size = (150, 150)
        self.square_texts = list(SQUARE_OFFSETS.keys())

        for text in self.square_texts:
            with self.image_container.canvas:
                c = Color(0.5, 0.5, 0.5, 0.5)
                r = Rectangle(size=self.square_size)
                label = CoreLabel(text=text, font_size=24, color=(1, 1, 1, 1))
                label.refresh()
                t_rect = Rectangle(texture=label.texture, size=label.texture.size)
            self.squares.append((c, r, label, t_rect))

        self._history = [deque(maxlen=30) for _ in self.squares]
        self.image_container.bind(size=self._update_overlay, pos=self._update_overlay)

    def _sync_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def _update_overlay(self, *args):
        img_w, img_h = self.panel_image.texture_size
        scale = min(self.image_container.width / img_w, self.image_container.height / img_h)
        w, h = img_w * scale, img_h * scale
        x = (self.image_container.width - w) / 2
        y = (self.image_container.height - h) / 2

        base_positions = {
            "Jab":        (x + w / 2 - 75, y + h - 140),
            "Left Hook":  (x + w / 4 - 75, y + h - 140),
            "Right Hook": (x + 3 * w / 4 - 75, y + h - 140),
            "Uppercut":   (x + w / 2 - 75, y + h - 305),
            "Body Shots": (x + w / 2 - 75, y + h - 470),
        }

        for (c, r, label, t_rect), name in zip(self.squares, self.square_texts):
            dx, dy = SQUARE_OFFSETS[name]
            px, py = base_positions[name]
            r.pos = (px + dx, py + dy)
            t_rect.pos = (
                r.pos[0] + 75 - t_rect.size[0] / 2,
                r.pos[1] + 75 - t_rect.size[1] / 2
            )

    def set_square_color(self, i, good):
        color, *_ = self.squares[i]
        color.rgb = (0, 1, 0) if good else (1, 0, 0)
        color.a = 0.5

    def update_squares_from_results(self, results):
        if results is None:
            results = [False] * len(self.squares)

        for i, state in enumerate(results):
            self._history[i].append(state)
            if state:
                self.set_square_color(i, True)
            elif self._history[i].count(False) / len(self._history[i]) >= 0.7:
                self.set_square_color(i, False)

# Right side panel that has information about head movement and its predictibility
class StatsPanel(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(
            orientation='vertical',
            size_hint_x=None,
            width=SIDE_PANEL_W,
            padding=40,
            spacing=RIGHT_PANEL_SPACING,
            **kwargs
        )

        with self.canvas.before:
            Color(*BG_GREY)
            self.rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[RADIUS])
        self.bind(pos=self._sync_rect, size=self._sync_rect)

        self.title = Label(
            text="Head Movement",
            font_size='36sp',
            bold=True,
            size_hint=(1, None),
            height=70 + RIGHT_TITLE_Y_OFFSET,
            halign='center',
            valign='middle'
        )
        self.title.bind(size=lambda s, _: setattr(s, 'text_size', s.size))
        self.add_widget(self.title)

        # Creates history of previous head movement positions to use in prediction
        self.recent_predictions = deque(maxlen=ROLLING_PRED_WINDOW)
        self.graph_values = deque(maxlen=GRAPH_HISTORY_POINTS)

        self.predict_label = Label(
            text="Predictability: 0%",
            font_size='32sp',
            size_hint=(1, None),
            height=80,
            halign='center',
            valign='middle'
        )
        self.predict_label.bind(size=lambda s, _: setattr(s, 'text_size', s.size))
        self.add_widget(self.predict_label)

        self.bar_container = Widget(size_hint=(1, None), height=80)
        self.add_widget(self.bar_container)

        with self.bar_container.canvas:
            self.bar_bg = Color(0.25, 0.25, 0.25, 1)
            self.bar_bg_rect = RoundedRectangle(radius=[20])
            self.bar_color = Color(0, 1, 0, 0.9)
            self.bar_rect = RoundedRectangle(radius=[20])

        self.bar_container.bind(pos=self._update_bar, size=self._update_bar)

        self.graph_title = Label(
            text="Predictability over time",
            font_size='24sp',
            size_hint=(1, None),
            height=40,
            halign='center',
            valign='middle'
        )
        self.graph_title.bind(size=lambda s, _: setattr(s, 'text_size', s.size))
        self.add_widget(self.graph_title)

        self.line_graph = Widget(size_hint=(1, None), height=GRAPH_HEIGHT)
        self.add_widget(self.line_graph)

    def _sync_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def _update_bar(self, *args):
        pad = 10
        self.bar_bg_rect.pos = (self.bar_container.x + pad, self.bar_container.y + pad)
        self.bar_bg_rect.size = (
            self.bar_container.width - 2 * pad,
            self.bar_container.height - 2 * pad
        )

        if not self.recent_predictions:
            percent = 0
        else:
            percent = sum(self.recent_predictions) / len(self.recent_predictions)

        self.bar_rect.pos = self.bar_bg_rect.pos
        self.bar_rect.size = (
            self.bar_bg_rect.size[0] * percent,
            self.bar_bg_rect.size[1]
        )

        if percent < 0.10:
            self.bar_color.rgb = (0, 1, 0)
        elif percent < 0.30:
            self.bar_color.rgb = (1, 0.7, 0)
        else:
            self.bar_color.rgb = (1, 0, 0)

    def update_predictability(self, accurate):
        self.recent_predictions.append(1 if accurate else 0)

        percent = sum(self.recent_predictions) / len(self.recent_predictions)
        percent_display = percent * 100

        self.predict_label.text = f"Predictability: {percent_display:.1f}%"

        # store rolling percentage for graph
        self.graph_values.append(percent)

        self._update_bar()
        self._draw_line_graph()

    def _draw_line_graph(self):
        self.line_graph.canvas.clear()
        if len(self.graph_values) < 2:
            return

        with self.line_graph.canvas:
            Color(0.4, 0.4, 0.4, 0.4)
            for i in range(5):
                y = self.line_graph.y + i * self.line_graph.height / 4
                Line(points=[self.line_graph.x, y,
                             self.line_graph.right, y], width=1)

            Color(0.2, 0.8, 1, 1)
            points = []
            w, h = self.line_graph.width, self.line_graph.height

            for i, v in enumerate(self.graph_values):
                x = self.line_graph.x + (i / (len(self.graph_values) - 1)) * w
                y = self.line_graph.y + v * h
                points.extend([x, y])

            Line(points=points, width=2)

# Main Root Layout Setup
class RootLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='horizontal', spacing=10, **kwargs)

        camera = CameraWidget(size_hint=(1, 1))
        panel = ControlPanel(camera)
        stats = StatsPanel()

        camera.stats_panel = stats

        self.add_widget(panel)
        self.add_widget(camera)
        self.add_widget(stats)

# Running Application
class CameraApp(App):
    def build(self):
        return RootLayout()

if __name__ == '__main__':
    CameraApp().run()
