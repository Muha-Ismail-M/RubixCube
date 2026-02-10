#!/usr/bin/env python3
"""
3D Rubik's Cube Game
====================
Controls:
    Mouse drag (left): Rotate camera
    Scroll: Zoom
    U/D/L/R/F/B: Rotate faces clockwise
    Shift + key: Counter-clockwise
    S: Scramble
    Space: Reset
    C: Reset camera
    +/-: Animation speed
    Esc: Quit

Requirements:
    pip install pygame PyOpenGL numpy
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import random
import time
from typing import List, Tuple, Dict


# =============================================================================
# Constants
# =============================================================================
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60
CUBIE_SIZE = 0.85
GAP = 1.0

COLOR_MAP = {
    'W': (1.0, 1.0, 1.0),
    'Y': (1.0, 1.0, 0.0),
    'R': (0.8, 0.0, 0.0),
    'O': (1.0, 0.5, 0.0),
    'B': (0.0, 0.0, 0.9),
    'G': (0.0, 0.6, 0.0),
    'K': (0.1, 0.1, 0.1),
}

# Face definitions: (axis_index, sign, label)
# 0=U(+Y), 1=D(-Y), 2=R(+X), 3=L(-X), 4=F(+Z), 5=B(-Z)
FACE_DEFS = [
    (1, 1, 'U'),
    (1, -1, 'D'),
    (0, 1, 'R'),
    (0, -1, 'L'),
    (2, 1, 'F'),
    (2, -1, 'B'),
]

MOVE_INFO = {
    'U': {'axis': 1, 'layer': 1, 'cw_dir': -1},
    'D': {'axis': 1, 'layer': -1, 'cw_dir': 1},
    'R': {'axis': 0, 'layer': 1, 'cw_dir': -1},
    'L': {'axis': 0, 'layer': -1, 'cw_dir': 1},
    'F': {'axis': 2, 'layer': 1, 'cw_dir': -1},
    'B': {'axis': 2, 'layer': -1, 'cw_dir': 1},
}


# =============================================================================
# Cubie
# =============================================================================
class Cubie:
    """One small cube with position and 6 sticker colors."""

    def __init__(self, x, y, z):
        self.pos = np.array([float(x), float(y), float(z)])
        # Stickers indexed matching FACE_DEFS: [U, D, R, L, F, B]
        self.stickers = []
        for axis_i, sign_i, label in FACE_DEFS:
            self.stickers.append(self._initial_color(x, y, z, label))

    def _initial_color(self, x, y, z, label):
        if label == 'U' and y == 1:
            return 'W'
        if label == 'D' and y == -1:
            return 'Y'
        if label == 'R' and x == 1:
            return 'R'
        if label == 'L' and x == -1:
            return 'O'
        if label == 'F' and z == 1:
            return 'B'
        if label == 'B' and z == -1:
            return 'G'
        return 'K'


# =============================================================================
# Rotation Math
# =============================================================================
def make_rotation_matrix_90(axis_index, direction):
    """
    3x3 rotation matrix for 90 degrees.
    axis_index: 0=X, 1=Y, 2=Z
    direction: +1 or -1
    """
    # cos(90)=0, sin(90)=1
    c = 0
    s = direction  # +1 or -1

    if axis_index == 0:
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ], dtype=float)
    elif axis_index == 1:
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ], dtype=float)
    else:
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ], dtype=float)


# =============================================================================
# RubiksCube
# =============================================================================
class RubiksCube:
    """Full 3x3x3 Rubik's Cube with animation support."""

    def __init__(self):
        self.cubies = []
        self.move_count = 0
        self.move_history = []

        # Animation state
        self.animating = False
        self.anim_axis_index = 0
        self.anim_axis_vec = np.array([0.0, 0.0, 0.0])
        self.anim_layer = 0
        self.anim_dir = 1
        self.anim_angle = 0.0
        self.anim_target_angle = 0.0
        self.anim_cubies = []
        self.anim_speed = 5.0

        self.move_queue = []

        self._create_cubies()

    def _create_cubies(self):
        self.cubies = []
        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    self.cubies.append(Cubie(x, y, z))

    def reset(self):
        self._create_cubies()
        self.move_count = 0
        self.move_history = []
        self.animating = False
        self.move_queue = []
        self.anim_angle = 0.0
        self.anim_cubies = []

    def _get_layer_cubies(self, axis_index, layer_val):
        result = []
        for c in self.cubies:
            if int(round(c.pos[axis_index])) == layer_val:
                result.append(c)
        return result

    def start_move(self, face, clockwise=True):
        if self.animating:
            self.move_queue.append((face, clockwise))
            return

        info = MOVE_INFO[face]
        ax = info['axis']
        layer = info['layer']
        cw_dir = info['cw_dir']
        direction = cw_dir if clockwise else -cw_dir

        self.anim_axis_index = ax
        axis_vec = np.zeros(3)
        axis_vec[ax] = 1.0
        self.anim_axis_vec = axis_vec
        self.anim_layer = layer
        self.anim_dir = direction
        self.anim_cubies = self._get_layer_cubies(ax, layer)
        self.anim_angle = 0.0
        self.anim_target_angle = 90.0 * direction
        self.animating = True

        move_name = face if clockwise else face + "'"
        self.move_history.append(move_name)
        self.move_count += 1

    def _finalize_move(self):
        """Apply the 90-degree rotation to cubie data when animation ends."""
        rot_90 = make_rotation_matrix_90(self.anim_axis_index, self.anim_dir)

        for cubie in self.anim_cubies:
            # Rotate position
            cubie.pos = rot_90 @ cubie.pos
            cubie.pos = np.array([round(cubie.pos[i]) for i in range(3)], dtype=float)

            # Remap stickers
            old_stickers = list(cubie.stickers)
            for i, (axis_i, sign_i, label_i) in enumerate(FACE_DEFS):
                normal = np.zeros(3)
                normal[axis_i] = float(sign_i)
                # Find where this normal came from before rotation
                inv_rot = rot_90.T
                old_normal = inv_rot @ normal
                old_idx = self._find_face_index(old_normal)
                cubie.stickers[i] = old_stickers[old_idx]

        self.animating = False
        self.anim_cubies = []
        self.anim_angle = 0.0

        # Process queue
        if self.move_queue:
            face, cw = self.move_queue.pop(0)
            self.start_move(face, cw)

    def _find_face_index(self, normal):
        """Find which FACE_DEFS index matches a normal vector."""
        for i, (axis_i, sign_i, label_i) in enumerate(FACE_DEFS):
            expected = np.zeros(3)
            expected[axis_i] = float(sign_i)
            if np.allclose(normal, expected, atol=0.5):
                return i
        return 0

    def update(self):
        if not self.animating:
            if self.move_queue:
                face, cw = self.move_queue.pop(0)
                self.start_move(face, cw)
            return

        remaining = self.anim_target_angle - self.anim_angle
        step = self.anim_speed

        if abs(remaining) <= step:
            self.anim_angle = self.anim_target_angle
            self._finalize_move()
        else:
            direction = 1 if remaining > 0 else -1
            self.anim_angle += step * direction

    def is_solved(self):
        if self.animating or self.move_queue:
            return False

        for face_idx, (axis_i, sign_i, label) in enumerate(FACE_DEFS):
            layer_val = sign_i
            layer_cubies = self._get_layer_cubies(axis_i, layer_val)
            colors = set()
            for cubie in layer_cubies:
                c = cubie.stickers[face_idx]
                if c == 'K':
                    return False
                colors.add(c)
            if len(colors) != 1:
                return False
        return True

    def scramble(self, num_moves=25):
        faces = list(MOVE_INFO.keys())
        moves = []
        last_face = None
        for _ in range(num_moves):
            face = random.choice(faces)
            while face == last_face:
                face = random.choice(faces)
            clockwise = random.choice([True, False])
            self.move_queue.append((face, clockwise))
            move_name = face if clockwise else face + "'"
            moves.append(move_name)
            last_face = face
        return moves

    def set_speed(self, speed):
        self.anim_speed = max(1.0, min(45.0, speed))


# =============================================================================
# Renderer
# =============================================================================
class Renderer:
    """OpenGL rendering."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self._init_gl()

    def _init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)

        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 10.0, 7.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.9, 0.9, 0.9, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])

        glLightfv(GL_LIGHT1, GL_POSITION, [-5.0, -3.0, 5.0, 1.0])
        glLightfv(GL_LIGHT1, GL_AMBIENT, [0.05, 0.05, 0.05, 1.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.3, 0.3, 0.3, 1.0])

        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glClearColor(0.15, 0.15, 0.22, 1.0)
        self.setup_projection()

    def setup_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / max(self.height, 1), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def resize(self, w, h):
        self.width = w
        self.height = h
        glViewport(0, 0, w, h)
        self.setup_projection()

    def begin_frame(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

    def render_cube(self, cube):
        for cubie in cube.cubies:
            is_anim = cube.animating and cubie in cube.anim_cubies
            self._render_cubie(cubie, cube, is_anim)

    def _render_cubie(self, cubie, cube, is_animating):
        glPushMatrix()

        if is_animating:
            ax = cube.anim_axis_vec
            glRotatef(cube.anim_angle, ax[0], ax[1], ax[2])

        glTranslatef(cubie.pos[0] * GAP, cubie.pos[1] * GAP, cubie.pos[2] * GAP)

        h = CUBIE_SIZE / 2.0

        for face_idx, (axis_i, sign_i, label) in enumerate(FACE_DEFS):
            color_key = cubie.stickers[face_idx]
            color = COLOR_MAP.get(color_key, COLOR_MAP['K'])

            normal = [0.0, 0.0, 0.0]
            normal[axis_i] = float(sign_i)

            verts = self._face_verts(label, h)

            glColor3f(color[0], color[1], color[2])
            glBegin(GL_QUADS)
            glNormal3f(normal[0], normal[1], normal[2])
            for v in verts:
                glVertex3f(v[0], v[1], v[2])
            glEnd()

            # Black border for visible stickers
            if color_key != 'K':
                glDisable(GL_LIGHTING)
                glColor3f(0.0, 0.0, 0.0)
                glLineWidth(3.0)
                glBegin(GL_LINE_LOOP)
                ox = normal[0] * 0.002
                oy = normal[1] * 0.002
                oz = normal[2] * 0.002
                for v in verts:
                    glVertex3f(v[0] + ox, v[1] + oy, v[2] + oz)
                glEnd()
                glEnable(GL_LIGHTING)

        glPopMatrix()

    def _face_verts(self, label, h):
        if label == 'U':
            return [(-h, h, -h), (h, h, -h), (h, h, h), (-h, h, h)]
        elif label == 'D':
            return [(-h, -h, h), (h, -h, h), (h, -h, -h), (-h, -h, -h)]
        elif label == 'R':
            return [(h, -h, -h), (h, -h, h), (h, h, h), (h, h, -h)]
        elif label == 'L':
            return [(-h, -h, h), (-h, -h, -h), (-h, h, -h), (-h, h, h)]
        elif label == 'F':
            return [(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)]
        elif label == 'B':
            return [(h, -h, -h), (-h, -h, -h), (-h, h, -h), (h, h, -h)]
        return []


# =============================================================================
# Camera
# =============================================================================
class Camera:
    def __init__(self):
        self.distance = 10.0
        self.rot_x = 25.0
        self.rot_y = -35.0
        self.sensitivity = 0.3
        self.zoom_speed = 0.5

    def reset(self):
        self.distance = 10.0
        self.rot_x = 25.0
        self.rot_y = -35.0

    def rotate(self, dx, dy):
        self.rot_y += dx * self.sensitivity
        self.rot_x += dy * self.sensitivity
        self.rot_x = max(-89, min(89, self.rot_x))

    def zoom(self, amount):
        self.distance -= amount * self.zoom_speed
        self.distance = max(4.0, min(25.0, self.distance))

    def apply(self):
        glTranslatef(0, 0, -self.distance)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)


# =============================================================================
# TextDrawer - draws 2D text overlay via Pygame surfaces -> GL textures
# =============================================================================
class TextDrawer:
    """Renders 2D text as OpenGL textured quads. Caches textures."""

    def __init__(self):
        pygame.font.init()
        self.font_large = pygame.font.SysFont('Arial', 32, bold=True)
        self.font_medium = pygame.font.SysFont('Arial', 22)
        self.font_small = pygame.font.SysFont('Arial', 16)
        self._cache = {}
        self._max_cache = 200

    def draw(self, text, x, y, sw, sh, font=None, color=(255, 255, 255)):
        """Draw text at screen pixel (x,y) from top-left."""
        if not text or not text.strip():
            return 0
        if font is None:
            font = self.font_medium

        key = (text, id(font), color)

        if key in self._cache:
            tex_id, tw, th = self._cache[key]
        else:
            # Evict old entries if cache too big
            if len(self._cache) > self._max_cache:
                self._evict_cache()
            surface = font.render(text, True, color)
            data = pygame.image.tostring(surface, 'RGBA', True)
            tw, th = surface.get_size()
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, th, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, data)
            self._cache[key] = (tex_id, tw, th)

        # Draw textured quad
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, sw, sh, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindTexture(GL_TEXTURE_2D, tex_id)
        glColor4f(1, 1, 1, 1)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 1)
        glVertex2f(x, y)
        glTexCoord2f(1, 1)
        glVertex2f(x + tw, y)
        glTexCoord2f(1, 0)
        glVertex2f(x + tw, y + th)
        glTexCoord2f(0, 0)
        glVertex2f(x, y + th)
        glEnd()

        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        return tw

    def _evict_cache(self):
        """Remove oldest half of cache entries."""
        keys = list(self._cache.keys())
        remove = keys[:len(keys) // 2]
        for k in remove:
            tex_id, tw, th = self._cache[k]
            try:
                glDeleteTextures([tex_id])
            except Exception:
                pass
            del self._cache[k]

    def clear_cache(self):
        for tex_id, tw, th in self._cache.values():
            try:
                glDeleteTextures([tex_id])
            except Exception:
                pass
        self._cache = {}


# =============================================================================
# Game
# =============================================================================
class Game:
    def __init__(self):
        pygame.init()
        self.width = WINDOW_WIDTH
        self.height = WINDOW_HEIGHT

        self.screen = pygame.display.set_mode(
            (self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE
        )
        pygame.display.set_caption("3D Rubik's Cube")

        self.renderer = Renderer(self.width, self.height)
        self.cube = RubiksCube()
        self.camera = Camera()
        self.text = TextDrawer()
        self.clock = pygame.time.Clock()
        self.running = True

        # Mouse
        self.dragging = False
        self.last_mouse = (0, 0)

        # Timer
        self.timer_running = False
        self.timer_start = 0.0
        self.timer_value = 0.0
        self.solve_time = 0.0

        # State
        self.scramble_text = ""
        self.is_solved = True
        self.was_scrambled = False
        self.solve_notified = True
        self.show_solved_msg = False
        self.solved_msg_time = 0.0

    def run(self):
        print("=" * 50)
        print("  3D RUBIK'S CUBE")
        print("=" * 50)
        print("U/D/L/R/F/B = rotate face, Shift = reverse")
        print("S = scramble, Space = reset, C = reset camera")
        print("+/- = animation speed, Esc = quit")
        print("=" * 50)

        while self.running:
            self._handle_events()
            self._update()
            self._render()
            self.clock.tick(FPS)

        self.text.clear_cache()
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False

            elif event.type == KEYDOWN:
                self._on_key(event)

            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.dragging = True
                    self.last_mouse = event.pos
                elif event.button == 4:
                    self.camera.zoom(1)
                elif event.button == 5:
                    self.camera.zoom(-1)

            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False

            elif event.type == MOUSEMOTION:
                if self.dragging:
                    dx = event.pos[0] - self.last_mouse[0]
                    dy = event.pos[1] - self.last_mouse[1]
                    self.camera.rotate(dx, dy)
                    self.last_mouse = event.pos

            elif event.type == VIDEORESIZE:
                self.width = event.w
                self.height = event.h
                self.screen = pygame.display.set_mode(
                    (self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE
                )
                self.renderer.resize(self.width, self.height)
                self.text.clear_cache()

    def _on_key(self, event):
        mods = pygame.key.get_mods()
        shift = bool(mods & KMOD_SHIFT)

        face_keys = {
            K_u: 'U', K_d: 'D', K_l: 'L',
            K_r: 'R', K_f: 'F', K_b: 'B'
        }

        if event.key in face_keys:
            face = face_keys[event.key]
            clockwise = not shift
            self.cube.start_move(face, clockwise)

            if not self.timer_running and not self.is_solved:
                self.timer_running = True
                self.timer_start = time.time()
                self.solve_notified = False
                self.show_solved_msg = False

        elif event.key == K_s:
            if not self.cube.animating and not self.cube.move_queue:
                self.cube.reset()
                self.timer_value = 0.0
                self.timer_running = False
                self.is_solved = False
                self.was_scrambled = True
                self.solve_notified = False
                self.show_solved_msg = False
                moves = self.cube.scramble(25)
                self.scramble_text = " ".join(moves)
                print("Scramble: " + self.scramble_text)

        elif event.key == K_SPACE:
            self.cube.reset()
            self.timer_value = 0.0
            self.timer_running = False
            self.scramble_text = ""
            self.is_solved = True
            self.was_scrambled = False
            self.solve_notified = True
            self.show_solved_msg = False

        elif event.key == K_c:
            self.camera.reset()

        elif event.key in (K_PLUS, K_EQUALS, K_KP_PLUS):
            self.cube.set_speed(self.cube.anim_speed + 1.0)
        elif event.key in (K_MINUS, K_KP_MINUS):
            self.cube.set_speed(self.cube.anim_speed - 1.0)

        elif event.key == K_ESCAPE:
            self.running = False

    def _update(self):
        self.cube.update()

        if self.timer_running:
            self.timer_value = time.time() - self.timer_start

        if not self.cube.animating and not self.cube.move_queue:
            if self.was_scrambled:
                self.was_scrambled = False
                self.cube.move_count = 0
                self.cube.move_history = []

            self.is_solved = self.cube.is_solved()

            if self.is_solved and not self.solve_notified and self.cube.move_count > 0:
                self.timer_running = False
                self.solve_time = self.timer_value
                self.solve_notified = True
                self.show_solved_msg = True
                self.solved_msg_time = time.time()
                print("CUBE SOLVED! Time: {:.2f}s, Moves: {}".format(
                    self.solve_time, self.cube.move_count))

        if self.show_solved_msg and time.time() - self.solved_msg_time > 8.0:
            self.show_solved_msg = False

    def _render(self):
        self.renderer.begin_frame()
        self.camera.apply()
        self.renderer.render_cube(self.cube)
        self._render_hud()
        pygame.display.flip()

    def _format_time(self, seconds):
        mins = int(seconds) // 60
        secs = seconds % 60
        return "{:02d}:{:05.2f}".format(mins, secs)

    def _render_hud(self):
        w = self.width
        h = self.height
        y = 12

        self.text.draw("RUBIK'S CUBE", 12, y, w, h,
                        self.text.font_large, (100, 200, 255))
        y += 42

        self.text.draw("Moves: {}".format(self.cube.move_count),
                        12, y, w, h, self.text.font_medium, (230, 230, 230))
        y += 30

        tstr = "Time: " + self._format_time(self.timer_value)
        tcol = (100, 255, 100) if self.timer_running else (230, 230, 230)
        self.text.draw(tstr, 12, y, w, h, self.text.font_medium, tcol)
        y += 30

        self.text.draw("Speed: {} deg/frame".format(int(self.cube.anim_speed)),
                        12, y, w, h, self.text.font_small, (170, 170, 170))
        y += 24

        if self.scramble_text:
            self.text.draw("Scramble:", 12, y, w, h,
                            self.text.font_small, (255, 200, 100))
            y += 20
            st = self.scramble_text
            while len(st) > 0:
                line = st[:50]
                st = st[50:]
                self.text.draw(line, 22, y, w, h,
                                self.text.font_small, (255, 220, 150))
                y += 18

        if self.cube.move_queue:
            self.text.draw("Queue: {} moves".format(len(self.cube.move_queue)),
                            12, y, w, h, self.text.font_small, (255, 150, 150))

        # Solved message
        if self.show_solved_msg:
            self.text.draw("*** CUBE SOLVED! ***",
                            w // 2 - 160, h // 2 - 40, w, h,
                            self.text.font_large, (50, 255, 50))
            st2 = "Time: {}  Moves: {}".format(
                self._format_time(self.solve_time), self.cube.move_count)
            self.text.draw(st2, w // 2 - 140, h // 2 + 10, w, h,
                            self.text.font_medium, (255, 255, 100))

        # Controls at bottom-right
        controls = [
            "--- Controls ---",
            "U/D/L/R/F/B: Rotate",
            "Shift+key: Reverse",
            "S: Scramble",
            "Space: Reset",
            "C: Reset camera",
            "+/-: Speed",
            "Scroll: Zoom",
            "Esc: Quit",
        ]
        cy = h - len(controls) * 20 - 10
        for line in controls:
            col = (140, 140, 190) if line.startswith("---") else (130, 130, 155)
            self.text.draw(line, w - 220, cy, w, h, self.text.font_small, col)
            cy += 20


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    game = Game()
    game.run()