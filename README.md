# RubixCube — 3D Rubik’s Cube (Python + Pygame + OpenGL)

RubixCube is a **3D, interactive Rubik’s Cube game** written in Python. It renders a full 3×3×3 cube using **PyOpenGL** inside a **Pygame** window, supports smooth **animated face turns**, and includes quality-of-life features like **scramble**, **move counter**, **solve timer**, **camera orbit/zoom**, and **adjustable animation speed**.

This project is focused on the **simulation + interaction** side (turning faces correctly, animating layers, and visualizing the cube), rather than implementing an auto-solver.

---

## Features

- **True 3×3×3 cubie model**
  - Each cubie stores its 3D position and sticker colors.
  - Face turns rotate only the correct layer, then “lock in” the final 90° state.

- **Smooth animations**
  - Face turns animate over multiple frames.
  - A move queue allows you to input moves while a rotation is still in progress.

- **Scramble + solve tracking**
  - Press one key to generate a scramble sequence and apply it.
  - Tracks **moves** and **time**, and shows a “cube solved” message when completed.

- **3D camera controls**
  - Mouse drag to orbit the view.
  - Scroll wheel to zoom.
  - Reset the camera to a clean default angle.

- **On-screen HUD**
  - Displays move count, time, speed, scramble text, queued moves, and a quick control legend.

- **Resizable window**
  - The OpenGL projection and viewport update when the window is resized.

---

## Controls

| Action | Input |
|---|---|
| Rotate faces clockwise | `U` `D` `L` `R` `F` `B` |
| Rotate faces counter-clockwise | `Shift` + (`U` `D` `L` `R` `F` `B`) |
| Scramble | `S` |
| Reset cube (solved state) | `Space` |
| Reset camera | `C` |
| Increase / decrease animation speed | `+` / `-` |
| Orbit camera | Left mouse drag |
| Zoom | Mouse wheel |
| Quit | `Esc` |

---

## How it works (high level)

### Cube representation
The cube is built from 26 “cubies” (the center is omitted). Each cubie contains:
- A position vector `(x, y, z)` in `{-1, 0, 1}`
- Six sticker slots (Up/Down/Right/Left/Front/Back)

Stickers are initialized by cubie position (for example, cubies on the top layer get a white “Up” sticker).

### Face rotations (the important part)
A face turn is implemented as:
1. Select the cubies on the rotated layer (e.g., all cubies where `y == 1` for `U`).
2. Animate a rotation angle until it reaches 90°.
3. When the animation finishes:
   - Apply a 90° rotation matrix to cubie positions (then round back to exact layer coordinates).
   - Remap stickers by rotating face normals and reassigning sticker indices accordingly.

This ensures the cube’s state stays consistent and doesn’t “drift” over time.

### Solved detection
The cube is considered solved when each face’s layer has **one uniform sticker color** (and no “blank” sticker values).

### Rendering
Rendering uses OpenGL’s fixed pipeline lighting and draws each cubie’s six faces as quads:
- Sticker colors are drawn as solid colored faces
- A thin black outline is drawn around visible stickers to improve readability

### HUD / overlay text
HUD text is rendered using Pygame fonts and then drawn into the OpenGL scene as textured quads (so you get crisp 2D text over a 3D view).

---

## Requirements

You’ll need:
- **Python 3**
- **pygame**
- **PyOpenGL**
- **numpy**
- A machine/environment with working **OpenGL** support

---

## Project layout

- `rubixcube.py` — the entire game (cube model, animation system, OpenGL renderer, camera, HUD, and main loop)

---

## Notes / limitations

- This is a **manual** Rubik’s Cube simulator (no automatic solving algorithm included).
- Performance depends on your system’s OpenGL support and drivers.
- The HUD displays the scramble sequence; long scrambles are wrapped into multiple lines.

---

## Roadmap ideas (optional improvements)

- Add a “notation input” panel (type a full algorithm like `R U R' U'`)
- Add undo/redo (reverse the move history)
- Add selectable scramble lengths and scramble types
- Add cubie picking (click a face to rotate like modern cube simulators)
- Add an optional “ghost” layer highlight so it’s clearer which slice is turning

---

## License

No license file is included yet.

---
