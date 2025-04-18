# 3D Implicit Surface Voxel Plotter

## Description

This application visualizes implicit mathematical surfaces defined by the equation $f(x,y,z) = 0$. Users can input their own functions via a graphical user interface. The application generates a 3D representation using voxel cubes (filling grid cells intersected by the surface) and renders it using PyOpenGL. The rendering employs Z-height based shading, where the color brightness corresponds to the Z-coordinate. Mesh generation is performed asynchronously to keep the UI responsive.

*Developed with assistance from Google Gemini.*

## Features

* Parses user-defined implicit functions $f(x,y,z) = 0$.
* Generates 3D geometry using voxelization (filling intersected grid cubes).
* Renders the geometry using PyOpenGL with GLSL shaders.
* Visualizes using Z-height based shading (color brightness ~ Z coordinate).
* Asynchronous mesh generation using Python's `threading` to prevent UI freezes during updates.
* Interactive 3D camera with mouse-look rotation and keyboard movement.
* User interface built with PyOpenGL and ImGui for:
    * Entering mathematical expressions.
    * Adjusting voxel grid resolution.
    * Adjusting the overall visual scale of the rendered scene.
* Displays reference grid and coordinate axes.

## Technologies Used

* **Python 3** (Developed with 3.11)
* **Pygame (pygame-ce):** Window creation, event handling, OpenGL context management.
* **PyOpenGL:** Python bindings for OpenGL API calls.
* **GLSL:** OpenGL Shading Language for vertex and fragment shaders.
* **NumPy:** Numerical operations, especially for array handling in mesh generation and math functions.
* **asteval:** Safely evaluates mathematical expression strings.
* **Dear ImGui (pyimgui):** Immediate mode graphical user interface integrated with PyOpenGL.

## Installation

1.  **Ensure Python 3 is installed.** (Preferably 3.8 or newer).
2.  **Clone the repository (or download the source code).**
3.  **Install required libraries:** Open a terminal or command prompt, navigate to the project directory, and run:
    ```bash
    pip install pygame-ce numpy asteval PyOpenGL PyOpenGL-accelerate "imgui[pygame]"
    ```

## Usage

1.  **Run the script:**
    ```bash
    python main.py
    ```
    (Replace `main.py` with the actual name of your script file if different).

2.  **Controls:**
    * **Mouse:** Look around (rotate camera view). Initially hidden and grabbed.
    * **W, A, S, D:** Move camera forward, left, backward, right (relative to view direction, horizontally).
    * **Spacebar:** Move camera straight up.
    * **Shift (Left or Right):** Move camera straight down.
    * **ESC:** Toggle mouse grab. Press once to release the mouse (cursor appears, camera rotation stops, movement stops) allowing UI interaction. Press again to re-grab the mouse for camera control.
    * **Left Mouse Click (when mouse is released):** Re-grabs the mouse for camera control.
    * **Q:** Quit the application (only when UI doesn't have keyboard focus).

3.  **Interface (ImGui Window):**
    * **`f(x,y,z) =` [ Text Box ]:** Enter your mathematical expression here.
        * Use `x`, `y`, `z` as variables.
        * Use `**` for exponentiation (e.g., `x**2`). The `^` symbol is automatically converted.
        * Use `*` for multiplication (e.g., `3*x`). Implied multiplication like `3x` is automatically converted.
        * Standard functions like `sin()`, `cos()`, `sqrt()`, `exp()`, `log()` are available. Use parentheses for function arguments (e.g., `sin(x)`).
        * Constants `pi` and `e` are available.
    * **`[ Update Graph ]` Button:** Click this (or press Enter while the text box is focused) to parse the expression and regenerate the voxel mesh. Calculation happens in the background.
    * **`Res: [-] [ 50 ] [+]`:** Adjust the voxel grid resolution. Higher values produce more detail but take longer to calculate. Click `-` / `+` to change by 5. Triggers recalculation.
    * **`Scale: [-] [ 1.0 ] [+]`:** Adjust the overall visual scale of the rendered graph, grid, and axes. Click `-` / `+` to change by 0.1. Does *not* trigger recalculation.

4.  **Status Display:**
    * A small "Calculating Mesh..." overlay appears in the bottom-left during background calculations.

## Notes

* The application uses a fixed rendering range of -10 to +10 for X, Y, and Z axes. Functions extending beyond this range will be clipped during generation.
* The Z-height shading maps the Z coordinate within the fixed range to brightness.
* Performance depends heavily on the chosen resolution and the complexity of the function. Asynchronous calculation prevents the UI from freezing, but the visual update will only occur after the calculation completes.
