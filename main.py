# --- Imports ---
import threading
import queue
import re
import pygame
from pygame.locals import *
from OpenGL.GL import *
import os # Removed - Unused
import math
import numpy as np
import asteval
import traceback
import imgui
from imgui.integrations.pygame import PygameRenderer
# import marching_cubes_tables # Keep if GraphFunction uses it
import ctypes

# --- GLSL Shaders ---
VERTEX_SHADER_SOURCE = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in float aHeight;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 FragPos;
out vec3 Normal;
out float vHeight;
void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = normalize(mat3(model) * aNormal);
    vHeight = aHeight;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

FRAGMENT_SHADER_SOURCE = """
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in float vHeight;
out vec4 FragColor;
uniform vec3 objectColor;
uniform float zMin;
uniform float zMax;
void main()
{
    float normalizedHeight = 0.5;
    if (zMax > zMin) {
      normalizedHeight = clamp((vHeight - zMin) / (zMax - zMin), 0.0, 1.0);
    }
    float brightness = 0.3 + normalizedHeight * 0.7;
    vec3 result = objectColor * brightness;
    FragColor = vec4(result, 1.0);
}
"""

SIMPLE_VERTEX_SHADER_SOURCE = """
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""
SIMPLE_FRAGMENT_SHADER_SOURCE = """
#version 330 core
out vec4 FragColor;
uniform vec3 lineColor;
void main()
{
    FragColor = vec4(lineColor, 1.0);
}
"""
# --- End GLSL Shaders ---


# --- Preprocessing Function ---
def preprocess_expression(raw_expr):
    if not isinstance(raw_expr, str): return ""
    processed = raw_expr.strip()
    if not processed: return ""
    # print(f"[Preprocess] Start: '{processed}'")
    processed = processed.replace('^', '**')
    # print(f"[Preprocess] After ^ -> ** : '{processed}'")
    pattern = re.compile(r'(-?(?:\d+\.\d*|\.\d+|\d+))\s*([xyz])')
    processed = pattern.sub(r'\1 * \2', processed)
    # print(f"[Preprocess] After num*[xyz]: '{processed}'")
    # print(f"[Preprocess] Final: '{processed}'")
    return processed

# --- OpenGL Helper Functions ---
# (Assuming create_opengl_mesh, compile_shader, create_shader_program,
#  create_perspective_matrix_gl, create_look_at_matrix_gl are defined here
#  or imported, and are correct)
def create_opengl_mesh(vertex_data, attribute_layout, indices=None, usage=GL_STATIC_DRAW, draw_mode=GL_TRIANGLES):
    vao_id, vbo_id, ebo_id = None, None, None
    element_count = 0
    if vertex_data is None or vertex_data.size == 0: return None, None, None, 0, draw_mode
    if not attribute_layout: return None, None, None, 0, draw_mode
    try:
        itemsize = vertex_data.itemsize
        last_attrib_loc, last_attrib_size, last_attrib_offset = attribute_layout[-1]
        stride = last_attrib_offset + (last_attrib_size * itemsize)
        components_per_vertex = stride // itemsize
        if vertex_data.size % components_per_vertex != 0: raise ValueError("Vertex data size mismatch.")
        vao_id = glGenVertexArrays(1); glBindVertexArray(vao_id)
        vbo_id = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, usage)
        for loc, size, offset_bytes in attribute_layout:
            offset_ptr = ctypes.c_void_p(offset_bytes)
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, stride, offset_ptr)
        if indices is not None and indices.size > 0 :
            ebo_id = glGenBuffers(1); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_id)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, usage)
            element_count = len(indices)
        else: element_count = vertex_data.size // components_per_vertex
        glBindVertexArray(0); glBindBuffer(GL_ARRAY_BUFFER, 0)
        if ebo_id is not None: glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        # Minimal print on success
        # print(f"Created/Updated OpenGL Mesh: VAO={vao_id}, VBO={vbo_id}, EBO={ebo_id}, Count={element_count}")
        return vao_id, vbo_id, ebo_id, element_count, draw_mode
    except Exception as e:
        print(f"ERROR INSIDE create_opengl_mesh: {e}"); traceback.print_exc()
        if vao_id: glDeleteVertexArrays(1, [vao_id])
        if vbo_id: glDeleteBuffers(1, [vbo_id])
        if ebo_id: glDeleteBuffers(1, [ebo_id])
        return None, None, None, 0, draw_mode

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        info_log = glGetShaderInfoLog(shader).decode()
        shader_type_str = "Vertex" if shader_type == GL_VERTEX_SHADER else "Fragment"
        print(f"ERROR::SHADER::{shader_type_str}::COMPILATION_FAILED\n{info_log}")
        glDeleteShader(shader); return None
    return shader

def create_shader_program(vertex_source, fragment_source):
    vertex_shader = compile_shader(vertex_source, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_source, GL_FRAGMENT_SHADER)
    if vertex_shader is None or fragment_shader is None: return None
    program = glCreateProgram()
    glAttachShader(program, vertex_shader); glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        info_log = glGetProgramInfoLog(program).decode()
        print(f"ERROR::PROGRAM::LINKING_FAILED\n{info_log}")
        glDeleteShader(vertex_shader); glDeleteShader(fragment_shader); glDeleteProgram(program); return None
    glDeleteShader(vertex_shader); glDeleteShader(fragment_shader)
    print("Shader program created successfully.")
    return program

def create_perspective_matrix_gl(fov_rad, aspect, near, far):
    f = 1.0 / np.tan(fov_rad / 2.0)
    fn = far + near; nf = near - far
    if abs(nf) < 1e-9: nf = -1e-9
    proj = np.array([
        [f / aspect, 0, 0, 0], [0, f, 0, 0],
        [0, 0, fn / nf, (2 * far * near) / nf], [0, 0, -1, 0]
    ], dtype=np.float32)
    return proj.T

def create_look_at_matrix_gl(eye, target, world_up):
    eye = np.asarray(eye, dtype=np.float32); target = np.asarray(target, dtype=np.float32); world_up = np.asarray(world_up, dtype=np.float32)
    f = target - eye; norm = np.linalg.norm(f); f = f / norm if norm > 1e-8 else np.array([0,0,-1], dtype=np.float32)
    s = np.cross(f, world_up); norm = np.linalg.norm(s)
    if norm < 1e-8: s = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else: s /= norm
    u = np.cross(s, f)
    R = np.identity(4, dtype=np.float32); R[0, :3] = s; R[1, :3] = u; R[2, :3] = -f
    T = np.identity(4, dtype=np.float32); T[:3, 3] = -eye
    view_matrix = R @ T
    return view_matrix

# --- Data Generation Helper ---
def create_voxel_data(expression, x_range, y_range, z_range, resolution, aeval_instance):
    # print(f"Creating voxel grid ({resolution}x{resolution}x{resolution})...")
    try:
        x_vals = np.linspace(x_range[0], x_range[1], resolution, dtype=np.float32)
        y_vals = np.linspace(y_range[0], y_range[1], resolution, dtype=np.float32)
        z_vals = np.linspace(z_range[0], z_range[1], resolution, dtype=np.float32)
        x_corners, y_corners, z_corners = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
        update_symbols = {'x': x_corners, 'y': y_corners, 'z': z_corners}
        aeval_instance.symtable.update(update_symbols)
        # print(f"Evaluating '{expression}' across grid...")
        scalar_values = aeval_instance.eval(expression)
        if aeval_instance.error: raise ValueError(f"Asteval error evaluating '{expression}': {aeval_instance.error_msg.strip()}")
        if not isinstance(scalar_values, np.ndarray) or scalar_values.shape != x_corners.shape: raise ValueError("Evaluation result mismatch.")
        if np.iscomplexobj(scalar_values): scalar_values = np.real(scalar_values).astype(np.float32)
        else: scalar_values = scalar_values.astype(np.float32)
        # print("Voxel grid and scalar field created successfully.")
        return x_corners, y_corners, z_corners, scalar_values
    except Exception as e: print(f"Error creating voxel data: {e}"); traceback.print_exc(); return None, None, None, None

# --- Camera Class ---
class Camera:
    def __init__(self, position, yaw_deg, pitch_deg, fov_deg, aspect_ratio, near, far, move_speed, rot_speed_dps):
        self.position=np.array(position,dtype=np.float32); self.yaw=math.radians(yaw_deg); self.pitch=math.radians(pitch_deg)
        self.world_up=np.array([0.,0.,1.],dtype=np.float32); self.fov_degrees=fov_deg; self.aspect_ratio=aspect_ratio
        self.znear=near; self.zfar=far; self.movement_speed=move_speed; self.rotation_speed=math.radians(rot_speed_dps)
        self.forward=np.array([0.,0.,0.], dtype=np.float32); self.right=np.array([0.,0.,0.], dtype=np.float32); self.up=np.array([0.,0.,0.], dtype=np.float32)
        self._update_vectors()
    def _update_vectors(self):
        fwd_x = math.cos(self.yaw) * math.cos(self.pitch); fwd_y = math.sin(self.yaw) * math.cos(self.pitch); fwd_z = math.sin(self.pitch)
        self.forward = np.array([fwd_x, fwd_y, fwd_z], dtype=np.float32); norm_fwd = np.linalg.norm(self.forward); self.forward = self.forward / norm_fwd if norm_fwd > 1e-8 else self.forward
        self.right = np.cross(self.forward, self.world_up); norm_right = np.linalg.norm(self.right)
        if norm_right < 1e-8: self.right = np.array([1., 0., 0.], dtype=np.float32); 
        else: self.right /= norm_right
        self.up = np.cross(self.right, self.forward)
    def rotate(self, dyaw_rad, dpitch_rad):
        self.yaw += dyaw_rad; self.pitch += dpitch_rad
        self.pitch = max(-math.pi/2 + 0.01, min(math.pi/2 - 0.01, self.pitch)); self.yaw %= (2 * math.pi)
        self._update_vectors()
    def move(self, direction_vector, dt): self.position += direction_vector * self.movement_speed * dt
    def update_aspect_ratio(self, width, height): self.aspect_ratio = width / max(1, height)

# --- TriangleMesh Class (Keep if GraphFunction uses it, e.g. self.triangle_mesh) ---
class TriangleMesh:
     def __init__(self, vertices_3d=None, faces=None, color=(200, 200, 200)): self.vertices_3d = np.array(vertices_3d if vertices_3d is not None else [], dtype=np.float32).reshape(-1, 3); self.faces = np.array(faces if faces is not None else [], dtype=np.uint32).reshape(-1, 3); self.color = color
     def update_geometry(self, vertices_3d, faces): self.vertices_3d = np.array(vertices_3d, dtype=np.float32); self.faces = np.array(faces, dtype=np.uint32)

# --- Graph Function Class ---
class GraphFunction:
    def __init__(self, implicit_expression="x**2 + y**2 + z**2 - 16",
                 x_range=(-10.0, 10.0), y_range=(-10.0, 10.0), z_range=(-10.0, 10.0),
                 resolution=50, color=(150, 0, 200)):
        self.x_range = tuple(x_range); self.y_range = tuple(y_range); self.z_range = tuple(z_range)
        self.resolution = resolution; self.color = color
        self.aeval = asteval.Interpreter()
        self.aeval.symtable['pi'] = math.pi; self.aeval.symtable['e'] = math.e
        self.aeval.symtable['sin'] = np.sin; self.aeval.symtable['cos'] = np.cos # Add others if needed
        self.expression = preprocess_expression(implicit_expression)
        self.gl_vao, self.gl_vbo, self.gl_ebo, self.gl_element_count, self.gl_draw_mode = None, None, None, 0, GL_TRIANGLES
        self.triangle_mesh = TriangleMesh(color=self.color) # Optional python-side storage
        print(f"GraphFunction initialized for: f(x, y, z) = {self.expression} = 0")

    def set_parameters(self, resolution=None):
        changed = False
        if resolution is not None and resolution != self.resolution:
            self.resolution = resolution; changed = True
            print(f"GraphFunction: Resolution updated to {self.resolution}")
        return changed

    def set_implicit_expression(self, processed_expression_string):
        if processed_expression_string != self.expression:
            print(f"GraphFunction: Expression updated to '{processed_expression_string}'")
            self.expression = processed_expression_string
            return True
        return False

    def _calculate_mesh_data(self, expression, resolution, x_range, y_range, z_range):
        """
        Performs CPU-bound mesh data calculation based on PASSED parameters.
        Should be run in a background thread.
        Returns: Interleaved NumPy vertex data array (Pos, Norm, Height), or None on error.
        """
        print(f"BG THREAD: Calculating mesh for Res: {resolution}...")

        try:
            thread_aeval = asteval.Interpreter()
            thread_aeval.symtable['pi'] = math.pi; thread_aeval.symtable['e'] = math.e
            thread_aeval.symtable['sin'] = np.sin; thread_aeval.symtable['cos'] = np.cos
            thread_aeval.symtable['tan'] = np.tan; thread_aeval.symtable['sqrt'] = np.sqrt
            thread_aeval.symtable['exp'] = np.exp; thread_aeval.symtable['log'] = np.log
            thread_aeval.symtable['log10'] = np.log10; thread_aeval.symtable['abs'] = np.abs
        except Exception as e:
            print(f"BG THREAD Error creating local aeval: {e}"); return None

        start_time = pygame.time.get_ticks() 
        try:
            corner_x, corner_y, corner_z, scalar_field = create_voxel_data(
                expression, x_range, y_range, z_range,
                resolution, thread_aeval 
            )
            if scalar_field is None: return None
            nx, ny, nz = scalar_field.shape
            if nx < 2 or ny < 2 or nz < 2: print(f"Res ({resolution}) too low."); return None
        except Exception as e:
            print(f"BG THREAD ERROR during create_voxel_data: {e}"); traceback.print_exc(); return None

        unit_corners=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=np.float32)
        indices_face=np.array([1,5,6,1,6,2,0,3,7,0,7,4,3,2,6,3,6,7,0,4,5,0,5,1,4,7,6,4,6,5,0,1,2,0,2,3],dtype=int)
        unit_cube_verts_precomputed=unit_corners[indices_face]
        normals_face_list=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
        unit_cube_normals_precomputed=np.repeat(normals_face_list,6,axis=0).astype(np.float32)

        origins_list = []
        heights_list = []
        iso_level = 0.0
        try:
            for i in range(nx - 1):
                for j in range(ny - 1):
                    for k in range(nz - 1):
                        cube_corners_values=np.array([
                            scalar_field[i,j,k], scalar_field[i+1,j,k], scalar_field[i+1,j+1,k], scalar_field[i,j+1,k],
                            scalar_field[i,j,k+1], scalar_field[i+1,j,k+1], scalar_field[i+1,j+1,k+1], scalar_field[i,j+1,k+1]
                        ], dtype=np.float32)
                        cube_index=0
                        for ci in range(8): cube_index |= (1<<ci) if cube_corners_values[ci]<iso_level else 0
                        if 0 < cube_index < 255:
                            origins_list.append([corner_x[i,j,k], corner_y[i,j,k], corner_z[i,j,k]])
                            heights_list.append(corner_z[i,j,k])

            generated_cube_count = len(origins_list)
            if generated_cube_count == 0: print("BG THREAD: No intersecting cubes found."); return None

            active_origins_np = np.array(origins_list, dtype=np.float32)
            active_heights_np = np.array(heights_list, dtype=np.float32).reshape(-1, 1)
        except Exception as e: print(f"BG THREAD ERROR during cube scan: {e}"); traceback.print_exc(); return None

        try:
            N = generated_cube_count
            dx = (x_range[1] - x_range[0]) / (nx - 1) if nx > 1 else 0
            dy = (y_range[1] - y_range[0]) / (ny - 1) if ny > 1 else 0
            dz = (z_range[1] - z_range[0]) / (nz - 1) if nz > 1 else 0
            scaled_unit_verts = unit_cube_verts_precomputed * [dx, dy, dz]
            all_vertex_pos = active_origins_np[:, np.newaxis, :] + scaled_unit_verts[np.newaxis, :, :]
            all_vertex_pos = all_vertex_pos.reshape(N * 36, 3)
            all_normals = np.tile(unit_cube_normals_precomputed, (N, 1))
            all_heights = np.repeat(active_heights_np, 36, axis=0)
        except Exception as e: print(f"BG THREAD ERROR during vector math: {e}"); traceback.print_exc(); return None

        try:
            interleaved_data = np.hstack((all_vertex_pos, all_normals, all_heights)).astype(np.float32).ravel()
            total_time = pygame.time.get_ticks() - start_time
            print(f"BG THREAD: Calculation complete ({total_time:.0f} ms). Vertices: {interleaved_data.size // 7}")
            return interleaved_data
        except Exception as e: print(f"BG THREAD ERROR during interleaving: {e}"); traceback.print_exc(); return None

    def _cleanup_gl_resources(self):
        deleted = []; 
        try:
            if self.gl_vao: glDeleteVertexArrays(1, [self.gl_vao]); deleted.append(f"VAO={self.gl_vao}"); self.gl_vao = None
            if self.gl_vbo: glDeleteBuffers(1, [self.gl_vbo]); deleted.append(f"VBO={self.gl_vbo}"); self.gl_vbo = None
            self.gl_element_count = 0
            # if deleted: print(f"Cleaned up GraphFunction GL: {', '.join(deleted)}")
        except Exception as e: print(f"Error cleaning up GraphFunction GL resources: {e}")

# --- App Class ---
class App:
    # --- Init ---
    def __init__(self):
        pygame.init()
        os.environ["SDL_VIDEO_CENTERED"] = "1"
        self.width, self.height = 1024, 768
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_COMPATIBILITY)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        display_flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        try: self.screen = pygame.display.set_mode((self.width, self.height), display_flags)
        except pygame.error as e: print(f"Error setting display mode: {e}"); self.running = False; return
        pygame.display.set_caption("3D Voxel Plotter - OpenGL + ImGui")
        self.clock = pygame.time.Clock(); self.fps = 60
        self.bg_color = (1.0, 1.0, 1.0, 1.0); self.grid_color = (0.0, 0.0, 0.0); self.axes_color = (0.0, 0.0, 0.0); self.graph_color = (150, 0, 200)
        try:
            glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LESS); glClearColor(*self.bg_color)
            print("OpenGL Version:", glGetString(GL_VERSION)); print("GLSL Version:", glGetString(GL_SHADING_LANGUAGE_VERSION))
        except Exception as e: print(f"Error initializing OpenGL state: {e}"); self.running = False; return
        self.graph_x_range = (-10.0, 10.0); self.graph_y_range = (-10.0, 10.0); self.graph_z_range = (-10.0, 10.0)
        self.current_resolution = 50
        self.visual_scale = 1.0; self.scale_increment = 0.1; self.min_scale = 0.1
        self.calculation_active = False; self.calculation_thread = None; self.result_queue = queue.Queue(maxsize=1); self.pending_params = None
        self.shader_program = create_shader_program(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)
        if self.shader_program is None: self.running = False; return
        try:
            self.loc_model=glGetUniformLocation(self.shader_program,"model"); self.loc_view=glGetUniformLocation(self.shader_program,"view")
            self.loc_projection=glGetUniformLocation(self.shader_program,"projection"); self.loc_objectColor=glGetUniformLocation(self.shader_program,"objectColor")
            self.loc_zMin=glGetUniformLocation(self.shader_program,"zMin"); self.loc_zMax=glGetUniformLocation(self.shader_program,"zMax")
            # print(f"Main Shader Uniforms: model={self.loc_model}, view={self.loc_view}, proj={self.loc_projection}, objCol={self.loc_objectColor}, zMin={self.loc_zMin}, zMax={self.loc_zMax}")
            if any(loc == -1 for loc in [self.loc_model,self.loc_view,self.loc_projection,self.loc_objectColor,self.loc_zMin,self.loc_zMax]): print("Warning: Main shader uniform missing.")
        except Exception as e: print(f"Error getting main shader uniforms: {e}"); self.running = False; return
        self.simple_shader_program = create_shader_program(SIMPLE_VERTEX_SHADER_SOURCE, SIMPLE_FRAGMENT_SHADER_SOURCE)
        if self.simple_shader_program is None: self.running = False; return
        try:
            self.loc_simple_model=glGetUniformLocation(self.simple_shader_program,"model"); self.loc_simple_view=glGetUniformLocation(self.simple_shader_program,"view")
            self.loc_simple_projection=glGetUniformLocation(self.simple_shader_program,"projection"); self.loc_simple_lineColor=glGetUniformLocation(self.simple_shader_program,"lineColor")
            # print(f"Simple Shader Uniforms: model={self.loc_simple_model}, view={self.loc_simple_view}, proj={self.loc_simple_projection}, color={self.loc_simple_lineColor}")
            if any(loc == -1 for loc in [self.loc_simple_model,self.loc_simple_view,self.loc_simple_projection,self.loc_simple_lineColor]): print("Warning: Simple shader uniform missing.")
        except Exception as e: print(f"Error getting simple shader uniforms: {e}"); self.running = False; return
        self.camera = Camera( position=[15.0, 15.0, 12.0], yaw_deg=225.0, pitch_deg=-30.0, fov_deg=60.0, aspect_ratio=self.width / self.height, near=0.1, far=100.0, move_speed=10.0, rot_speed_dps=90.0 )
        self.grid_vao, self.grid_vbo, self.grid_element_count = None, None, 0
        self.axes_vao, self.axes_vbo, self.axes_element_count = None, None, 0
        self._create_initial_static_geometry()
        default_expression = "x**2 + y**2 + z**2 - 16"
        self.graph_function = GraphFunction( implicit_expression=default_expression, x_range=self.graph_x_range, y_range=self.graph_y_range, z_range=self.graph_z_range, resolution=self.current_resolution, color=self.graph_color )
        print("Initializing ImGui...")
        imgui.create_context()
        try: glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        except Exception as e: print(f" Warning: Failed to set GL_UNPACK_ALIGNMENT: {e}")
        try:
            self.imgui_renderer = PygameRenderer(); self.io = imgui.get_io(); self.io.display_size = (self.width, self.height)
            print("ImGui Initialized Successfully.")
        except Exception as e: print("!!! FAILED TO INITIALIZE PygameRenderer !!!"); traceback.print_exc(); self.running = False; return
        self.imgui_function_input_text = self.graph_function.expression
        self.mouse_grabbed = True; self.mouse_sensitivity = 0.0005
        try: pygame.mouse.set_visible(False); pygame.event.set_grab(True); print("Mouse hidden and grabbed for camera control.")
        except pygame.error as e: print(f"Warning: Could not initially hide/grab mouse: {e}")
        self._trigger_calculation()
        self.running = True

    # --- Static Geo Creation ---
    def _create_initial_static_geometry(self):
        print("Creating initial static geometry...")
        try:
            grid_size_coord = self.graph_x_range[1]; num_grid_lines=20; grid_spacing=(grid_size_coord*2)/num_grid_lines
            grid_line_endpoints_list = []
            for i in range(-num_grid_lines//2, num_grid_lines//2 + 1):
                 coord = i * grid_spacing
                 grid_line_endpoints_list.extend([[coord,-grid_size_coord,0.0],[coord,grid_size_coord,0.0]])
                 grid_line_endpoints_list.extend([[-grid_size_coord,coord,0.0],[grid_size_coord,coord,0.0]])
            grid_verts_np = np.array(grid_line_endpoints_list, dtype=np.float32).flatten()
            line_attribute_layout = [(0, 3, 0)]
            vao_grid, vbo_grid, _, count_grid, _ = create_opengl_mesh(grid_verts_np, line_attribute_layout, draw_mode=GL_LINES, usage=GL_STATIC_DRAW)
            if vao_grid: self.grid_vao, self.grid_vbo, self.grid_element_count = vao_grid, vbo_grid, count_grid
            axis_length = grid_size_coord
            axis_verts_list = [[-axis_length,0,0],[axis_length,0,0],[0,-axis_length,0],[0,axis_length,0],[0,0,-axis_length],[0,0,axis_length]]
            axis_verts_np = np.array(axis_verts_list, dtype=np.float32).flatten()
            vao_axes, vbo_axes, _, count_axes, _ = create_opengl_mesh(axis_verts_np, line_attribute_layout, draw_mode=GL_LINES, usage=GL_STATIC_DRAW)
            if vao_axes: self.axes_vao, self.axes_vbo, self.axes_element_count = vao_axes, vbo_axes, count_axes
            # print("Initial static geometry created.")
        except Exception as e: print(f"Error creating initial static OpenGL meshes: {e}")

    # --- Calculation Trigger ---
    def _trigger_calculation(self):
        current_params = { 'expression': self.graph_function.expression, 'resolution': self.current_resolution, 'x_range': self.graph_function.x_range, 'y_range': self.graph_function.y_range, 'z_range': self.graph_function.z_range }
        if self.calculation_active: print("Calculation already in progress. Queuing update."); self.pending_params = current_params; return
        print(f"Starting background mesh calculation for res={current_params['resolution']}...")
        self.calculation_active = True; self.pending_params = None; self.calculation_thread = None
        self.calculation_thread = threading.Thread( target=self._worker_generate_mesh, args=(current_params,), daemon=True )
        self.calculation_thread.start()

    # --- Worker Thread Target ---
    def _worker_generate_mesh(self, params):
        # print("BG THREAD: Worker started.")
        mesh_data = self.graph_function._calculate_mesh_data(**params) # Use ** to pass dict as keywords
        try:
            while not self.result_queue.empty():
                try: self.result_queue.get_nowait()
                except queue.Empty: break
            self.result_queue.put(mesh_data)
            # print("BG THREAD: Result placed in queue.")
        except Exception as e: print(f"BG THREAD ERROR: Failed to put result in queue: {e}"); self.result_queue.put(None)

    # --- UI Update Handler ---
    def _update_graph_from_input(self):
        user_expression_raw = self.imgui_function_input_text
        if not user_expression_raw.strip(): print("Input field is empty."); return
        # print(f"DEBUG: Raw input received: '{user_expression_raw}'")
        processed_expression = preprocess_expression(user_expression_raw)
        if not processed_expression: print("Preprocessing failed."); return
        # print(f"DEBUG: Processed expression: '{processed_expression}'")
        current_graph_expr = self.graph_function.expression
        # print(f"DEBUG: Current graph expr: '{current_graph_expr}'")
        if processed_expression != current_graph_expr:
            # print(f"DEBUG: Expression Comparison = DIFFERENT")
            changed = self.graph_function.set_implicit_expression(processed_expression)
            if changed:
                # print(f"DEBUG: graph_function.expression updated. Triggering calculation...")
                self._trigger_calculation()
            # else: print(f"DEBUG: ERROR? Comparison showed different, but set_implicit_expression returned False.")
        # else: print(f"DEBUG: Expression Comparison = SAME. Graph not updated.")
        # print(f"--- Update Graph Finished ---")

    # --- Event Loop ---
    def _handle_events(self):
        io = imgui.get_io() if getattr(self, 'imgui_renderer', None) else None
        for event in pygame.event.get():
            if io and hasattr(self, 'imgui_renderer'):
                try: self.imgui_renderer.process_event(event)
                except Exception as e: print(f"Error processing ImGui event: {e}")
            if event.type == pygame.QUIT: self.running = False; break
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h; self.height = max(1, self.height)
                # print(f"Window resized to: {self.width}x{self.height}")
                if io: io.display_size = (self.width, self.height)
                self.camera.update_aspect_ratio(self.width, self.height)
                try: glViewport(0, 0, self.width, self.height)
                except Exception as e: print(f"Error setting viewport after resize: {e}")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and hasattr(self, 'mouse_grabbed') and not self.mouse_grabbed and (not io or not io.want_capture_mouse):
                    self.mouse_grabbed = True
                    try: pygame.mouse.set_visible(False); pygame.event.set_grab(True); pygame.mouse.get_rel()
                    # print("Mouse Grabbed on Click (Camera Control Active)")
                    except pygame.error as e: print(f"Error setting mouse grab/visible on click: {e}")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if hasattr(self, 'mouse_grabbed'):
                        self.mouse_grabbed = not self.mouse_grabbed
                        try: pygame.mouse.set_visible(not self.mouse_grabbed); pygame.event.set_grab(self.mouse_grabbed)
                        except pygame.error as e: print(f"Error setting mouse grab/visible: {e}")
                        # if self.mouse_grabbed: print("Mouse Grabbed (Camera Control Active)")
                        # else: print("Mouse Released (UI Interaction Enabled)")
                elif io and not io.want_capture_keyboard:
                    if event.key == pygame.K_q: print("Q key pressed, quitting."); self.running = False; break

    # --- Input Processing ---
    def _process_input(self, dt):
        try:
            io = imgui.get_io() if hasattr(self, 'imgui_renderer') and self.imgui_renderer else None
            if io is None: return
            dyaw, dpitch = 0.0, 0.0; mouse_is_grabbed = getattr(self, 'mouse_grabbed', False)
            if mouse_is_grabbed and not io.want_capture_mouse:
                try:
                    dx, dy = pygame.mouse.get_rel()
                    if dx != 0: dyaw = -dx * self.mouse_sensitivity
                    if dy != 0: dpitch = -dy * self.mouse_sensitivity
                except pygame.error: pass
            if dyaw != 0. or dpitch != 0.:
                try: self.camera.rotate(dyaw, dpitch)
                except Exception as e: print(f"ERROR in camera.rotate: {e}"); traceback.print_exc(); self.running = False; return
            if (io and io.want_capture_keyboard) or not mouse_is_grabbed: return
            try:
                keys = pygame.key.get_pressed();
                if keys is None: return
                if not all(hasattr(self.camera, attr) and getattr(self.camera, attr) is not None for attr in ['forward', 'right', 'world_up']): return
                fwd=self.camera.forward; fwd_h=np.array([fwd[0],fwd[1],0.0],dtype=np.float32); fwd_h_norm=np.linalg.norm(fwd_h)
                if fwd_h_norm > 1e-8: fwd_h = fwd_h / fwd_h_norm; 
                else: fwd_h = np.array([0.,0.,0.], dtype=np.float32)
                right_h=self.camera.right; move_dir=np.array([0.,0.,0.],dtype=np.float32)
                if keys[pygame.K_w]: move_dir += fwd_h
                if keys[pygame.K_s]: move_dir -= fwd_h
                if keys[pygame.K_a]: move_dir -= right_h
                if keys[pygame.K_d]: move_dir += right_h
                if keys[pygame.K_SPACE]: move_dir += self.camera.world_up
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: move_dir -= self.camera.world_up
                norm_move = np.linalg.norm(move_dir)
                if norm_move > 1e-8: direction_vector = move_dir / norm_move; self.camera.move(direction_vector, dt)
            except Exception as e: print(f"ERROR in _process_input (movement): {e}"); traceback.print_exc(); self.running = False; return
        except Exception as e: print(f"!!! UNCAUGHT ERROR in _process_input: {e} !!!"); traceback.print_exc(); self.running = False

    # --- Rendering ---
    def _render(self):
        try:
            if not self.result_queue.empty():
                # print("Main Thread: Found result in queue.")
                try:
                    new_mesh_data = self.result_queue.get_nowait()
                    self.calculation_active = False
                    # start_gl_update = pygame.time.get_ticks()
                    if hasattr(self.graph_function, '_cleanup_gl_resources'): self.graph_function._cleanup_gl_resources()
                    if new_mesh_data is not None and new_mesh_data.size > 0:
                        # print("Main Thread: Updating OpenGL vertex buffer...")
                        attribute_layout = [(0, 3, 0), (1, 3, 12), (2, 1, 24)]
                        vao, vbo, _, count, _ = create_opengl_mesh( new_mesh_data, attribute_layout, indices=None, usage=GL_DYNAMIC_DRAW, draw_mode=GL_TRIANGLES )
                        if vao: self.graph_function.gl_vao = vao; self.graph_function.gl_vbo = vbo; self.graph_function.gl_element_count = count
                            # gl_update_time = pygame.time.get_ticks() - start_gl_update; print(f"Main Thread: OpenGL resources updated in {gl_update_time:.0f} ms.")
                        else: print("Main Thread ERROR: Failed to create new OpenGL mesh from thread result."); self.graph_function.gl_element_count = 0
                    else: print("Main Thread: Calculation yielded no data. Graph cleared."); self.graph_function.gl_element_count = 0
                    if self.pending_params is not None:
                        print("Main Thread: Processing pending parameter update...")
                        params_to_run = self.pending_params; self.pending_params = None
                        self._trigger_calculation() # Corrected: Trigger with current state, pending acts as flag
                except queue.Empty: pass
                except Exception as e: print(f"Error processing mesh result: {e}"); self.calculation_active = False

            glViewport(0, 0, self.width, self.height); glClearColor(*self.bg_color); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); glEnable(GL_DEPTH_TEST)
            proj_matrix = create_perspective_matrix_gl(np.radians(self.camera.fov_degrees), self.camera.aspect_ratio, self.camera.znear, self.camera.zfar)
            view_matrix = create_look_at_matrix_gl(self.camera.position, self.camera.position + self.camera.forward, self.camera.world_up)
            base_model_matrix = np.identity(4, dtype=np.float32); scale = self.visual_scale; scale_matrix = np.diag([scale, scale, scale, 1.0]).astype(np.float32); final_model_matrix = base_model_matrix @ scale_matrix

            if hasattr(self, 'simple_shader_program') and self.simple_shader_program:
                glUseProgram(self.simple_shader_program); glUniformMatrix4fv(self.loc_simple_projection, 1, GL_FALSE, proj_matrix); glUniformMatrix4fv(self.loc_simple_view, 1, GL_TRUE, view_matrix); glUniformMatrix4fv(self.loc_simple_model, 1, GL_TRUE, final_model_matrix)
                try: glLineWidth(1.0)
                except GLError: pass
                if hasattr(self, 'grid_vao') and self.grid_vao and self.grid_element_count > 0: glUniform3fv(self.loc_simple_lineColor, 1, self.grid_color); glBindVertexArray(self.grid_vao); glDrawArrays(GL_LINES, 0, self.grid_element_count)
                if hasattr(self, 'axes_vao') and self.axes_vao and self.axes_element_count > 0: glUniform3fv(self.loc_simple_lineColor, 1, self.axes_color); glBindVertexArray(self.axes_vao); glDrawArrays(GL_LINES, 0, self.axes_element_count)
                glLineWidth(1.0); glBindVertexArray(0)

            if (hasattr(self, 'shader_program') and self.shader_program and hasattr(self, 'graph_function') and self.graph_function and hasattr(self.graph_function, 'gl_vao') and self.graph_function.gl_vao and self.graph_function.gl_element_count > 0):
                glUseProgram(self.shader_program); glUniformMatrix4fv(self.loc_projection, 1, GL_FALSE, proj_matrix); glUniformMatrix4fv(self.loc_view, 1, GL_TRUE, view_matrix); glUniformMatrix4fv(self.loc_model, 1, GL_TRUE, final_model_matrix)
                graph_col_gl = [c / 255.0 for c in self.graph_function.color]; glUniform3fv(self.loc_objectColor, 1, graph_col_gl)
                if hasattr(self.graph_function, 'z_range') and len(self.graph_function.z_range) == 2: glUniform1f(self.loc_zMin, self.graph_function.z_range[0]); glUniform1f(self.loc_zMax, self.graph_function.z_range[1])
                else: glUniform1f(self.loc_zMin, -1.0); glUniform1f(self.loc_zMax, 1.0)
                glBindVertexArray(self.graph_function.gl_vao); glDrawArrays(GL_TRIANGLES, 0, self.graph_function.gl_element_count); glBindVertexArray(0)

            glUseProgram(0)

            if hasattr(self, 'imgui_renderer') and self.imgui_renderer:
                self.imgui_renderer.process_inputs(); imgui.new_frame()
                imgui.set_next_window_position(10, 10, imgui.ONCE)
                window_is_visible = imgui.begin("Controls", flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_ALWAYS_AUTO_RESIZE)
                ui_res_changed = False; temp_resolution = self.current_resolution; widget_width = 60
                if window_is_visible:
                    imgui.text("f(x,y,z) ="); imgui.same_line(); imgui.push_item_width(400)
                    input_changed, self.imgui_function_input_text = imgui.input_text("##func_input", self.imgui_function_input_text, 256)
                    imgui.pop_item_width(); enter_pressed = imgui.is_item_focused() and imgui.is_key_pressed(imgui.KEY_ENTER)
                    imgui.same_line(); button_pressed = imgui.button("Update Graph")
                    if button_pressed or enter_pressed: self._update_graph_from_input()
                    imgui.separator()
                    imgui.align_text_to_frame_padding(); imgui.text("Res:")
                    imgui.same_line();
                    if imgui.button("-##res_down"): ui_res_changed = True; temp_resolution = max(5, self.current_resolution - 5)
                    imgui.same_line(); imgui.push_item_width(widget_width); imgui.text(f"{self.current_resolution}"); imgui.pop_item_width()
                    imgui.same_line()
                    if imgui.button("+##res_up"): ui_res_changed = True; temp_resolution = min(250, self.current_resolution + 5)
                    if ui_res_changed:
                        # print(f"UI Res changed to: {temp_resolution}, triggering calculation...")
                        self.current_resolution = temp_resolution
                        self.graph_function.set_parameters(resolution=self.current_resolution)
                        self._trigger_calculation()
                    imgui.same_line(); imgui.text(" | "); imgui.same_line()
                    imgui.align_text_to_frame_padding(); imgui.text("Zoom:")
                    imgui.same_line()
                    if imgui.button("-##scale_down"): self.visual_scale = round(max(self.min_scale, self.visual_scale - self.scale_increment), 2); # print(f"Scale changed: {self.visual_scale:.2f}")
                    imgui.same_line(); imgui.push_item_width(widget_width); imgui.text(f"{self.visual_scale:.1f}"); imgui.pop_item_width()
                    imgui.same_line()
                    if imgui.button("+##scale_up"): self.visual_scale = round(self.visual_scale + self.scale_increment, 2); # print(f"Scale changed: {self.visual_scale:.2f}")
                imgui.end()
                if self.calculation_active:
                    imgui.set_next_window_bg_alpha(0.35); imgui.set_next_window_position(10, self.height - 30, condition=imgui.ALWAYS)
                    imgui.begin("StatusOverlay", flags=imgui.WINDOW_NO_DECORATION | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_FOCUS_ON_APPEARING | imgui.WINDOW_NO_NAV | imgui.WINDOW_NO_MOVE)
                    imgui.text("Calculating Mesh...")
                    imgui.end()
                glDisable(GL_DEPTH_TEST); imgui.render(); self.imgui_renderer.render(imgui.get_draw_data()); glEnable(GL_DEPTH_TEST)

            pygame.display.flip()
        except Exception as e: print(f"Error during render loop: {e}"); traceback.print_exc(); self.running = False

    # --- Cleanup ---
    def _cleanup(self):
        if self.calculation_thread is not None and self.calculation_thread.is_alive():
             print("Waiting for calculation thread to finish...")
             self.calculation_thread.join(timeout=0.5)
             if self.calculation_thread.is_alive(): print("Warning: Calculation thread did not finish quickly.")
        self.calculation_thread = None
        print("Cleaning up OpenGL resources...")
        try:
            if hasattr(self.graph_function, '_cleanup_gl_resources'): self.graph_function._cleanup_gl_resources()
            vaos_to_delete = []; buffers_to_delete = []
            if hasattr(self, 'grid_vao') and self.grid_vao: vaos_to_delete.append(self.grid_vao)
            if hasattr(self, 'grid_vbo') and self.grid_vbo: buffers_to_delete.append(self.grid_vbo)
            if hasattr(self, 'axes_vao') and self.axes_vao: vaos_to_delete.append(self.axes_vao)
            if hasattr(self, 'axes_vbo') and self.axes_vbo: buffers_to_delete.append(self.axes_vbo)
            if vaos_to_delete: glDeleteVertexArrays(len(vaos_to_delete), vaos_to_delete); # print(f"Deleted Static VAOs: {vaos_to_delete}")
            if buffers_to_delete: glDeleteBuffers(len(buffers_to_delete), buffers_to_delete); # print(f"Deleted Static Buffers: {buffers_to_delete}")
            if hasattr(self, 'shader_program') and self.shader_program: glDeleteProgram(self.shader_program); # print("Deleted main shader program.")
            if hasattr(self, 'simple_shader_program') and self.simple_shader_program: glDeleteProgram(self.simple_shader_program); # print("Deleted simple shader program.")
        except Exception as e: print(f"Error during OpenGL resource cleanup: {e}")
        print("Shutting down ImGui renderer...")
        try:
            if hasattr(self, 'imgui_renderer') and self.imgui_renderer: self.imgui_renderer.shutdown(); # print("ImGui renderer shut down.")
        except Exception as e: print(f"Error during ImGui cleanup: {e}")
        print("Cleaning up Pygame..."); pygame.quit(); print("Application closed.")

    # --- Run Loop ---
    def run(self):
        if not hasattr(self, 'running'): self.running = True
        while self.running:
            dt = self.clock.tick(self.fps) / 1000.0; dt = min(dt, 0.1)
            self._handle_events();
            if not self.running: break
            self._process_input(dt)
            self._render()
        self._cleanup()

# --- Main Execution ---
if __name__ == '__main__':
    try:
        app = App()
        if hasattr(app, 'running') and app.running: app.run()
    except ImportError as e:
        print(f"\nError: Library not found. {e}\n"
              f"Please ensure Pygame, PyOpenGL, numpy, asteval, and imgui[pygame] are installed.\n"
              f"Install command: pip install pygame-ce numpy asteval PyOpenGL PyOpenGL-accelerate \"imgui[pygame]\"")
    except Exception as e: print(f"\nAn unexpected error occurred: {e}"); traceback.print_exc()