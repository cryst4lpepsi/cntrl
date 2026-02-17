from __future__ import annotations

import time
import ctypes
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
import glfw
import moderngl


@dataclass
class TrackingData:
    fingertips: List[Tuple[float, float]]  # normalized (0..1)
    pinch: float  # 0..1


class GPUVisualizer:
    """
    Separate window GPU visualizer.
    - Background: camera texture
    - Foreground: neon dust particles (GPU)
    - Attractors: fingertips
    - Burst: pinch trigger
    """

    def __init__(self, width=1280, height=720, particle_count=40000):
        self.width = int(width)
        self.height = int(height)
        self.particle_count = int(particle_count)

        self.running = False
        self.window = None
        self.ctx: Optional[moderngl.Context] = None

        self.frame_bgr: Optional[np.ndarray] = None
        self.frame_tex = None

        self.tracking = TrackingData([], 0.0)
        self._pinch_prev = 0.0

        # GPU buffers
        self.p_pos = None
        self.p_vel = None

        self.prog_bg = None
        self.prog_particles = None
        self.vao_bg = None
        self.vao_particles = None

        self.start_time = time.time()

    # --------------------------
    # Public API
    # --------------------------

    def update_frame(self, bgr_frame: np.ndarray):
        self.frame_bgr = bgr_frame

    def update_tracking(self, fingertips: List[Tuple[float, float]], pinch: float):
        self.tracking = TrackingData(fingertips=fingertips, pinch=float(pinch))

    def start(self):
        self.running = True
        self._run()

    def stop(self):
        self.running = False

    # --------------------------
    # Internal
    # --------------------------

    def _init_window(self):
        if not glfw.init():
            raise RuntimeError("Failed to init GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(self.width, self.height, "GPU Visualizer (Particle Cloud)", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # vsync

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def _init_shaders(self):
        # Fullscreen quad shader (background camera)
        self.prog_bg = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos;
            in vec2 in_uv;
            out vec2 v_uv;
            void main() {
                v_uv = in_uv;
                gl_Position = vec4(in_pos, 0.0, 1.0);
            }
            """,
            fragment_shader="""
            #version 330
            uniform sampler2D u_tex;
            in vec2 v_uv;
            out vec4 f_color;
            void main() {
                vec3 col = texture(u_tex, v_uv).rgb;

                // slight cinematic grade
                col = pow(col, vec3(1.0/1.15));
                col *= vec3(1.05, 1.02, 1.08);

                f_color = vec4(col, 1.0);
            }
            """,
        )

        # Particle shader
        self.prog_particles = self.ctx.program(
            vertex_shader="""
            #version 330

            in vec2 in_pos;
            in vec2 in_vel;

            uniform float u_time;
            uniform vec2 u_attractors[10];
            uniform int u_attractor_count;
            uniform float u_pinch;
            uniform float u_burst;

            out float v_speed;
            out float v_seed;

            float hash(float n) { return fract(sin(n) * 43758.5453); }

            void main() {
                vec2 pos = in_pos;
                vec2 vel = in_vel;

                // find nearest attractor
                if (u_attractor_count > 0) {
                    float best = 999.0;
                    vec2 bestA = vec2(0.0);

                    for (int i = 0; i < u_attractor_count; i++) {
                        vec2 a = u_attractors[i];
                        float d = distance(pos, a);
                        if (d < best) {
                            best = d;
                            bestA = a;
                        }
                    }

                    vec2 dir = bestA - pos;
                    float dist = length(dir) + 0.0001;
                    vec2 dn = dir / dist;

                    // attraction
                    vel += dn * 0.0018;

                    // swirl
                    vec2 swirl = vec2(-dn.y, dn.x);
                    vel += swirl * 0.0012;
                }

                // pinch burst
                if (u_burst > 0.5) {
                    float s = hash(pos.x * 1000.0 + pos.y * 2000.0 + u_time);
                    vec2 burst_dir = normalize(vec2(cos(s*6.283), sin(s*6.283)));
                    vel += burst_dir * 0.03;
                }

                // damping
                vel *= 0.985;

                // update
                pos += vel;

                // wrap
                pos = fract(pos);

                // convert to clip space
                vec2 clip = pos * 2.0 - 1.0;
                clip.y *= -1.0; // flip y

                gl_Position = vec4(clip, 0.0, 1.0);

                // point size based on speed
                float sp = length(vel);
                v_speed = sp;
                v_seed = hash(pos.x * 999.0 + pos.y * 777.0);

                gl_PointSize = 1.0 + clamp(sp * 800.0, 0.0, 6.0);
            }
            """,
            fragment_shader="""
            #version 330

            in float v_speed;
            in float v_seed;

            uniform float u_time;

            out vec4 f_color;

            void main() {
                // circular soft particle
                vec2 p = gl_PointCoord * 2.0 - 1.0;
                float d = dot(p, p);

                float alpha = exp(-d * 2.5);

                // neon gradient cyan -> magenta
                float t = fract(u_time * 0.08 + v_seed);
                vec3 c1 = vec3(0.0, 1.0, 1.0);
                vec3 c2 = vec3(1.0, 0.0, 1.0);
                vec3 col = mix(c1, c2, t);

                // glow boost with speed
                float glow = clamp(v_speed * 1200.0, 0.0, 1.0);

                col *= (0.6 + glow * 1.2);
                alpha *= 0.15 + glow * 0.45;

                f_color = vec4(col, alpha);
            }
            """,
        )

        # background quad geometry
        quad = np.array([
            # x, y, u, v
            -1.0, -1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 0.0,
             1.0,  1.0, 1.0, 0.0,
        ], dtype=np.float32)

        vbo = self.ctx.buffer(quad.tobytes())
        self.vao_bg = self.ctx.vertex_array(
            self.prog_bg,
            [(vbo, "2f 2f", "in_pos", "in_uv")]
        )

        # particles buffer
        pos = np.random.rand(self.particle_count, 2).astype(np.float32)
        vel = (np.random.randn(self.particle_count, 2) * 0.002).astype(np.float32)

        self.p_pos = self.ctx.buffer(pos.tobytes())
        self.p_vel = self.ctx.buffer(vel.tobytes())

        self.vao_particles = self.ctx.vertex_array(
            self.prog_particles,
            [
                (self.p_pos, "2f", "in_pos"),
                (self.p_vel, "2f", "in_vel"),
            ],
        )

    def _init_texture(self):
        # init with blank
        blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.frame_tex = self.ctx.texture((self.width, self.height), 3, data=blank.tobytes())
        self.frame_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    def _upload_camera_frame(self):
        if self.frame_bgr is None:
            return

        frame = self.frame_bgr

        # resize to visualizer size
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # flip vertically because OpenGL UV
        rgb = np.flipud(rgb)

        self.frame_tex.write(rgb.tobytes())

    def _run(self):
        self._init_window()
        self._init_shaders()
        self._init_texture()

        self.running = True
        last = time.time()

        while self.running and not glfw.window_should_close(self.window):
            now = time.time()
            dt = now - last
            last = now

            glfw.poll_events()

            # upload latest camera frame
            self._upload_camera_frame()

            # build attractor array
            tips = self.tracking.fingertips[:10]
            attractor_count = len(tips)
            attractors = [(0.5, 0.5)] * 10
            for i in range(attractor_count):
                attractors[i] = tips[i]

            # pinch burst trigger
            pinch = float(self.tracking.pinch)
            burst = 1.0 if (pinch > 0.78 and self._pinch_prev <= 0.78) else 0.0
            self._pinch_prev = pinch

            # render
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)

            # background
            self.frame_tex.use(0)
            self.prog_bg["u_tex"].value = 0
            self.vao_bg.render(moderngl.TRIANGLE_STRIP)

            # particles uniforms
            t = now - self.start_time
            self.prog_particles["u_time"].value = t
            self.prog_particles["u_attractor_count"].value = attractor_count
            self.prog_particles["u_pinch"].value = pinch
            self.prog_particles["u_burst"].value = burst

            # pass attractors
            # moderngl wants a flat tuple
            flat = []
            for i in range(10):
                flat.extend([float(attractors[i][0]), float(attractors[i][1])])
            self.prog_particles["u_attractors"].value = tuple(flat)

            # draw
            self.vao_particles.render(mode=moderngl.POINTS)

            glfw.swap_buffers(self.window)

        glfw.terminate()
