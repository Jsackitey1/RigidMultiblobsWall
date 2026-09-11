'''
Standalone Interactive 2D HTML5 Canvas Simulation Player Generator.
Zero-dependency, high-performance 2D pan/zoom interactive application.
'''

import os
import json
import math
import numpy as np


def export_interactive_html(traj, output_path, target_duration=10.0):
    '''
    Export self-contained 2D HTML5 Canvas interactive simulation player.
    target_duration: Target playback length in seconds (default: 10.0s).
    '''
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Compute global z range for dynamic colour normalization (Bug 4)
    all_z = traj.positions[..., 2]
    z_min = float(min(0.0, np.min(all_z)))
    z_max = float(max(np.max(all_z) + 0.5, traj.sphere_radius * 3.0))

    # Generate true viridis 256-entry LUT from matplotlib (Bug 5)
    try:
        import matplotlib.cm as cm
        viridis_lut = []
        for i in range(256):
            r, g, b, _ = cm.viridis(i / 255.0)
            viridis_lut.append([int(round(r * 255)), int(round(g * 255)), int(round(b * 255))])
    except ImportError:
        # Hardcoded fallback — first/last entries from true viridis
        viridis_lut = [[68, 1, 84]] * 128 + [[253, 231, 37]] * 128

    frames_data = []
    for f in range(traj.num_frames):
        pos_f = traj.positions[f].round(4).tolist()
        quat_f = traj.quaternions[f].round(4).tolist()
        speeds_f = traj.speeds[f].round(4).tolist()
        angles_f = traj.get_inplane_orientation_angles(f).round(4).tolist()
        # Minor: removed dead velocities payload — never read by JS

        frames_data.append({
            'time': round(float(traj.time_array[f]), 4),
            'mean_z': round(float(traj.mean_z[f]), 4),
            'positions': pos_f,
            'quaternions': quat_f,
            'speeds': speeds_f,
            'angles': angles_f
        })

    vertex_blobs_list = traj.vertex_blobs.round(4).tolist() if traj.vertex_blobs is not None else []

    sim_payload = {
        'num_frames': traj.num_frames,
        'num_bodies': traj.num_bodies,
        'sphere_radius': float(traj.sphere_radius),
        'blob_radius': float(traj.blob_radius),
        'domain_bounds': [float(x) for x in traj.domain_bounds],
        'vertex_blobs': vertex_blobs_list,
        'target_duration': float(target_duration),
        'z_min': z_min,
        'z_max': z_max,
        'viridis_lut': viridis_lut,
        'frames': frames_data
    }

    json_data = json.dumps(sim_payload)

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RigidMultiblobsWall - Interactive 2D Simulation Player</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            user-select: none;
        }}
        body {{
            background-color: #0b0f19;
            color: #f1f5f9;
            overflow: hidden;
            width: 100vw;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        #canvas-wrapper {{
            flex: 1;
            position: relative;
            width: 100%;
            height: 100%;
            cursor: grab;
        }}
        #canvas-wrapper:active {{
            cursor: grabbing;
        }}
        canvas {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }}
        .hud-overlay {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(15, 23, 42, 0.9);
            backdrop-filter: blur(10px);
            padding: 16px 20px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            pointer-events: none;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            min-width: 280px;
            z-index: 10;
        }}
        .hud-title {{
            font-size: 15px;
            font-weight: 700;
            color: #38bdf8;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .hud-stat {{
            font-size: 13px;
            color: #94a3b8;
            margin-top: 4px;
        }}
        .hud-stat span {{
            color: #f8fafc;
            font-weight: 600;
        }}
        .controls-panel {{
            position: absolute;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(15, 23, 42, 0.92);
            backdrop-filter: blur(12px);
            padding: 12px 24px;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            display: flex;
            align-items: center;
            gap: 16px;
            box-shadow: 0 12px 32px rgba(0,0,0,0.65);
            z-index: 20;
        }}
        button {{
            background: #0284c7;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 13px;
        }}
        button:hover {{
            background: #0369a1;
            transform: scale(1.02);
        }}
        .btn-secondary {{
            background: #334155;
        }}
        .btn-secondary:hover {{
            background: #475569;
        }}
        .control-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .control-group label {{
            font-size: 12px;
            color: #94a3b8;
            font-weight: 500;
        }}
        input[type="range"] {{
            width: 220px;
            cursor: pointer;
            accent-color: #38bdf8;
        }}
        select {{
            background: #1e293b;
            color: #f8fafc;
            border: 1px solid #475569;
            padding: 6px 10px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
        }}
        .view-badge {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 12px;
            color: #94a3b8;
            z-index: 10;
        }}
    </style>
</head>
<body>
    <div id="canvas-wrapper">
        <canvas id="sim-canvas"></canvas>
        <div class="hud-overlay">
            <div class="hud-title">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
                RigidMultiblobsWall 2D Player
            </div>
            <div class="hud-stat">Physical Time: <span id="stat-time">0.000</span> s</div>
            <div class="hud-stat">Frame: <span id="stat-frame">1</span> / <span id="stat-total-frames">{traj.num_frames}</span></div>
            <div class="hud-stat">Duration: <span id="stat-duration">{target_duration:.1f}</span> s</div>
            <div class="hud-stat">Active Bodies: <span>{traj.num_bodies}</span></div>
            <div class="hud-stat">Mean Height ⟨z⟩: <span id="stat-mean-z">0.000</span></div>
        </div>

        <div class="view-badge">
            Scroll to Zoom | Drag to Pan
        </div>

        <div class="controls-panel">
            <button id="btn-play">Pause</button>
            <div class="control-group">
                <label>Timeline:</label>
                <input type="range" id="slider-timeline" min="0" max="{traj.num_frames - 1}" value="0">
            </div>
            <div class="control-group">
                <label>View:</label>
                <select id="select-view">
                    <option value="top" selected>Top-Down (X-Y)</option>
                    <option value="side">Side-Elevation (X-Z)</option>
                    <option value="dual">Dual (X-Y + X-Z)</option>
                </select>
            </div>
            <div class="control-group">
                <label>Mode:</label>
                <select id="select-mode">
                    <option value="spheres" selected>Discs + Rotation</option>
                    <option value="blobs">Multi-Blobs</option>
                </select>
            </div>
            <div class="control-group">
                <label>Speed:</label>
                <select id="select-speed">
                    <option value="0.1">0.1x (Slowest)</option>
                    <option value="0.25">0.25x (Slow)</option>
                    <option value="0.5">0.5x (Half)</option>
                    <option value="1" selected>1.0x (Normal)</option>
                    <option value="2">2.0x (Fast)</option>
                    <option value="4">4.0x (Fastest)</option>
                </select>
            </div>
            <button id="btn-reset-view" class="btn-secondary">Reset View</button>
        </div>

        <!-- Bug 5: Colour-bar legend for z-height -->
        <div id="legend" style="position:absolute;bottom:90px;right:20px;background:rgba(15,23,42,0.9);border:1px solid rgba(255,255,255,0.12);border-radius:10px;padding:10px 14px;z-index:10;">
            <div style="font-size:11px;color:#94a3b8;margin-bottom:6px;font-weight:600;">Height z</div>
            <div style="display:flex;align-items:center;gap:8px;">
                <span id="legend-min" style="font-size:10px;color:#f8fafc;">{z_min:.2f}</span>
                <canvas id="legend-bar" width="120" height="14" style="border-radius:3px;"></canvas>
                <span id="legend-max" style="font-size:10px;color:#f8fafc;">{z_max:.2f}</span>
            </div>
        </div>
    </div>

    <script>
        const simData = {json_data};
        const numFrames = simData.num_frames;
        const numBodies = simData.num_bodies;
        const R = simData.sphere_radius;
        const blobRadius = simData.blob_radius;
        const vertexBlobs = simData.vertex_blobs;
        const [xmin, xmax, ymin, ymax, zmin, zmax] = simData.domain_bounds;
        const zMin = simData.z_min;
        const zMax = simData.z_max;
        const viridisLUT = simData.viridis_lut;

        let currentFrame = 0;
        let isPlaying = true;
        let playSpeed = 1.0;
        let viewMode = 'top';
        let renderMode = 'spheres';

        const canvas = document.getElementById('sim-canvas');
        const ctx = canvas.getContext('2d');
        const wrapper = document.getElementById('canvas-wrapper');

        let camera = {{
            x: (xmin + xmax) / 2,
            y: (ymin + ymax) / 2,
            scale: 1.0
        }};
        let isDragging = false;
        let dragStart = {{ x: 0, y: 0 }};
        let cameraStart = {{ x: 0, y: 0 }};

        function resize() {{
            canvas.width = wrapper.clientWidth * window.devicePixelRatio;
            canvas.height = wrapper.clientHeight * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
            fitCamera();
            draw();
        }}
        window.addEventListener('resize', resize);

        function fitCamera() {{
            const w = wrapper.clientWidth;
            const h = wrapper.clientHeight;
            const domainW = (xmax - xmin) + 4 * R;
            const domainH = (ymax - ymin) + 4 * R;
            camera.scale = Math.min(w / domainW, h / domainH) * 0.88;
            camera.x = (xmin + xmax) / 2;
            camera.y = (ymin + ymax) / 2;
        }}

        function worldToScreen(wx, wy, customW, customH, customCam) {{
            const cam = customCam || camera;
            const cw = customW || wrapper.clientWidth;
            const ch = customH || wrapper.clientHeight;
            const sx = cw / 2 + (wx - cam.x) * cam.scale;
            const sy = ch / 2 - (wy - cam.y) * cam.scale;
            return {{ x: sx, y: sy }};
        }}

        function getViridisColor(t) {{
            t = Math.max(0, Math.min(1, t));
            const idx = Math.min(255, Math.round(t * 255));
            const [r, g, b] = viridisLUT[idx];
            return `rgb(${{r}}, ${{g}}, ${{b}})`;
        }}

        function zNormalize(z) {{
            const range = zMax - zMin;
            if (range <= 0) return 0.5;
            return (z - zMin) / range;
        }}

        function drawTopDown(targetCtx, w, h, cam) {{
            const frame = simData.frames[currentFrame];
            const pos = frame.positions;
            const angles = frame.angles;

            // Draw domain boundary box
            const tl = worldToScreen(xmin, ymax, w, h, cam);
            const br = worldToScreen(xmax, ymin, w, h, cam);
            targetCtx.strokeStyle = '#334155';
            targetCtx.lineWidth = 1.5;
            targetCtx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);

            // Draw grid
            targetCtx.strokeStyle = '#1e293b';
            targetCtx.lineWidth = 0.5;
            const gridStep = Math.max(5, Math.round((xmax - xmin) / 8));
            for (let gx = Math.ceil(xmin / gridStep) * gridStep; gx <= xmax; gx += gridStep) {{
                const p1 = worldToScreen(gx, ymin, w, h, cam);
                const p2 = worldToScreen(gx, ymax, w, h, cam);
                targetCtx.beginPath();
                targetCtx.moveTo(p1.x, p1.y);
                targetCtx.lineTo(p2.x, p2.y);
                targetCtx.stroke();
            }}
            for (let gy = Math.ceil(ymin / gridStep) * gridStep; gy <= ymax; gy += gridStep) {{
                const p1 = worldToScreen(xmin, gy, w, h, cam);
                const p2 = worldToScreen(xmax, gy, w, h, cam);
                targetCtx.beginPath();
                targetCtx.moveTo(p1.x, p1.y);
                targetCtx.lineTo(p2.x, p2.y);
                targetCtx.stroke();
            }}

            // Draw bodies / blobs
            if (renderMode === 'spheres') {{
                // Bug 6: Sort bodies by ascending z for correct depth ordering
                const indices = Array.from({{length: numBodies}}, (_, i) => i);
                indices.sort((a, b) => pos[a][2] - pos[b][2]);

                for (const i of indices) {{
                    const [bx, by, bz] = pos[i];
                    const sp = worldToScreen(bx, by, w, h, cam);
                    const sRadius = R * cam.scale;

                    // Body Disc
                    targetCtx.beginPath();
                    targetCtx.arc(sp.x, sp.y, sRadius, 0, Math.PI * 2);
                    targetCtx.fillStyle = getViridisColor(zNormalize(bz));
                    targetCtx.fill();
                    targetCtx.strokeStyle = '#ffffff';
                    targetCtx.lineWidth = 1.0;
                    targetCtx.stroke();

                    // Orientation Radius Pointer
                    const th = angles[i];
                    const endX = sp.x + sRadius * Math.cos(th);
                    const endY = sp.y - sRadius * Math.sin(th);
                    targetCtx.beginPath();
                    targetCtx.moveTo(sp.x, sp.y);
                    targetCtx.lineTo(endX, endY);
                    targetCtx.strokeStyle = '#facc15';
                    targetCtx.lineWidth = 2.0;
                    targetCtx.stroke();

                    // Center dot
                    targetCtx.beginPath();
                    targetCtx.arc(sp.x, sp.y, 2, 0, Math.PI * 2);
                    targetCtx.fillStyle = '#ffffff';
                    targetCtx.fill();
                }}
            }} else if (renderMode === 'blobs') {{
                const bRadius = blobRadius * cam.scale;
                const quats = frame.quaternions;

                // Bug 6: Sort bodies by ascending z for correct depth ordering
                const indices = Array.from({{length: numBodies}}, (_, i) => i);
                indices.sort((a, b) => pos[a][2] - pos[b][2]);

                for (const i of indices) {{
                    const [bx, by, bz] = pos[i];
                    const q = quats[i];
                    const q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
                    const diag = q0 * q0 - 0.5;

                    // Rotation matrix SO(3)
                    const r00 = 2 * (q1 * q1 + diag);
                    const r01 = 2 * (q1 * q2 - q0 * q3);
                    const r02 = 2 * (q1 * q3 + q0 * q2);
                    const r10 = 2 * (q2 * q1 + q0 * q3);
                    const r11 = 2 * (q2 * q2 + diag);
                    const r12 = 2 * (q2 * q3 - q0 * q1);
                    const r20 = 2 * (q3 * q1 - q0 * q2);
                    const r21 = 2 * (q3 * q2 + q0 * q1);
                    const r22 = 2 * (q3 * q3 + diag);

                    if (vertexBlobs && vertexBlobs.length > 0) {{
                        for (let k = 0; k < vertexBlobs.length; k++) {{
                            const [lx, ly, lz] = vertexBlobs[k];
                            const gx = bx + (r00 * lx + r01 * ly + r02 * lz);
                            const gy = by + (r10 * lx + r11 * ly + r12 * lz);
                            const gz = bz + (r20 * lx + r21 * ly + r22 * lz);

                            const bp = worldToScreen(gx, gy, w, h, cam);
                            targetCtx.beginPath();
                            targetCtx.arc(bp.x, bp.y, Math.max(1.5, bRadius), 0, Math.PI * 2);
                            targetCtx.fillStyle = getViridisColor(zNormalize(gz));
                            targetCtx.fill();
                            targetCtx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
                            targetCtx.lineWidth = 0.5;
                            targetCtx.stroke();
                        }}
                    }} else {{
                        const bp = worldToScreen(bx, by, w, h, cam);
                        targetCtx.beginPath();
                        targetCtx.arc(bp.x, bp.y, Math.max(2, bRadius), 0, Math.PI * 2);
                        targetCtx.fillStyle = getViridisColor(zNormalize(bz));
                        targetCtx.fill();
                    }}

                    // Body center marker
                    const cp = worldToScreen(bx, by, w, h, cam);
                    targetCtx.beginPath();
                    targetCtx.arc(cp.x, cp.y, 2, 0, Math.PI * 2);
                    targetCtx.fillStyle = '#38bdf8';
                    targetCtx.fill();
                }}
            }}
        }}

        function drawSideElevation(targetCtx, w, h, cam) {{
            const frame = simData.frames[currentFrame];
            const pos = frame.positions;

            // Floor Wall at z = 0
            const pWallL = worldToScreen(xmin - 4*R, 0, w, h, cam);
            const pWallR = worldToScreen(xmax + 4*R, 0, w, h, cam);
            targetCtx.strokeStyle = '#f43f5e';
            targetCtx.lineWidth = 3.0;
            targetCtx.beginPath();
            targetCtx.moveTo(pWallL.x, pWallL.y);
            targetCtx.lineTo(pWallR.x, pWallR.y);
            targetCtx.stroke();

            // Contact Clearance at z = R
            const pContL = worldToScreen(xmin - 4*R, R, w, h, cam);
            const pContR = worldToScreen(xmax + 4*R, R, w, h, cam);
            targetCtx.strokeStyle = '#fbbf24';
            targetCtx.lineWidth = 1.2;
            targetCtx.setLineDash([4, 4]);
            targetCtx.beginPath();
            targetCtx.moveTo(pContL.x, pContL.y);
            targetCtx.lineTo(pContR.x, pContR.y);
            targetCtx.stroke();
            targetCtx.setLineDash([]);

            // Draw sphere circles in X-Z
            // Bug 6: Sort by ascending y (depth into screen) for side view
            const indices = Array.from({{length: numBodies}}, (_, i) => i);
            indices.sort((a, b) => pos[a][1] - pos[b][1]);

            for (const i of indices) {{
                const [bx, by, bz] = pos[i];
                const sp = worldToScreen(bx, bz, w, h, cam);
                const sRadius = R * cam.scale;

                targetCtx.beginPath();
                targetCtx.arc(sp.x, sp.y, sRadius, 0, Math.PI * 2);
                targetCtx.fillStyle = getViridisColor(zNormalize(bz));
                targetCtx.fill();
                targetCtx.strokeStyle = '#ffffff';
                targetCtx.lineWidth = 0.8;
                targetCtx.stroke();

                targetCtx.beginPath();
                targetCtx.arc(sp.x, sp.y, 2, 0, Math.PI * 2);
                targetCtx.fillStyle = '#facc15';
                targetCtx.fill();
            }}

            // Mean height
            const pMeanL = worldToScreen(xmin - 4*R, frame.mean_z, w, h, cam);
            const pMeanR = worldToScreen(xmax + 4*R, frame.mean_z, w, h, cam);
            targetCtx.strokeStyle = '#34d399';
            targetCtx.lineWidth = 2.0;
            targetCtx.beginPath();
            targetCtx.moveTo(pMeanL.x, pMeanL.y);
            targetCtx.lineTo(pMeanR.x, pMeanR.y);
            targetCtx.stroke();
        }}

        function draw() {{
            const w = wrapper.clientWidth;
            const h = wrapper.clientHeight;
            ctx.clearRect(0, 0, w, h);

            if (viewMode === 'top') {{
                drawTopDown(ctx, w, h, camera);
            }} else if (viewMode === 'side') {{
                const sideCam = {{ x: camera.x, y: zmax / 2, scale: camera.scale }};
                drawSideElevation(ctx, w, h, sideCam);
            }} else if (viewMode === 'dual') {{
                const halfW = w / 2;
                ctx.save();
                ctx.beginPath();
                ctx.rect(0, 0, halfW, h);
                ctx.clip();
                drawTopDown(ctx, halfW, h, camera);
                ctx.restore();

                ctx.save();
                ctx.beginPath();
                ctx.rect(halfW, 0, halfW, h);
                ctx.clip();
                ctx.translate(halfW, 0);
                const sideCam = {{ x: camera.x, y: zmax / 2, scale: camera.scale }};
                drawSideElevation(ctx, halfW, h, sideCam);
                ctx.restore();

                // Separator
                ctx.strokeStyle = '#334155';
                ctx.lineWidth = 2.0;
                ctx.beginPath();
                ctx.moveTo(halfW, 0);
                ctx.lineTo(halfW, h);
                ctx.stroke();
            }}

            // Update stats
            const frame = simData.frames[currentFrame];
            document.getElementById('stat-time').innerText = frame.time.toFixed(3);
            document.getElementById('stat-frame').innerText = (currentFrame + 1);
            document.getElementById('stat-mean-z').innerText = frame.mean_z.toFixed(4);
            document.getElementById('slider-timeline').value = currentFrame;
        }}

        // --- Interaction Listeners ---
        wrapper.addEventListener('mousedown', (e) => {{
            isDragging = true;
            dragStart = {{ x: e.clientX, y: e.clientY }};
            cameraStart = {{ x: camera.x, y: camera.y }};
        }});

        window.addEventListener('mousemove', (e) => {{
            if (!isDragging) return;
            const dx = e.clientX - dragStart.x;
            const dy = e.clientY - dragStart.y;
            camera.x = cameraStart.x - dx / camera.scale;
            camera.y = cameraStart.y + dy / camera.scale;
            draw();
        }});

        window.addEventListener('mouseup', () => {{ isDragging = false; }});

        wrapper.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const factor = e.deltaY < 0 ? 1.15 : 0.87;
            // Minor: Cursor-anchored zoom
            const rect = wrapper.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            const w = wrapper.clientWidth;
            const h = wrapper.clientHeight;
            // World position under cursor before zoom
            const wx = camera.x + (mx - w / 2) / camera.scale;
            const wy = camera.y - (my - h / 2) / camera.scale;
            camera.scale *= factor;
            // Adjust camera so world point stays under cursor
            camera.x = wx - (mx - w / 2) / camera.scale;
            camera.y = wy + (my - h / 2) / camera.scale;
            draw();
        }}, {{ passive: false }});

        // --- UI Controls ---
        const btnPlay = document.getElementById('btn-play');
        btnPlay.addEventListener('click', () => {{
            isPlaying = !isPlaying;
            btnPlay.innerText = isPlaying ? 'Pause' : 'Play';
        }});

        const slider = document.getElementById('slider-timeline');
        slider.addEventListener('input', (e) => {{
            isPlaying = false;
            btnPlay.innerText = 'Play';
            currentFrame = parseInt(e.target.value);
            draw();
        }});

        document.getElementById('select-view').addEventListener('change', (e) => {{
            viewMode = e.target.value;
            fitCamera();
            draw();
        }});

        document.getElementById('select-mode').addEventListener('change', (e) => {{
            renderMode = e.target.value;
            draw();
        }});

        document.getElementById('select-speed').addEventListener('change', (e) => {{
            playSpeed = parseFloat(e.target.value);
        }});

        document.getElementById('btn-reset-view').addEventListener('click', () => {{
            fitCamera();
            draw();
        }});

        // --- Animation Loop ---
        let lastTimestamp = 0;
        let accumulator = 0;
        function animate(timestamp) {{
            requestAnimationFrame(animate);
            if (!lastTimestamp) lastTimestamp = timestamp;
            const dt = (timestamp - lastTimestamp) / 1000;
            lastTimestamp = timestamp;

            if (isPlaying && numFrames > 1) {{
                const targetDuration = simData.target_duration || 10.0;
                const playbackFPS = numFrames / Math.max(1.0, targetDuration);
                accumulator += dt * playbackFPS * playSpeed;
                if (accumulator >= 1.0) {{
                    const steps = Math.floor(accumulator);
                    accumulator -= steps;
                    currentFrame = (currentFrame + steps) % numFrames;
                    draw();
                }}
            }}
        }}

        resize();

        // Bug 5: Render colour-bar legend
        const legendCanvas = document.getElementById('legend-bar');
        const lctx = legendCanvas.getContext('2d');
        for (let px = 0; px < legendCanvas.width; px++) {{
            const t = px / (legendCanvas.width - 1);
            lctx.fillStyle = getViridisColor(t);
            lctx.fillRect(px, 0, 1, legendCanvas.height);
        }}

        requestAnimationFrame(animate);
    </script>
</body>
</html>
'''
    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"[+] Saved Interactive 2D HTML5 Player: {output_path}")
    return output_path
