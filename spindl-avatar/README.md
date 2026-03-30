# SpindL Avatar

Standalone desktop avatar renderer for [SpindL](https://github.com/your-repo/spindl). Renders VRM models with procedural animation, real-time cursor tracking, and post-processing — as a frameless, always-on-top overlay window.

Built with Tauri 2 + Three.js + @pixiv/three-vrm.

## Features

- **VRM rendering** with MToon material enhancement, 7-light rig, bloom, color grade, film grain
- **Mixamo animation playback** — FBX clips retargeted to any VRM skeleton with smooth crossfades
- **Procedural idle animation** — blinking, breathing, saccades, weight shifts, micro-fidgets, contrapposto
- **Cursor tracking** — eyes and head follow the mouse (spring-damped)
- **Gaze return** — eyes drift back to camera after 1.5s of cursor inactivity
- **Transparent window mode** — floating character on the desktop with full post-processing
- **WebSocket client** — receives state/mood/amplitude/speaking events from SpindL
- **Lipsync** — amplitude-driven viseme cycling with spring-smoothed blend shapes
- **Expression system** — 20 mood presets with staggered transitions and micro-overshoot
- **Camera controls** — scroll zoom, right-drag pan, Shift+scroll orbit (all spring-smoothed, persisted)
- **Background system** — gradient presets, custom image with blur, or fully transparent
- **VRM persistence** — loaded models saved to IndexedDB, restored on next launch
- **Window state persistence** — position, size, zoom, orbit angle saved across sessions
- **Context menu** — right-click for all configuration (mood preview, background, camera, avatar loading)

## Prerequisites

### All Platforms

- **Node.js** 20.x or later
- **Rust** stable toolchain (install via [rustup](https://rustup.rs/))
- **npm** (included with Node.js)

### Windows

- **Visual Studio Build Tools** with "Desktop development with C++" workload
- Or full Visual Studio 2022 with C++ components

### macOS

- **Xcode Command Line Tools**: `xcode-select --install`

### Linux

- **webkit2gtk 4.1** and development dependencies:

```bash
# Debian/Ubuntu
sudo apt install libwebkit2gtk-4.1-dev build-essential curl wget file \
  libssl-dev libayatana-appindicator3-dev librsvg2-dev

# Fedora
sudo dnf install webkit2gtk4.1-devel openssl-devel curl wget file \
  libappindicator-gtk3-devel librsvg2-devel

# Arch
sudo pacman -S webkit2gtk-4.1 base-devel curl wget file openssl \
  libappindicator-gtk3 librsvg
```

## Getting Started

```bash
# Clone and enter the project
cd spindl-avatar

# Install Node dependencies
npm install

# Run in development mode (opens the app with hot reload)
npm run tauri dev
```

On first launch, the bundled default avatar (AvatarSample_B) loads automatically. Right-click the window for all configuration options.

## Loading a Custom Avatar

Right-click → **Load Avatar...** → select a `.vrm` file. The model is saved to IndexedDB and restored automatically on next launch. Right-click → **Reset Avatar** to return to the default.

VRM models can be created with [VRoid Studio](https://vroid.com/en/studio) (free) or downloaded from [VRoid Hub](https://hub.vroid.com/).

## Animations

SpindL Avatar can play Mixamo FBX animations retargeted onto any VRM skeleton. When a clip plays, procedural body idle (contrapposto, fidgets, arm noise) is gated so the clip drives the skeleton, while face animation (blink, saccade, expressions, lipsync) continues layering on top.

### Downloading from Mixamo

1. Go to [mixamo.com](https://www.mixamo.com/) and sign in (free Adobe account)
2. Search for animations. Recommended starting set:
   - **Standing Idle** — default rest animation
   - **Happy Idle** — upbeat, animated idle
   - **Angry** — tense, aggressive stance
   - **Surprised** — startled reaction
   - **Sad Idle** — melancholy, low-energy
3. For each animation, click **Download** with these settings:
   - **Format:** FBX Binary (.fbx)
   - **Skin:** Without Skin
   - **Frames per Second:** 30
   - **Keyframe Reduction:** none
4. Place the downloaded `.fbx` files in `public/animations/`
5. Right-click the avatar → **Animation** → select a clip

Animations are selected per-session via the context menu and persisted in localStorage. The filename (without `.fbx`) is used as the display name.

> **Note:** Mixamo FBX files cannot be redistributed. They are excluded from git via `.gitignore`.

## Controls

| Input | Action |
|-------|--------|
| Left-click + drag | Move window |
| Right-click | Context menu |
| Right-click + drag | Pan camera vertically |
| Scroll | Zoom in/out |
| Shift + scroll | Orbit camera horizontally |

## Production Build

```bash
# TypeScript + Vite frontend build
npm run build

# Rust release build (produces native binary)
cd src-tauri && cargo build --release
```

The release binary is at `src-tauri/target/release/spindl-avatar` (or `.exe` on Windows).

## WebSocket Protocol

The avatar connects to `ws://127.0.0.1:8765/ws/avatar` by default (configurable via localStorage key `spindl-avatar-ws-url`). When no connection is available, the avatar idles autonomously.

**Inbound messages (JSON):**

| Message | Purpose |
|---------|---------|
| `{"state": "idle" \| "thinking" \| "speaking"}` | Mode → posture shifts |
| `{"speaking": true \| false}` | Speech start/stop → lipsync |
| `{"amplitude": 0.0-1.0}` | Lipsync amplitude (~30ms interval) |
| `{"mood": "success" \| "error" \| ...}` | Expression + lighting changes |
| `{"tool_mood": "search" \| "execute" \| ...}` | Brief tool-use visual flash |

## Project Structure

```
spindl-avatar/
├── src/                    # TypeScript frontend
│   ├── main.ts             # Entry point, state, controls, WebSocket
│   ├── scene.ts            # Three.js scene, lighting, post-processing
│   ├── avatar.ts           # VRM loading, MToon enhancement
│   ├── mixer.ts            # AnimationMixer — FBX clip load/play/crossfade
│   ├── retarget.ts         # Mixamo → VRM bone retargeting
│   ├── idle.ts             # Procedural idle (blink, breathe, saccade, fidget)
│   ├── body.ts             # Body posture (mode/mood), beat-bobs
│   ├── expressions.ts      # Mood → blend shape presets
│   ├── lipsync.ts          # Amplitude-driven viseme cycling
│   ├── wind.ts             # Spring bone wind system
│   ├── spring.ts           # Critically/underdamped spring physics
│   ├── noise.ts            # 1D Perlin noise + FBM
│   ├── state.ts            # Reactive state container
│   ├── ws-client.ts        # WebSocket client with reconnect
│   └── background.ts       # Gradient/image background system
├── src-tauri/              # Rust backend
│   └── src/lib.rs          # Window management, cursor tracking, context menu
├── public/
│   ├── animations/         # Mixamo FBX clips (user-supplied, gitignored)
│   └── models/avatar.vrm   # Bundled default avatar (AvatarSample_B)
├── THIRD_PARTY_NOTICES.md  # Avatar model license
├── index.html
├── package.json
└── tsconfig.json
```

## License

SpindL Avatar is part of the SpindL project. See the root repository for license details.

The bundled default avatar (AvatarSample_B) is provided by pixiv Inc. under VRoid Hub conditions of use. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
