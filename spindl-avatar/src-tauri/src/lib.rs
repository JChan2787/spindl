use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tauri::menu::{MenuBuilder, SubmenuBuilder};
use tauri::{AppHandle, Manager, WindowEvent};

// ── Window State Persistence ──────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Debug)]
struct WindowState {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

fn config_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".config").join("spindl-avatar").join("window.json"))
}

fn load_window_state() -> Option<WindowState> {
    let path = config_path()?;
    let data = fs::read_to_string(path).ok()?;
    serde_json::from_str(&data).ok()
}

fn save_window_state(state: &WindowState) {
    if let Some(path) = config_path() {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(path, serde_json::to_string_pretty(state).unwrap_or_default());
    }
}

fn capture_window_state(window: &tauri::Window) -> Option<WindowState> {
    let pos = window.outer_position().ok()?;
    let size = window.outer_size().ok()?;
    let scale = window.scale_factor().unwrap_or(1.0);
    Some(WindowState {
        x: pos.x as f64 / scale,
        y: pos.y as f64 / scale,
        width: size.width as f64 / scale,
        height: size.height as f64 / scale,
    })
}

fn capture_webview_window_state(window: &tauri::WebviewWindow) -> Option<WindowState> {
    let pos = window.outer_position().ok()?;
    let size = window.outer_size().ok()?;
    let scale = window.scale_factor().unwrap_or(1.0);
    Some(WindowState {
        x: pos.x as f64 / scale,
        y: pos.y as f64 / scale,
        width: size.width as f64 / scale,
        height: size.height as f64 / scale,
    })
}

// ── Tauri Commands ────────────────────────────────────────────────────

/// Read a file from disk and return its bytes. Used for VRM loading
/// since the dialog plugin returns a path but WebView can't fetch local files.
#[tauri::command]
fn read_file_bytes(path: String) -> Result<Vec<u8>, String> {
    fs::read(&path).map_err(|e| format!("Failed to read {}: {}", path, e))
}

/// Resolve the animations directory (handles CWD being project root or src-tauri/).
#[tauri::command]
fn get_animations_dir() -> String {
    let cwd = std::env::current_dir().unwrap_or_default();
    let from_root = cwd.join("public").join("animations");
    let from_src_tauri = cwd.join("..").join("public").join("animations");
    let dir = if from_root.exists() { from_root } else { from_src_tauri };
    // Canonicalize to absolute path, fall back to the raw path
    dir.canonicalize().unwrap_or(dir).to_string_lossy().to_string()
}

/// List filenames in a directory. Returns empty vec if path doesn't exist.
#[tauri::command]
fn list_directory(path: String) -> Result<Vec<String>, String> {
    let dir = PathBuf::from(&path);
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let entries = fs::read_dir(&dir).map_err(|e| format!("Failed to read dir {}: {}", path, e))?;
    let mut names = Vec::new();
    for entry in entries.flatten() {
        if let Some(name) = entry.file_name().to_str() {
            names.push(name.to_string());
        }
    }
    Ok(names)
}

/// Returns global cursor position relative to overlay window center, normalized to -1..1.
fn normalize_cursor(app: &AppHandle, cursor_x: f64, cursor_y: f64) -> Option<(f64, f64)> {
    let window = app.get_webview_window("overlay")?;
    let win_pos = window.outer_position().ok()?;
    let win_size = window.outer_size().ok()?;
    let scale = window.scale_factor().unwrap_or(1.0);

    let center_x = win_pos.x as f64 / scale + win_size.width as f64 / scale / 2.0;
    let center_y = win_pos.y as f64 / scale + win_size.height as f64 / scale / 2.0;

    let range = 500.0;
    let norm_x = ((cursor_x - center_x) / range).clamp(-1.0, 1.0);
    let norm_y = ((cursor_y - center_y) / range).clamp(-1.0, 1.0);
    Some((norm_x, norm_y))
}

#[tauri::command]
fn get_cursor_position(app: AppHandle) -> Option<(f64, f64)> {
    #[cfg(target_os = "macos")]
    {
        use core_graphics::event::CGEvent;
        use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};

        let source = CGEventSource::new(CGEventSourceStateID::CombinedSessionState).ok()?;
        let event = CGEvent::new(source).ok()?;
        let cursor = event.location();
        return normalize_cursor(&app, cursor.x, cursor.y);
    }

    #[cfg(target_os = "windows")]
    {
        use windows::Win32::UI::WindowsAndMessaging::GetCursorPos;
        use windows::Win32::Foundation::POINT;
        let mut point = POINT::default();
        unsafe { let _ = GetCursorPos(&mut point); }
        return normalize_cursor(&app, point.x as f64, point.y as f64);
    }

    #[cfg(target_os = "linux")]
    {
        use x11rb::connection::Connection;
        use x11rb::protocol::xproto::ConnectionExt;
        if let Ok((conn, screen_num)) = x11rb::connect(None) {
            let setup = conn.setup();
            let root = setup.roots[screen_num].root;
            if let Ok(reply) = conn.query_pointer(root) {
                if let Ok(pointer) = reply.reply() {
                    return normalize_cursor(&app, pointer.root_x as f64, pointer.root_y as f64);
                }
            }
        }
        return None;
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    { None }
}

/// Returns normalized direction from avatar window center to screen center (-1..1, -1..1).
#[tauri::command]
fn get_viewer_direction(app: AppHandle) -> Option<(f64, f64)> {
    let window = app.get_webview_window("overlay")?;
    let win_pos = window.outer_position().ok()?;
    let win_size = window.outer_size().ok()?;
    let scale = window.scale_factor().unwrap_or(1.0);

    let monitor = window.current_monitor().ok()??;
    let mon_size = monitor.size();
    let mon_pos = monitor.position();

    let win_cx = win_pos.x as f64 / scale + win_size.width as f64 / scale / 2.0;
    let win_cy = win_pos.y as f64 / scale + win_size.height as f64 / scale / 2.0;

    let screen_cx = mon_pos.x as f64 / scale + mon_size.width as f64 / scale / 2.0;
    let screen_cy = mon_pos.y as f64 / scale + mon_size.height as f64 / scale / 2.0;

    let half_w = mon_size.width as f64 / scale / 2.0;
    let half_h = mon_size.height as f64 / scale / 2.0;

    let dx = ((screen_cx - win_cx) / half_w).clamp(-1.0, 1.0);
    let dy = ((screen_cy - win_cy) / half_h).clamp(-1.0, 1.0);

    Some((dx, dy))
}

#[tauri::command]
fn show_context_menu(app: AppHandle) {
    if let Some(window) = app.get_webview_window("overlay") {
        let Ok(mood_sub) = SubmenuBuilder::new(&app, "Mood")
            .text("mood_amused", "Amused")
            .text("mood_melancholy", "Melancholy")
            .text("mood_annoyed", "Annoyed")
            .text("mood_curious", "Surprised")
            .text("mood_default", "Neutral")
            .build() else { return };

        // Animation submenu — dynamically scan public/animations/ for FBX files
        let mut anim_builder = SubmenuBuilder::new(&app, "Animation")
            .text("anim_none", "None");

        // Scan for FBX files in public/animations/
        // In dev, CWD is src-tauri/ so we go up one level to reach the project root.
        let cwd = std::env::current_dir().unwrap_or_default();
        let from_root = cwd.join("public").join("animations");
        let from_src_tauri = cwd.join("..").join("public").join("animations");
        let search_dir = if from_root.exists() { from_root } else { from_src_tauri };

        if search_dir.exists() {
            if let Ok(entries) = fs::read_dir(search_dir) {
                let mut fbx_names: Vec<String> = entries
                    .flatten()
                    .filter_map(|e| {
                        let name = e.file_name().to_string_lossy().to_string();
                        if name.to_lowercase().ends_with(".fbx") {
                            Some(name.trim_end_matches(".fbx").trim_end_matches(".FBX").to_string())
                        } else {
                            None
                        }
                    })
                    .collect();
                fbx_names.sort();

                if !fbx_names.is_empty() {
                    // Unwrap builder into a new one with separator + animation entries
                    anim_builder = anim_builder.separator();
                    for name in &fbx_names {
                        let id = format!("anim_{}", name.to_lowercase().replace(' ', "_"));
                        anim_builder = anim_builder.text(id, name);
                    }
                }
            }
        }
        let Ok(anim_sub) = anim_builder.build() else { return };

        let Ok(blur_sub) = SubmenuBuilder::new(&app, "Blur")
            .text("blur_none", "None")
            .text("blur_light", "Light")
            .text("blur_medium", "Medium")
            .text("blur_heavy", "Heavy")
            .build() else { return };

        let Ok(bg_sub) = SubmenuBuilder::new(&app, "Background")
            .text("bg_transparent", "Transparent")
            .separator()
            .text("bg_dark_purple", "Dark Purple")
            .text("bg_midnight", "Midnight Blue")
            .text("bg_purple", "Purple")
            .text("bg_ocean", "Ocean")
            .text("bg_warm", "Warm Dark")
            .separator()
            .text("bg_chroma_green", "Chroma Green")
            .text("bg_chroma_magenta", "Chroma Magenta")
            .separator()
            .text("bg_custom", "Load Image...")
            .item(&blur_sub)
            .build() else { return };

        let Ok(camera_sub) = SubmenuBuilder::new(&app, "Camera")
            .text("zoom_in", "Zoom In")
            .text("zoom_out", "Zoom Out")
            .text("camera_up", "Camera Up")
            .text("camera_down", "Camera Down")
            .separator()
            .text("toggle_dutch", "Toggle Dutch Tilt")
            .text("toggle_border", "Toggle Border")
            .build() else { return };

        if let Ok(menu) = MenuBuilder::new(&app)
            .item(&mood_sub)
            .item(&anim_sub)
            .item(&bg_sub)
            .item(&camera_sub)
            .separator()
            .text("load_avatar", "Load Avatar...")
            .text("reset_avatar", "Reset Avatar")
            .separator()
            .text("reload", "Reload")
            .text("reset_position", "Reset Position")
            .text("quit", "Quit")
            .build()
        {
            let _ = window.popup_menu(&menu);
        }
    }
}

// ── App Entry ─────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            if let Some(window) = app.get_webview_window("overlay") {
                if let Some(state) = load_window_state() {
                    use tauri::{LogicalPosition, LogicalSize};
                    let _ = window.set_position(LogicalPosition::new(state.x, state.y));
                    let _ = window.set_size(LogicalSize::new(state.width, state.height));
                }
            }
            Ok(())
        })
        .on_menu_event(|app, event| {
            let id = event.id().0.as_str();
            match id {
                "zoom_in" => {
                    if let Some(window) = app.get_webview_window("overlay") {
                        let _ = window.eval("window.__SPINDL_ZOOM_IN?.()");
                    }
                }
                "zoom_out" => {
                    if let Some(window) = app.get_webview_window("overlay") {
                        let _ = window.eval("window.__SPINDL_ZOOM_OUT?.()");
                    }
                }
                "camera_up" => {
                    if let Some(window) = app.get_webview_window("overlay") {
                        let _ = window.eval("window.__SPINDL_CAMERA_UP?.()");
                    }
                }
                "camera_down" => {
                    if let Some(window) = app.get_webview_window("overlay") {
                        let _ = window.eval("window.__SPINDL_CAMERA_DOWN?.()");
                    }
                }
                "toggle_dutch" => {
                    if let Some(window) = app.get_webview_window("overlay") {
                        let _ = window.eval("window.__SPINDL_TOGGLE_DUTCH?.()");
                    }
                }
                "toggle_border" => {
                    if let Some(window) = app.get_webview_window("overlay") {
                        let _ = window.eval("window.__SPINDL_TOGGLE_BORDER?.()");
                    }
                }
                id if id.starts_with("mood_") => {
                    if let Some(window) = app.get_webview_window("overlay") {
                        let mood = &id[5..];
                        let js = format!("window.__SPINDL_PREVIEW_MOOD?.('{}')", mood);
                        let _ = window.eval(&js);
                    }
                }
                "anim_none" => {
                    if let Some(window) = app.get_webview_window("overlay") {
                        let _ = window.eval("window.__SPINDL_PLAY_ANIMATION?.('')");
                    }
                }
                id if id.starts_with("anim_") => {
                    if let Some(window) = app.get_webview_window("overlay") {
                        let name = &id[5..];
                        let js = format!("window.__SPINDL_PLAY_ANIMATION?.('{}')", name);
                        let _ = window.eval(&js);
                    }
                }
                "bg_transparent" => {
                    if let Some(window) = app.get_webview_window("overlay") {
                        let _ = window.eval("window.__SPINDL_SET_TRANSPARENT?.(true)");
                    }
                }
                "bg_custom" => {
                    if let Some(window) = app.get_webview_window("overlay") {
                        let _ = window.eval("window.__SPINDL_SET_TRANSPARENT?.(false); window.__SPINDL_CHANGE_BG?.('custom', 0)");
                    }
                }
                id if id.starts_with("blur_") => {
                    if let Some(window) = app.get_webview_window("overlay") {
                        let blur = match id {
                            "blur_none" => 0,
                            "blur_light" => 12,
                            "blur_medium" => 25,
                            "blur_heavy" => 40,
                            _ => 0,
                        };
                        let js = format!("window.__SPINDL_SET_BLUR?.({})", blur);
                        let _ = window.eval(&js);
                    }
                }
                id if id.starts_with("bg_") => {
                    if let Some(window) = app.get_webview_window("overlay") {
                        let preset = match id {
                            "bg_dark_purple" => "darkPurple",
                            "bg_midnight" => "midnight",
                            "bg_purple" => "purple",
                            "bg_ocean" => "ocean",
                            "bg_warm" => "warmDark",
                            "bg_chroma_green" => "chromaGreen",
                            "bg_chroma_magenta" => "chromaMagenta",
                            _ => "darkPurple",
                        };
                        let js = format!("window.__SPINDL_SET_TRANSPARENT?.(false); window.__SPINDL_CHANGE_BG?.('{}')", preset);
                        let _ = window.eval(&js);
                    }
                }
                "load_avatar" => {
                    if let Some(window) = app.get_webview_window("overlay") {
                        let _ = window.eval("console.log('[RUST] load_avatar menu clicked'); document.title = 'LOAD CLICKED'; if (window.__SPINDL_LOAD_AVATAR) { window.__SPINDL_LOAD_AVATAR(); } else { document.title = 'FUNCTION NOT FOUND'; }");
                    }
                }
                "reset_avatar" => {
                    if let Some(window) = app.get_webview_window("overlay") {
                        let _ = window.eval("window.__SPINDL_RESET_AVATAR?.()");
                    }
                }
                "reload" => {
                    if let Some(window) = app.get_webview_window("overlay") {
                        let _ = window.eval("location.reload()");
                    }
                }
                "reset_position" => {
                    if let Some(window) = app.get_webview_window("overlay") {
                        use tauri::{LogicalPosition, LogicalSize};
                        let _ = window.set_position(LogicalPosition::new(100.0, 100.0));
                        let _ = window.set_size(LogicalSize::new(300.0, 400.0));
                        save_window_state(&WindowState {
                            x: 100.0,
                            y: 100.0,
                            width: 300.0,
                            height: 400.0,
                        });
                    }
                }
                "quit" => {
                    if let Some(window) = app.get_webview_window("overlay") {
                        if let Some(state) = capture_webview_window_state(&window) {
                            save_window_state(&state);
                        }
                    }
                    app.exit(0);
                }
                _ => {}
            }
        })
        .on_window_event(|window, event| match event {
            WindowEvent::Moved(_) | WindowEvent::Resized(_) => {
                if let Some(state) = capture_window_state(window) {
                    save_window_state(&state);
                }
            }
            WindowEvent::CloseRequested { .. } => {
                if let Some(state) = capture_window_state(window) {
                    save_window_state(&state);
                }
            }
            _ => {}
        })
        .invoke_handler(tauri::generate_handler![show_context_menu, get_cursor_position, get_viewer_direction, read_file_bytes, list_directory, get_animations_dir])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
