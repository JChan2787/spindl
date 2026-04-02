use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tauri::{Manager, WindowEvent};

// ── Window State Persistence ──────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Debug)]
struct WindowState {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

fn config_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".config").join("spindl-stream-deck").join("window.json"))
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

// ── App Entry ─────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(|app| {
            if let Some(window) = app.get_webview_window("deck") {
                if let Some(state) = load_window_state() {
                    use tauri::{LogicalPosition, LogicalSize};
                    let _ = window.set_position(LogicalPosition::new(state.x, state.y));
                    let _ = window.set_size(LogicalSize::new(state.width, state.height));
                }
            }
            Ok(())
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
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
