use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tauri::{AppHandle, Manager, WindowEvent};
use tauri::menu::{MenuBuilder, SubmenuBuilder};

// ── Window State Persistence ──────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Debug)]
struct WindowState {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

fn config_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".config").join("spindl-subtitles").join("window.json"))
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

// ── Context Menu ──────────────────────────────────────────────────────

#[tauri::command]
fn show_context_menu(app: AppHandle) {
    if let Some(window) = app.get_webview_window("subtitles") {
        let Ok(bg_sub) = SubmenuBuilder::new(&app, "Background")
            .text("bg_black", "Black (Default)")
            .separator()
            .text("bg_chroma_green", "Chroma Green")
            .text("bg_chroma_magenta", "Chroma Magenta")
            .build() else { return };

        if let Ok(menu) = MenuBuilder::new(&app)
            .item(&bg_sub)
            .separator()
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
        .setup(|app| {
            if let Some(window) = app.get_webview_window("subtitles") {
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
                "bg_black" => {
                    if let Some(window) = app.get_webview_window("subtitles") {
                        let _ = window.eval("window.__SPINDL_SET_BG?.('#000000')");
                    }
                }
                "bg_chroma_green" => {
                    if let Some(window) = app.get_webview_window("subtitles") {
                        let _ = window.eval("window.__SPINDL_SET_BG?.('#00FF00')");
                    }
                }
                "bg_chroma_magenta" => {
                    if let Some(window) = app.get_webview_window("subtitles") {
                        let _ = window.eval("window.__SPINDL_SET_BG?.('#FF00FF')");
                    }
                }
                "reset_position" => {
                    if let Some(window) = app.get_webview_window("subtitles") {
                        use tauri::{LogicalPosition, LogicalSize};
                        let _ = window.set_position(LogicalPosition::new(560.0, 880.0));
                        let _ = window.set_size(LogicalSize::new(800.0, 150.0));
                        save_window_state(&WindowState {
                            x: 560.0,
                            y: 880.0,
                            width: 800.0,
                            height: 150.0,
                        });
                    }
                }
                "quit" => {
                    if let Some(window) = app.get_webview_window("subtitles") {
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
        .invoke_handler(tauri::generate_handler![show_context_menu])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
