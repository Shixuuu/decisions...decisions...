//! `decide-tui`: the terminal studio for Decision Forge.
//!
//! Model → Data → Recipe → Launch → Run. This binary only draws and takes keys; searching,
//! hardware detection, and training happen in the Python backend it starts (see `bridge.rs`).

mod app;
mod bridge;
mod theme;
mod ui;

use ratatui::crossterm::event::{self, Event, KeyEventKind};
use std::io;
use std::time::Duration;

/// About 30 frames a second, enough for the spinners and fades to read as smooth.
const FRAME: Duration = Duration::from_millis(33);

fn main() -> io::Result<()> {
    let mut app = app::App::new().map_err(|error| {
        io::Error::new(error.kind(), format!("could not start the Python backend (set DECIDE_PYTHON): {error}"))
    })?;
    let mut terminal = ratatui::init();
    let outcome = loop {
        if let Err(error) = terminal.draw(|frame| ui::draw(frame, &app)) {
            break Err(error);
        }
        match event::poll(FRAME) {
            Ok(true) => match event::read() {
                Ok(Event::Key(key)) if key.kind == KeyEventKind::Press => app.key(key),
                Ok(_) => {}
                Err(error) => break Err(error),
            },
            Ok(false) => {}
            Err(error) => break Err(error),
        }
        app.tick();
        if app.quit {
            break Ok(());
        }
    };
    ratatui::restore();
    outcome
}
