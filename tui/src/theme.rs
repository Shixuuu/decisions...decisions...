//! Palette, glyphs, motion curves, and number formatting.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;
use std::time::{Duration, Instant};

pub const TEXT: Color = Color::Rgb(224, 224, 230);
pub const DIM: Color = Color::Rgb(132, 132, 142);
pub const FAINT: Color = Color::Rgb(78, 78, 88);
pub const TRACK: Color = Color::Rgb(42, 42, 50);
pub const ACCENT: Color = Color::Rgb(242, 152, 88);
pub const GOOD: Color = Color::Rgb(126, 202, 148);
pub const WARN: Color = Color::Rgb(228, 192, 106);
pub const BAD: Color = Color::Rgb(234, 114, 114);
/// What fades start from. The terminal background itself is never painted.
const BACKDROP: (u8, u8, u8) = (18, 18, 22);

pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
pub const EIGHTHS: [&str; 8] = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉"];
pub const BARS: [&str; 8] = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];

pub fn fg(color: Color) -> Style {
    Style::new().fg(color)
}

pub fn bold(color: Color) -> Style {
    Style::new().fg(color).add_modifier(Modifier::BOLD)
}

pub fn spinner(clock: Duration) -> &'static str {
    SPINNER[(clock.as_millis() / 80) as usize % SPINNER.len()]
}

/// Eased 0..1 progress of a transition that began at `since`.
pub fn progress(since: Instant, millis: u64) -> f32 {
    let linear = (since.elapsed().as_secs_f32() * 1000.0 / millis as f32).clamp(0.0, 1.0);
    1.0 - (1.0 - linear).powi(3)
}

fn mix(from: (u8, u8, u8), to: (u8, u8, u8), amount: f32) -> Color {
    let channel = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * amount).round() as u8;
    Color::Rgb(channel(from.0, to.0), channel(from.1, to.1), channel(from.2, to.2))
}

/// `color` at `visibility` 0..1, rising out of the backdrop.
pub fn faded(color: Color, visibility: f32) -> Color {
    match color {
        Color::Rgb(r, g, b) => mix(BACKDROP, (r, g, b), visibility),
        other => other,
    }
}

/// Text with a soft highlight sweeping across it, for work in flight.
pub fn shimmer(text: &str, clock: Duration) -> Vec<Span<'static>> {
    let length = text.chars().count() as f32;
    let head = (clock.as_secs_f32() % 1.6) / 1.6 * (length + 10.0) - 5.0;
    text.chars()
        .enumerate()
        .map(|(i, c)| {
            let glow = (1.0 - (i as f32 - head).abs() / 4.0).clamp(0.0, 1.0);
            Span::styled(c.to_string(), fg(mix((132, 132, 142), (244, 244, 248), glow)))
        })
        .collect()
}

/// 19141315 → "19.1M"
pub fn count(n: u64) -> String {
    match n {
        0..=999 => n.to_string(),
        1_000..=999_999 => format!("{:.1}K", n as f64 / 1e3),
        1_000_000..=999_999_999 => format!("{:.1}M", n as f64 / 1e6),
        _ => format!("{:.1}B", n as f64 / 1e9),
    }
}

/// Parameter count given in billions: 0.137 → "137M", 1.54 → "1.5B"
pub fn params(billions: Option<f64>) -> String {
    match billions {
        None => String::new(),
        Some(b) if b < 1.0 => format!("{:.0}M", b * 1000.0),
        Some(b) => format!("{b:.1}B"),
    }
}

/// 130s → "2m 10s"
pub fn clock(duration: Duration) -> String {
    let seconds = duration.as_secs();
    match seconds {
        0..=59 => format!("{seconds}s"),
        60..=3599 => format!("{}m {:02}s", seconds / 60, seconds % 60),
        _ => format!("{}h {:02}m", seconds / 3600, seconds % 3600 / 60),
    }
}

pub fn percent(fraction: f64) -> String {
    format!("{:.1}%", fraction * 100.0)
}
