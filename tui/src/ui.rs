//! Drawing. Every function here reads `App` and writes cells; none of them changes state.

use crate::app::{owner, short_name, App, DatasetHit, Fit, Input, ModelHit, Preview, Run, RunResult, Source, Step};
use crate::theme::{self, bold, fg, ACCENT, BAD, DIM, FAINT, GOOD, TEXT, TRACK, WARN};
use ratatui::buffer::Buffer;
use ratatui::layout::{Position, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph, Wrap};
use ratatui::Frame;
use std::time::Instant;

const DETAIL_WIDTH: u16 = 46;
const GUTTER: u16 = 4;
/// Visibility of results that answer an earlier query while the current one is in flight.
const STALE: f32 = 0.45;

pub fn draw(frame: &mut Frame, app: &App) {
    let screen = frame.area();
    if screen.width < 60 || screen.height < 16 {
        frame.render_widget(Paragraph::new("Decision Forge needs a little more room.").style(fg(DIM)), screen);
        return;
    }
    let page = Rect::new(screen.x + 3, screen.y + 1, screen.width - 6, screen.height - 2);
    if let Some(problem) = &app.fatal {
        let message = vec![
            Line::styled("Something went wrong", bold(BAD)),
            Line::raw(""),
            Line::styled(problem.as_str(), fg(TEXT)),
            Line::raw(""),
            Line::styled("Press any key to leave.", fg(DIM)),
        ];
        frame.render_widget(Paragraph::new(message).wrap(Wrap { trim: true }), page);
        return;
    }

    header(frame, app, row(page, 0));
    stepper(frame, app, row(page, 2));
    footer(frame, app, row(page, page.height - 1));

    // Each step slides in from a few cells to the right while it fades up.
    let arrival = theme::progress(app.entered_at, 240);
    let slide = ((1.0 - arrival) * 4.0).round() as u16;
    let body = Rect::new(page.x + slide, page.y + 4, page.width - slide, page.height - 6);
    match app.step {
        Step::Model => model_step(frame, app, body),
        Step::Data => data_step(frame, app, body),
        Step::Recipe => recipe_step(frame, app, body),
        Step::Launch => launch_step(frame, app, body),
        Step::Run => run_step(frame, app, body),
    }
    fade(frame.buffer_mut(), body, arrival);
}

// ── Frame around every step ─────────────────────────────────────────────────

fn header(frame: &mut Frame, app: &App, area: Rect) {
    put(frame, area, vec![Span::styled("◆ ", fg(ACCENT)), Span::styled("Decision Forge", bold(TEXT))]);
    let machine = match &app.hello {
        Some(hello) => vec![Span::styled(hello.machine.clone(), fg(FAINT))],
        None => vec![spin(app), Span::styled(" reading hardware", fg(FAINT))],
    };
    put_right(frame, area, machine);
}

fn stepper(frame: &mut Frame, app: &App, area: Rect) {
    let chosen_data = match app.source {
        Source::Hub => app.datasets.current().map_or(String::new(), |d| short_name(&d.repo_id).to_string()),
        Source::File => short_name(&app.file.text).to_string(),
        Source::Mix => "General mix".into(),
        Source::ZeroShot => "Zero shot".into(),
    };
    let steps = [
        (Step::Model, "Model", app.model.as_ref().map_or(String::new(), |m| short_name(&m.repo_id).to_string())),
        (Step::Data, "Data", chosen_data),
        (Step::Recipe, "Recipe", String::new()),
        (Step::Launch, "Launch", String::new()),
    ];
    let mut spans = Vec::new();
    for (step, label, chosen) in steps {
        if step == app.step || (app.step == Step::Run && step == Step::Launch) {
            spans.push(Span::styled("● ", fg(ACCENT)));
            spans.push(Span::styled(label, bold(TEXT)));
        } else if step < app.step {
            spans.push(Span::styled("● ", fg(FAINT)));
            spans.push(Span::styled(label, fg(DIM)));
            if !chosen.is_empty() {
                spans.push(Span::styled(format!("  {}", truncate(&chosen, 22)), fg(FAINT)));
            }
        } else {
            spans.push(Span::styled("○ ", fg(FAINT)));
            spans.push(Span::styled(label, fg(FAINT)));
        }
        spans.push(Span::raw("      "));
    }
    put(frame, area, spans);
}

fn footer(frame: &mut Frame, app: &App, area: Rect) {
    let hints: &[(&str, &str)] = match app.step {
        Step::Model => &[("↑↓", "browse"), ("↵", "choose"), ("esc", "clear"), ("^C", "quit")],
        Step::Data => match app.source {
            Source::Hub => &[("↑↓", "browse"), ("←→", "subset"), ("tab", "source"), ("↵", "choose"), ("esc", "back")],
            _ => &[("tab", "source"), ("↵", "continue"), ("esc", "back")],
        },
        Step::Recipe => &[("↑↓", "setting"), ("←→", "change"), ("↵", "continue"), ("esc", "back")],
        Step::Launch => &[("↑↓", "target"), ("↵", "start"), ("esc", "back")],
        Step::Run => match &app.run {
            Some(run) if run.outcome.is_some() => &[("↵", "new run"), ("esc", "back"), ("q", "quit")],
            Some(run) if run.cancel_armed => &[("esc", "press again to stop the run")],
            _ => &[("esc", "stop")],
        },
    };
    let mut spans = Vec::new();
    for (key, action) in hints {
        spans.push(Span::styled(*key, fg(DIM)));
        spans.push(Span::styled(format!(" {action}    "), fg(FAINT)));
    }
    put(frame, area, spans);
}

// ── Model ───────────────────────────────────────────────────────────────────

fn model_step(frame: &mut Frame, app: &App, area: Rect) {
    let (list, detail) = split_detail(area);
    let search = &app.models;
    let status = if search.busy() {
        vec![spin(app), Span::styled(" searching", fg(DIM))]
    } else {
        vec![Span::styled(format!("{} models", search.items().len()), fg(FAINT))]
    };
    search_bar(frame, rows(list, 0, 3), &search.input, "Search Hugging Face models", status);

    let results = rows(list, 4, list.height.saturating_sub(4));
    if let Some(error) = &search.results.error {
        problem(frame, results, error);
    } else if search.items().is_empty() {
        let typed = search.input.text.trim();
        if search.busy() {
            put(frame, row(results, 0), theme::shimmer("Searching the Hugging Face catalog", app.opened.elapsed()));
        } else {
            put(frame, row(results, 0), vec![Span::styled(format!("Nothing on the Hub matches “{typed}”."), fg(DIM))]);
            put(frame, row(results, 2), vec![Span::styled("↵ uses it as a repo id or a local folder.", fg(FAINT))]);
        }
    } else {
        result_list(
            frame,
            results,
            search.items().len(),
            search.selected,
            search.moved_at,
            search.results.loaded_at,
            |i, width| model_row(&search.items()[i], width),
        );
        fade(frame.buffer_mut(), results, if search.busy() { STALE } else { 1.0 });
    }

    if let (Some(pane), Some(hit)) = (detail, search.current()) {
        model_detail(frame, pane, hit);
    }
}

fn model_row(hit: &ModelHit, width: u16) -> (Vec<Span<'static>>, Vec<Span<'static>>) {
    let name = short_name(&hit.info.repo_id).to_string();
    let org = truncate(owner(&hit.info.repo_id), 18);
    let right = vec![
        Span::styled(format!("{:>6}", theme::params(hit.info.param_count_b)), fg(DIM)),
        Span::styled(format!("{:>9} ↓  ", theme::count(hit.info.downloads)), fg(FAINT)),
        Span::styled("●", fg(fit_color(&hit.fit))),
    ];
    let room = (width as usize).saturating_sub(22 + org.chars().count());
    (vec![Span::styled(truncate(&name, room), fg(TEXT)), Span::styled(format!("  {org}"), fg(FAINT))], right)
}

fn model_detail(frame: &mut Frame, area: Rect, hit: &ModelHit) {
    let info = &hit.info;
    put(frame, row(area, 0), vec![Span::styled(short_name(&info.repo_id).to_string(), bold(TEXT))]);
    put(frame, row(area, 1), vec![Span::styled(owner(&info.repo_id).to_string(), fg(DIM))]);

    let mut traits = Vec::new();
    if info.has_chat_template {
        traits.push("chat template");
    }
    if info.supports_thinking_toggle {
        traits.push("thinking toggle");
    }
    if info.requires_trust_remote_code {
        traits.push("runs custom code");
    }
    let facts = [
        ("Parameters", theme::params(info.param_count_b)),
        ("Architecture", [info.architecture_family.as_str(), info.model_type.as_str()].join(" · ")),
        ("License", info.license.clone().unwrap_or_default()),
        ("Downloads", theme::count(info.downloads)),
        ("Likes", theme::count(info.likes)),
        ("Prompting", traits.join(" · ")),
    ];
    let mut y = 3;
    for (label, value) in facts.iter().filter(|(_, value)| !value.is_empty()) {
        put(
            frame,
            row(area, y),
            vec![Span::styled(format!("{label:<14}"), fg(FAINT)), Span::styled(value.clone(), fg(TEXT))],
        );
        y += 1;
    }

    put(frame, row(area, y + 1), vec![Span::styled("Fits for LoRA training", fg(DIM))]);
    for (i, fit) in hit.fits.iter().enumerate() {
        put(frame, row(area, y + 2 + i as u16), fit_line(fit));
    }
}

fn fit_line(fit: &Fit) -> Vec<Span<'static>> {
    let verdict = match fit.level.as_str() {
        "fits" => "fits",
        "tight" => "tight",
        "qlora" => "fits in 4 bit",
        "too_large" => "too large",
        _ => "unknown",
    };
    // "3.4GB est. on 15.4GB shared memory" → "3.4 of 15.4 GB"
    let sizes = fit.detail.replace("GB est. on ", " of ").replace("GB est. > ", " over ");
    let sizes = sizes.split_once("GB").map_or(String::new(), |(sizes, _)| format!("{sizes} GB"));
    vec![
        Span::styled("● ", fg(fit_color(&fit.level))),
        Span::styled(format!("{:<12}", fit.label), fg(TEXT)),
        Span::styled(format!("{verdict:<15}"), fg(DIM)),
        Span::styled(sizes, fg(FAINT)),
    ]
}

fn fit_color(level: &str) -> Color {
    match level {
        "fits" => GOOD,
        "tight" | "qlora" => WARN,
        "too_large" => BAD,
        _ => FAINT,
    }
}

// ── Data ────────────────────────────────────────────────────────────────────

fn data_step(frame: &mut Frame, app: &App, area: Rect) {
    let mut tabs = Vec::new();
    for (source, label) in Source::ALL {
        tabs.push(Span::styled(label, if source == app.source { bold(ACCENT) } else { fg(DIM) }));
        tabs.push(Span::raw("     "));
    }
    put(frame, row(area, 0), tabs);
    let area = rows(area, 2, area.height.saturating_sub(2));

    match app.source {
        Source::Hub => hub_source(frame, app, area),
        Source::File => file_source(frame, app, area),
        Source::Mix => {
            let intro = "A balanced blend of public tasks, for a decision model that is not tied to one domain.";
            put(frame, row(area, 0), vec![Span::styled(intro, fg(TEXT))]);
            for (i, recipe) in app.hello.iter().flat_map(|h| &h.mix).enumerate() {
                let line = vec![
                    Span::styled(format!("{:<18}", recipe.name), fg(TEXT)),
                    Span::styled(format!("{:<10}", recipe.primitive), fg(DIM)),
                    Span::styled(format!("{}  {}", short_name(&recipe.repo_id), owner(&recipe.repo_id)), fg(FAINT)),
                ];
                put(frame, row(area, 2 + i as u16), line);
            }
        }
        Source::ZeroShot => {
            put(frame, row(area, 0), vec![Span::styled("No dataset and no weight updates.", fg(TEXT))]);
            let note = "The model is wrapped as it is and its answers are averaged over rotated option orders.";
            put(frame, row(area, 1), vec![Span::styled(note, fg(DIM))]);
        }
    }
}

fn hub_source(frame: &mut Frame, app: &App, area: Rect) {
    let (list, detail) = split_detail(area);
    let search = &app.datasets;
    let status = if search.busy() {
        vec![spin(app), Span::styled(" searching", fg(DIM))]
    } else {
        vec![Span::styled(format!("{} datasets", search.items().len()), fg(FAINT))]
    };
    search_bar(frame, rows(list, 0, 3), &search.input, "Search Hugging Face datasets", status);

    let results = rows(list, 4, list.height.saturating_sub(4));
    if let Some(error) = &search.results.error {
        problem(frame, results, error);
    } else if search.items().is_empty() {
        let line = if search.busy() {
            theme::shimmer("Searching the Hugging Face catalog", app.opened.elapsed())
        } else {
            vec![Span::styled(format!("No dataset matches “{}”.", search.input.text.trim()), fg(DIM))]
        };
        put(frame, row(results, 0), line);
    } else {
        result_list(
            frame,
            results,
            search.items().len(),
            search.selected,
            search.moved_at,
            search.results.loaded_at,
            |i, width| dataset_row(&search.items()[i], width),
        );
        fade(frame.buffer_mut(), results, if search.busy() { STALE } else { 1.0 });
    }

    let (Some(pane), Some(hit)) = (detail, search.current()) else { return };
    put(frame, row(pane, 0), vec![Span::styled(short_name(&hit.repo_id).to_string(), bold(TEXT))]);
    put(frame, row(pane, 1), vec![Span::styled(owner(&hit.repo_id).to_string(), fg(DIM))]);
    let below = rows(pane, 3, pane.height.saturating_sub(3));
    match (&app.preview.value, &app.preview.error) {
        (Some(preview), _) => {
            let mut subset = vec![
                Span::styled("Subset ", fg(FAINT)),
                Span::styled(preview.config.clone(), fg(TEXT)),
                Span::styled("   Split ", fg(FAINT)),
                Span::styled(preview.split.clone(), fg(TEXT)),
            ];
            if app.preview.loading() {
                subset.extend([Span::raw("  "), spin(app)]);
            } else if preview.configs.len() > 1 {
                subset.push(Span::styled(format!("   ←→ {} subsets", preview.configs.len()), fg(FAINT)));
            }
            put(frame, row(below, 0), subset);
            preview_table(frame, rows(below, 1, below.height.saturating_sub(1)), preview);
            fade(frame.buffer_mut(), below, theme::progress(app.preview.loaded_at, 260));
        }
        (None, Some(error)) => problem(frame, below, error),
        (None, None) => put(frame, row(below, 0), theme::shimmer("Reading the first rows", app.opened.elapsed())),
    }
}

fn dataset_row(hit: &DatasetHit, width: u16) -> (Vec<Span<'static>>, Vec<Span<'static>>) {
    let name = short_name(&hit.repo_id).to_string();
    let org = truncate(owner(&hit.repo_id), 18);
    let task = hit.tasks.first().map_or(String::new(), |t| t.replace('-', " "));
    let right = vec![
        Span::styled(if hit.gated { "gated  " } else { "" }, fg(WARN)),
        Span::styled(truncate(&task, 22), fg(DIM)),
        Span::styled(format!("{:>9} ↓", theme::count(hit.downloads)), fg(FAINT)),
    ];
    let room = (width as usize).saturating_sub(44 + org.chars().count());
    (vec![Span::styled(truncate(&name, room), fg(TEXT)), Span::styled(format!("  {org}"), fg(FAINT))], right)
}

fn file_source(frame: &mut Frame, app: &App, area: Rect) {
    let lookup = &app.file_preview;
    let status = match &lookup.value {
        _ if lookup.loading() => vec![spin(app), Span::styled(" reading", fg(DIM))],
        Some(preview) => {
            vec![Span::styled(format!("{} rows", theme::count(preview.total_rows.unwrap_or(0))), fg(FAINT))]
        }
        None => Vec::new(),
    };
    search_bar(frame, rows(area, 0, 3), &app.file, "Path to a CSV, TSV, JSON or JSONL file", status);
    let below = rows(area, 4, area.height.saturating_sub(4));
    match (&lookup.value, &lookup.error) {
        (Some(preview), _) => {
            preview_table(frame, below, preview);
            fade(frame.buffer_mut(), below, theme::progress(lookup.loaded_at, 260));
        }
        (None, Some(error)) => problem(frame, below, error),
        (None, None) => {}
    }
}

/// Schema line, then the first rows with the text and label columns leading.
fn preview_table(frame: &mut Frame, area: Rect, preview: &Preview) {
    let schema = &preview.schema;
    let reading = match (&schema.state_col, &schema.label_col) {
        (Some(text), Some(label)) => {
            let kind = schema.schema_type.clone().unwrap_or_default().replace('_', " ");
            vec![Span::styled(format!("{kind}   "), fg(ACCENT)), Span::styled(format!("{text} → {label}"), fg(DIM))]
        }
        _ => vec![Span::styled("No text and label columns found", fg(WARN))],
    };
    put(frame, row(area, 0), reading);

    let mut order: Vec<usize> = (0..preview.columns.len()).collect();
    order.sort_by_key(|i| match Some(&preview.columns[*i]) {
        name if name == schema.state_col.as_ref() => 0,
        name if name == schema.label_col.as_ref() => 1,
        _ => 2,
    });
    let shown: Vec<usize> = order.into_iter().take(3).collect();
    let narrow = 12u16.min(area.width / 4);
    let wide = area.width.saturating_sub(narrow * shown.len().saturating_sub(1) as u16 + 2 * shown.len() as u16);
    let width_of = |position: usize| if position == 0 { wide } else { narrow } as usize;

    let cells = |values: Vec<&str>, style: Style| -> Vec<Span<'static>> {
        values
            .into_iter()
            .enumerate()
            .map(|(position, value)| {
                Span::styled(format!("{:<w$}  ", truncate(value, width_of(position)), w = width_of(position)), style)
            })
            .collect()
    };
    put(frame, row(area, 2), cells(shown.iter().map(|i| preview.columns[*i].as_str()).collect(), fg(FAINT)));
    for (y, record) in preview.rows.iter().enumerate() {
        put(frame, row(area, 3 + y as u16), cells(shown.iter().map(|i| record[*i].as_str()).collect(), fg(DIM)));
    }
}

// ── Recipe ──────────────────────────────────────────────────────────────────

fn recipe_step(frame: &mut Frame, app: &App, area: Rect) {
    let mut y = 0;
    for index in app.visible_fields() {
        let field = &app.fields[index];
        if field.starts_group && y > 0 {
            y += 1;
        }
        let focused = index == app.focus;
        let mut spans = vec![
            Span::styled(if focused { "▌ " } else { "  " }, fg(ACCENT)),
            Span::styled(format!("{:<18}", field.label), if focused { bold(TEXT) } else { fg(DIM) }),
        ];
        for (position, (_, label)) in field.options.iter().enumerate() {
            let picked = position == field.picked;
            let style = match (picked, focused) {
                (true, true) => bold(ACCENT),
                (true, false) => fg(TEXT),
                (false, true) => fg(DIM),
                (false, false) => fg(FAINT),
            };
            spans.push(Span::styled(if picked { "● " } else { "  " }, style));
            spans.push(Span::styled(*label, style));
            spans.push(Span::raw("   "));
        }
        put(frame, row(area, y), spans);
        y += 1;
    }
    let help = app.fields[app.focus].help;
    put(frame, row(area, y + 1), vec![Span::raw("  "), Span::styled(help, fg(FAINT))]);
}

// ── Launch ──────────────────────────────────────────────────────────────────

fn launch_step(frame: &mut Frame, app: &App, area: Rect) {
    let Some(model) = &app.model else { return };
    let data = match app.source {
        Source::Hub => match (app.datasets.current(), &app.preview.value) {
            (Some(hit), Some(preview)) => {
                format!("{} · {} · {}", short_name(&hit.repo_id), preview.config, preview.split)
            }
            (Some(hit), None) => short_name(&hit.repo_id).to_string(),
            _ => String::new(),
        },
        Source::File => short_name(&app.file.text).to_string(),
        Source::Mix => "General mix of public tasks".into(),
        Source::ZeroShot => "None, zero shot".into(),
    };
    let recipe: Vec<String> = app
        .visible_fields()
        .map(|i| &app.fields[i])
        .map(|f| format!("{} {}", f.label.to_lowercase(), f.options[f.picked].1))
        .collect();
    let size = theme::params(model.params);
    let summary = [
        ("Model", [short_name(&model.repo_id), size.as_str()].join("  ")),
        ("Data", data),
        ("Recipe", recipe.join(" · ")),
    ];
    let mut y = 0;
    for (label, value) in summary {
        put(frame, row(area, y), vec![Span::styled(format!("{label:<10}"), fg(FAINT))]);
        let text =
            Rect::new(area.x + 10, area.y + y, area.width.saturating_sub(10), 3.min(area.height.saturating_sub(y)));
        let wrapped_lines = value.chars().count().div_ceil(text.width.max(1) as usize).clamp(1, 3);
        frame.render_widget(Paragraph::new(value).style(fg(TEXT)).wrap(Wrap { trim: true }), text);
        y += wrapped_lines as u16;
    }

    put(frame, row(area, y + 1), vec![Span::styled("Where should it run?", bold(TEXT))]);
    let Some(hello) = &app.hello else {
        put(frame, row(area, y + 3), theme::shimmer("Reading this machine's hardware", app.opened.elapsed()));
        return;
    };
    for (index, target) in hello.targets.iter().enumerate() {
        let focused = index == app.target;
        let fit_key = if target.key == "colab" { "colab_t4" } else { target.key.as_str() };
        let mut spans = vec![
            Span::styled(if focused { "▌ " } else { "  " }, fg(ACCENT)),
            Span::styled(
                format!("{:<18}", target.label),
                if !target.ready {
                    fg(FAINT)
                } else if focused {
                    bold(TEXT)
                } else {
                    fg(DIM)
                },
            ),
        ];
        if !target.ready {
            spans.push(Span::styled(target.note.clone(), fg(FAINT)));
        } else if let Some(fit) = model.fits.iter().find(|f| f.key == fit_key) {
            spans.extend(fit_line(fit).into_iter().enumerate().filter(|(i, _)| *i != 1).map(|(_, span)| span));
        }
        put(frame, row(area, y + 3 + index as u16), spans);
    }
}

// ── Run ─────────────────────────────────────────────────────────────────────

fn run_step(frame: &mut Frame, app: &App, area: Rect) {
    let Some(run) = &app.run else { return };
    put(frame, row(area, 0), vec![Span::styled(run.title.clone(), bold(TEXT))]);
    put_right(frame, row(area, 0), vec![Span::styled(theme::clock(run.took), fg(DIM))]);

    let mut y = 2;
    for (index, phase) in run.phases.iter().enumerate() {
        let failed = matches!(run.outcome, Some(Err(_)));
        let label = phase_label(&phase.name);
        let mut spans = match index.cmp(&run.active) {
            std::cmp::Ordering::Less => {
                vec![Span::styled("✓ ", fg(GOOD)), Span::styled(format!("{label:<14}"), fg(DIM))]
            }
            std::cmp::Ordering::Equal if failed => {
                vec![Span::styled("✗ ", fg(BAD)), Span::styled(format!("{label:<14}"), fg(TEXT))]
            }
            std::cmp::Ordering::Equal => {
                let mut spans = vec![spin(app), Span::raw(" ")];
                spans.extend(theme::shimmer(&format!("{label:<14}"), app.opened.elapsed()));
                spans
            }
            std::cmp::Ordering::Greater => {
                vec![Span::styled("○ ", fg(FAINT)), Span::styled(format!("{label:<14}"), fg(FAINT))]
            }
        };
        spans.push(Span::styled(phase.detail.clone(), fg(FAINT)));
        put(frame, row(area, y), spans);
        y += 1;
    }
    if run.phases.is_empty() {
        put(frame, row(area, y), theme::shimmer("Preparing the session", app.opened.elapsed()));
        y += 1;
    }
    y += 1;

    match &run.outcome {
        Some(Ok(result)) => run_result(frame, rows(area, y, area.height.saturating_sub(y)), result),
        Some(Err(error)) => problem(frame, rows(area, y, area.height.saturating_sub(y)), error),
        None => {
            if let Some(step) = run.latest {
                training_progress(frame, rows(area, y, 4), run, step);
                y += 5;
            }
            put(frame, row(area, y), vec![Span::styled(truncate(&run.log, area.width as usize), fg(FAINT))]);
        }
    }
}

fn phase_label(name: &str) -> &str {
    match name {
        "weights" => "Weights",
        "dataset" => "Dataset",
        "train" => "Training",
        "calibrate" => "Calibration",
        "bench" => "DecideBench",
        "export" => "GGUF export",
        "notebook" => "Notebook",
        other => other,
    }
}

fn training_progress(frame: &mut Frame, area: Rect, run: &Run, step: crate::app::TrainStep) {
    let bar = row(area, 0);
    let width = bar.width.saturating_sub(7);
    let filled = run.shown.clamp(0.0, 1.0) * width as f64;
    let buffer = frame.buffer_mut();
    for cell in 0..width {
        let fill = (filled - cell as f64).clamp(0.0, 1.0);
        let glyph = if fill >= 1.0 { "█" } else { theme::EIGHTHS[(fill * 8.0) as usize] };
        buffer.set_string(bar.x + cell, bar.y, glyph, Style::new().fg(ACCENT).bg(TRACK));
    }
    put_right(frame, bar, vec![Span::styled(format!("{:.0}%", run.shown * 100.0), bold(TEXT))]);

    let mut facts = vec![
        format!("step {} of {}", step.step, step.total_steps),
        format!("epoch {}", step.epoch),
        format!("loss {:.3}", step.loss),
        format!("lr {:.1e}", step.lr),
    ];
    if let Some(first) = run.first_step_at.filter(|_| step.step > 1 && step.step < step.total_steps) {
        let per_step = first.elapsed().as_secs_f64() / (step.step - 1) as f64;
        let left = std::time::Duration::from_secs_f64(per_step * (step.total_steps - step.step) as f64);
        facts.push(format!("about {} left", theme::clock(left)));
    }
    put(frame, row(area, 2), vec![Span::styled(facts.join("  ·  "), fg(DIM))]);

    // Loss history, newest on the right, scaled between its own lowest and highest value.
    let recent = &run.losses[run.losses.len().saturating_sub((area.width as usize).saturating_sub(6))..];
    let (low, high) = recent
        .iter()
        .copied()
        .filter(|loss| loss.is_finite())
        .fold((f64::MAX, f64::MIN), |(low, high), l| (low.min(l), high.max(l)));
    let spark: String = recent
        .iter()
        .map(|loss| {
            let idx = if loss.is_finite() {
                (((loss - low) / (high - low).max(1e-9)) * 7.0).round().clamp(0.0, 7.0) as usize
            } else {
                0
            };
            theme::BARS[idx]
        })
        .collect();
    put(frame, row(area, 3), vec![Span::styled(spark, fg(FAINT)), Span::styled("  loss", fg(FAINT))]);
}

fn run_result(frame: &mut Frame, area: Rect, result: &RunResult) {
    let mut lines: Vec<Vec<Span>> = Vec::new();
    let fact = |label: &str, value: String| {
        vec![Span::styled(format!("{label:<22}"), fg(FAINT)), Span::styled(value, fg(TEXT))]
    };

    if let Some(notebook) = &result.notebook_path {
        lines.push(vec![Span::styled("◆ ", fg(ACCENT)), Span::styled("Notebook ready", bold(TEXT))]);
        lines.push(Vec::new());
        lines.push(fact("Saved to", notebook.clone()));
        lines.push(Vec::new());
        lines.push(vec![Span::styled(
            "Upload it to colab.research.google.com, pick a GPU runtime, then Run all.",
            fg(DIM),
        )]);
    } else {
        lines.push(vec![Span::styled("◆ ", fg(ACCENT)), Span::styled("Decision model ready", bold(TEXT))]);
        lines.push(Vec::new());
        if let Some(metrics) = &result.metrics {
            if let Some(accuracy) = metrics.val_accuracy {
                lines.push(fact("Validation accuracy", theme::percent(accuracy)));
            }
            if let (Some(before), Some(after)) = (metrics.nll_before, metrics.nll_after) {
                lines.push(fact("Negative log likelihood", format!("{before:.3} → {after:.3} after calibration")));
            }
            if let Some(ece) = metrics.ece_10_after {
                lines.push(fact("Calibration error", format!("{ece:.3}")));
            }
            if let Some(flips) = metrics.order_flip_rate {
                lines.push(fact("Answers that flip", format!("{} when options are reordered", theme::percent(flips))));
            }
        }
        for score in &result.bench {
            let label = format!("DecideBench {} {}", score.version, score.protocol.replace('_', " "));
            let range = format!(
                "{}   {} to {}",
                theme::percent(score.accuracy),
                theme::percent(score.ci_low),
                theme::percent(score.ci_high)
            );
            lines.push(fact(&label, range));
        }
        if let Some(model_dir) = &result.model_dir {
            lines.push(Vec::new());
            lines.push(fact("Saved to", model_dir.clone()));
            lines.push(fact("Serve it", format!("decide serve --model-dir {model_dir}")));
        }
    }
    for (y, line) in lines.into_iter().enumerate() {
        put(frame, row(area, y as u16), line);
    }
}

// ── Shared pieces ───────────────────────────────────────────────────────────

/// A rounded input box: magnifier, text or placeholder, and a status on the right.
/// Places the terminal cursor after the text.
fn search_bar(frame: &mut Frame, area: Rect, input: &Input, placeholder: &str, status: Vec<Span>) {
    let block = Block::bordered().border_type(BorderType::Rounded).border_style(fg(FAINT));
    let inside = block.inner(area);
    frame.render_widget(block, area);
    let inside = Rect::new(inside.x + 1, inside.y, inside.width.saturating_sub(2), 1);

    let status_width = Line::from(status.clone()).width() as u16;
    put_right(frame, inside, status);
    let room = inside.width.saturating_sub(status_width + 5) as usize;
    let typed: String = {
        // Keep the end of long text in view, where the cursor is.
        let chars: Vec<char> = input.text.chars().collect();
        chars[chars.len().saturating_sub(room)..].iter().collect()
    };
    let text = if typed.is_empty() {
        Span::styled(placeholder.to_string(), fg(FAINT))
    } else {
        Span::styled(typed.clone(), fg(TEXT))
    };
    put(frame, inside, vec![Span::styled("⌕  ", fg(ACCENT)), text]);
    frame.set_cursor_position(Position::new(inside.x + 3 + Line::raw(typed).width() as u16, inside.y));
}

/// A scrolling list. `render` returns the left and right halves of row `i`.
/// New results fade in top to bottom; the selection marker grows in when it moves.
fn result_list(
    frame: &mut Frame,
    area: Rect,
    count: usize,
    selected: usize,
    moved_at: Instant,
    loaded_at: Instant,
    render: impl Fn(usize, u16) -> (Vec<Span<'static>>, Vec<Span<'static>>),
) {
    let capacity = area.height as usize;
    let first = (selected + 1).saturating_sub(capacity);
    for (line, index) in (first..count).take(capacity).enumerate() {
        let line_area = row(area, line as u16);
        let text_area = Rect::new(line_area.x + 2, line_area.y, line_area.width.saturating_sub(2), 1);
        let (mut left, right) = render(index, text_area.width);
        if index == selected {
            let grown = theme::progress(moved_at, 110);
            let marker = theme::EIGHTHS[1 + (grown * 3.0).round() as usize];
            frame.buffer_mut().set_string(line_area.x, line_area.y, marker, fg(ACCENT));
            left[0] = left[0].clone().style(bold(TEXT));
        }
        put(frame, text_area, left);
        put_right(frame, text_area, right);

        let delay = std::time::Duration::from_millis(16 * line as u64);
        let visible = loaded_at.checked_add(delay).map_or(1.0, |start| {
            if Instant::now() < start {
                0.0
            } else {
                theme::progress(start, 220)
            }
        });
        fade(frame.buffer_mut(), line_area, visible);
    }
}

fn problem(frame: &mut Frame, area: Rect, message: &str) {
    frame.render_widget(Paragraph::new(message).style(fg(BAD)).wrap(Wrap { trim: true }), area);
}

fn spin(app: &App) -> Span<'static> {
    Span::styled(theme::spinner(app.opened.elapsed()), fg(ACCENT))
}

/// Splits off a detail pane on the right when the terminal is wide enough for one.
fn split_detail(area: Rect) -> (Rect, Option<Rect>) {
    if area.width < 2 * DETAIL_WIDTH + GUTTER {
        return (area, None);
    }
    let list = Rect::new(area.x, area.y, area.width - DETAIL_WIDTH - GUTTER, area.height);
    (list, Some(Rect::new(list.right() + GUTTER, area.y, DETAIL_WIDTH, area.height)))
}

fn row(area: Rect, offset: u16) -> Rect {
    rows(area, offset, 1)
}

/// `height` rows of `area` starting `offset` rows down, clipped to the area.
fn rows(area: Rect, offset: u16, height: u16) -> Rect {
    let offset = offset.min(area.height);
    Rect::new(area.x, area.y + offset, area.width, height.min(area.height - offset))
}

fn put(frame: &mut Frame, area: Rect, spans: Vec<Span>) {
    if area.height > 0 {
        frame.buffer_mut().set_line(area.x, area.y, &Line::from(spans), area.width);
    }
}

fn put_right(frame: &mut Frame, area: Rect, spans: Vec<Span>) {
    let line = Line::from(spans);
    let width = (line.width() as u16).min(area.width);
    if area.height > 0 {
        frame.buffer_mut().set_line(area.right() - width, area.y, &line, width);
    }
}

/// Dims everything already drawn in `area` toward the backdrop; `visibility` 1 leaves it as is.
fn fade(buffer: &mut Buffer, area: Rect, visibility: f32) {
    if visibility >= 1.0 {
        return;
    }
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            let cell = &mut buffer[(x, y)];
            cell.fg = theme::faded(cell.fg, visibility);
            cell.bg = theme::faded(cell.bg, visibility);
        }
    }
}

fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_string();
    }
    let kept: String = text.chars().take(width.saturating_sub(1)).collect();
    format!("{kept}…")
}
