//! Everything the screen shows, and how keys and backend messages change it.

use crate::bridge::{Bridge, Incoming};
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::io;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant};

const TYPING_PAUSE: Duration = Duration::from_millis(220);
const BROWSING_PAUSE: Duration = Duration::from_millis(250);

// ── What the backend sends ──────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct Hello {
    pub machine: String,
    pub targets: Vec<Target>,
    pub mix: Vec<MixRecipe>,
}

#[derive(Deserialize)]
pub struct Target {
    pub key: String,
    pub label: String,
    pub ready: bool,
    pub note: String,
}

#[derive(Deserialize)]
pub struct MixRecipe {
    pub name: String,
    pub repo_id: String,
    pub primitive: String,
}

#[derive(Deserialize)]
pub struct ModelHit {
    pub info: ModelInfo,
    pub fits: Vec<Fit>,
    /// Fit level on the local device a run would use by default.
    pub fit: String,
}

#[derive(Deserialize)]
pub struct ModelInfo {
    pub repo_id: String,
    pub architecture_family: String,
    pub model_type: String,
    pub param_count_b: Option<f64>,
    pub downloads: u64,
    pub likes: u64,
    pub license: Option<String>,
    pub has_chat_template: bool,
    pub supports_thinking_toggle: bool,
    pub requires_trust_remote_code: bool,
}

#[derive(Deserialize, Clone)]
pub struct Fit {
    pub key: String,
    pub label: String,
    pub level: String,
    pub detail: String,
}

#[derive(Deserialize)]
pub struct DatasetHit {
    pub repo_id: String,
    pub downloads: u64,
    pub tasks: Vec<String>,
    pub gated: bool,
}

#[derive(Deserialize)]
pub struct Preview {
    #[serde(default)]
    pub config: String,
    #[serde(default)]
    pub configs: Vec<String>,
    #[serde(default)]
    pub split: String,
    pub path: Option<String>,
    pub total_rows: Option<u64>,
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub schema: Schema,
}

#[derive(Deserialize)]
pub struct Schema {
    pub schema_type: Option<String>,
    pub state_col: Option<String>,
    pub label_col: Option<String>,
}

#[derive(Deserialize)]
pub struct RunResult {
    pub notebook_path: Option<String>,
    pub model_dir: Option<String>,
    pub metrics: Option<Metrics>,
    #[serde(default)]
    pub bench: Vec<BenchScore>,
}

#[derive(Deserialize)]
pub struct Metrics {
    pub val_accuracy: Option<f64>,
    pub nll_before: Option<f64>,
    pub nll_after: Option<f64>,
    pub ece_10_after: Option<f64>,
    pub order_flip_rate: Option<f64>,
}

#[derive(Deserialize)]
pub struct BenchScore {
    pub version: String,
    pub protocol: String,
    pub accuracy: f64,
    pub ci_low: f64,
    pub ci_high: f64,
}

// ── Building blocks ─────────────────────────────────────────────────────────

/// A line of text the user types into. `settled` fires once per pause in typing.
#[derive(Default)]
pub struct Input {
    pub text: String,
    edited_at: Option<Instant>,
}

impl Input {
    fn edit(&mut self, key: &KeyEvent) -> bool {
        let before = self.text.len();
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('u') if ctrl => self.text.clear(),
            KeyCode::Char('w') if ctrl => {
                let kept = self.text.trim_end().rfind(' ').map_or(0, |space| space + 1);
                self.text.truncate(kept);
            }
            KeyCode::Char(c) if !ctrl && !key.modifiers.contains(KeyModifiers::ALT) => self.text.push(c),
            KeyCode::Backspace => {
                self.text.pop();
            }
            _ => return false,
        }
        if self.text.len() != before {
            self.edited_at = Some(Instant::now());
        }
        true
    }

    fn settled(&mut self) -> bool {
        let settled = self.edited_at.is_some_and(|at| at.elapsed() >= TYPING_PAUSE);
        if settled {
            self.edited_at = None;
        }
        settled
    }
}

/// The latest answer to a question asked of the backend. Answers to questions
/// that were since superseded are dropped.
pub struct Lookup<T> {
    pending: Option<u64>,
    pub value: Option<T>,
    pub error: Option<String>,
    pub loaded_at: Instant,
}

impl<T> Default for Lookup<T> {
    fn default() -> Self {
        Self { pending: None, value: None, error: None, loaded_at: Instant::now() }
    }
}

impl<T: DeserializeOwned> Lookup<T> {
    pub fn loading(&self) -> bool {
        self.pending.is_some()
    }

    fn accept(&mut self, id: u64, result: &Result<Value, String>) -> bool {
        if self.pending != Some(id) {
            return false;
        }
        self.pending = None;
        let parsed = result.clone().and_then(|value| serde_json::from_value(value).map_err(|e| e.to_string()));
        match parsed {
            Ok(value) => (self.value, self.error) = (Some(value), None),
            Err(error) => (self.value, self.error) = (None, Some(error)),
        }
        self.loaded_at = Instant::now();
        true
    }
}

/// A search bar with its live result list.
pub struct Search<T> {
    pub input: Input,
    pub results: Lookup<Vec<T>>,
    pub selected: usize,
    pub moved_at: Instant,
}

impl<T> Default for Search<T> {
    fn default() -> Self {
        Self { input: Input::default(), results: Lookup::default(), selected: 0, moved_at: Instant::now() }
    }
}

impl<T: DeserializeOwned> Search<T> {
    pub fn items(&self) -> &[T] {
        self.results.value.as_deref().unwrap_or(&[])
    }

    pub fn current(&self) -> Option<&T> {
        self.items().get(self.selected)
    }

    pub fn busy(&self) -> bool {
        self.input.edited_at.is_some() || self.results.loading()
    }

    fn step(&mut self, down: bool) {
        let last = self.items().len().saturating_sub(1);
        let next = if down { (self.selected + 1).min(last) } else { self.selected.saturating_sub(1) };
        if next != self.selected {
            self.selected = next;
            self.moved_at = Instant::now();
        }
    }

    fn accept(&mut self, id: u64, result: &Result<Value, String>) -> bool {
        let accepted = self.results.accept(id, result);
        if accepted {
            self.selected = 0;
        }
        accepted
    }
}

// ── The recipe form ─────────────────────────────────────────────────────────

/// One setting, named by the `decide train` argument it fills.
pub struct Field {
    pub key: &'static str,
    pub label: &'static str,
    pub help: &'static str,
    /// (argument value, shown label)
    pub options: &'static [(&'static str, &'static str)],
    pub picked: usize,
    pub starts_group: bool,
}

impl Field {
    pub fn value(&self) -> &'static str {
        self.options[self.picked].0
    }

    fn pick(&mut self, value: &str) {
        self.picked = self.options.iter().position(|(v, _)| *v == value).expect("known option");
    }
}

fn recipe_fields() -> Vec<Field> {
    let field = |key, label, help, options: &'static [(&'static str, &'static str)], default: &str, starts_group| {
        let mut field = Field { key, label, help, options, picked: 0, starts_group };
        field.pick(default);
        field
    };
    vec![
        field(
            "method",
            "Method",
            "LoRA trains small adapters. Full updates every weight. Zero shot only wraps and calibrates.",
            &[("lora", "LoRA"), ("full", "Full"), ("zero_shot", "Zero shot")],
            "lora",
            false,
        ),
        field(
            "attention",
            "Attention",
            "Bidirectional lifts the causal mask so every token sees the whole prompt.",
            &[("bidirectional", "Bidirectional"), ("set", "Set"), ("causal", "Causal")],
            "bidirectional",
            false,
        ),
        field(
            "primitive",
            "Decision",
            "The shape of the answer: one of several options, a yes or no gate, or a 1 to 9 score.",
            &[("choice", "Choice"), ("noul", "Yes or no"), ("score", "Score")],
            "choice",
            false,
        ),
        field(
            "epochs",
            "Epochs",
            "Passes over the training samples.",
            &[("1", "1"), ("2", "2"), ("3", "3"), ("5", "5")],
            "2",
            true,
        ),
        field(
            "lr",
            "Learning rate",
            "Peak rate of the cosine schedule. Full fine tuning wants a far smaller one than LoRA.",
            &[
                ("1e-5", "1e-5"),
                ("2e-5", "2e-5"),
                ("5e-5", "5e-5"),
                ("1e-4", "1e-4"),
                ("2e-4", "2e-4"),
                ("5e-4", "5e-4"),
            ],
            "2e-4",
            false,
        ),
        field(
            "batch_size",
            "Batch size",
            "Samples per forward pass. Gradients accumulate over four passes.",
            &[("1", "1"), ("2", "2"), ("4", "4"), ("8", "8"), ("16", "16")],
            "4",
            false,
        ),
        field(
            "max_seq_len",
            "Max tokens",
            "Longer prompts are trimmed from the start so the options always survive.",
            &[("256", "256"), ("512", "512"), ("1024", "1024"), ("2048", "2048")],
            "512",
            false,
        ),
        field(
            "max_train_samples",
            "Samples",
            "Upper bound on training samples drawn from the dataset.",
            &[("500", "500"), ("1000", "1,000"), ("2000", "2,000"), ("5000", "5,000"), ("10000", "10,000")],
            "2000",
            false,
        ),
        field(
            "calibration",
            "Calibration",
            "Fits a temperature on held out samples so confidence matches accuracy.",
            &[("on", "On"), ("off", "Off")],
            "on",
            true,
        ),
        field(
            "bench",
            "DecideBench",
            "Scores the finished model on the public decision benchmark.",
            &[("off", "Off"), ("v2", "v2"), ("v1", "v1"), ("both", "Both")],
            "v2",
            false,
        ),
        field(
            "bench_limit",
            "Bench items",
            "How many benchmark items to score. Fewer is faster and noisier.",
            &[("50", "50"), ("200", "200"), ("full", "All")],
            "50",
            false,
        ),
    ]
}

// ── A run in flight ─────────────────────────────────────────────────────────

pub struct Phase {
    pub name: String,
    pub detail: String,
}

#[derive(Deserialize, Clone, Copy)]
pub struct TrainStep {
    pub step: u64,
    pub total_steps: u64,
    pub epoch: u64,
    pub loss: f64,
    pub lr: f64,
}

pub struct Run {
    id: u64,
    pub title: String,
    pub started: Instant,
    pub phases: Vec<Phase>,
    /// Index of the phase in flight; `phases.len()` once all are done.
    pub active: usize,
    pub latest: Option<TrainStep>,
    pub first_step_at: Option<Instant>,
    pub losses: Vec<f64>,
    /// Progress bar fill, easing toward the true fraction.
    pub shown: f64,
    pub log: String,
    pub outcome: Option<Result<RunResult, String>>,
    pub took: Duration,
    pub cancel_armed: bool,
}

impl Run {
    pub fn fraction(&self) -> f64 {
        self.latest.map_or(0.0, |s| s.step as f64 / s.total_steps.max(1) as f64)
    }

    fn on_event(&mut self, event: &Value) {
        match event["event"].as_str() {
            Some("plan") => {
                let names = event["phases"].as_array().into_iter().flatten().filter_map(Value::as_str);
                self.phases = names.map(|name| Phase { name: name.into(), detail: String::new() }).collect();
            }
            Some("phase") => {
                let Some(index) = self.phases.iter().position(|p| p.name == event["name"]) else { return };
                self.active = index;
                if let Some(detail) = event["detail"].as_str() {
                    self.phases[index].detail = detail.into();
                }
            }
            Some("step") => {
                if let Ok(step) = TrainStep::deserialize(event) {
                    self.first_step_at.get_or_insert_with(Instant::now);
                    self.losses.push(step.loss);
                    self.latest = Some(step);
                }
            }
            _ => {}
        }
    }
}

// ── The app ─────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Step {
    Model,
    Data,
    Recipe,
    Launch,
    Run,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Hub,
    File,
    Mix,
    ZeroShot,
}

impl Source {
    pub const ALL: [(Source, &'static str); 4] = [
        (Source::Hub, "Hub dataset"),
        (Source::File, "Local file"),
        (Source::Mix, "General mix"),
        (Source::ZeroShot, "Zero shot"),
    ];
}

pub struct ChosenModel {
    pub repo_id: String,
    pub encoder: bool,
    pub params: Option<f64>,
    pub fits: Vec<Fit>,
}

pub struct App {
    bridge: Bridge,
    tx: Sender<Incoming>,
    rx: Receiver<Incoming>,
    hello_id: u64,
    pub opened: Instant,
    pub step: Step,
    pub entered_at: Instant,
    pub hello: Option<Hello>,
    pub models: Search<ModelHit>,
    pub model: Option<ChosenModel>,
    pub source: Source,
    pub datasets: Search<DatasetHit>,
    /// Preview of the highlighted Hub dataset, fetched once browsing pauses on it.
    pub preview: Lookup<Preview>,
    preview_repo: Option<String>,
    preview_due: Option<Instant>,
    pub file: Input,
    pub file_preview: Lookup<Preview>,
    pub fields: Vec<Field>,
    pub focus: usize,
    pub target: usize,
    pub run: Option<Run>,
    pub fatal: Option<String>,
    pub quit: bool,
}

impl App {
    pub fn new() -> io::Result<Self> {
        let (tx, rx) = channel();
        let mut bridge = Bridge::spawn(tx.clone())?;
        let hello_id = bridge.request("hello", json!({}));
        let mut app = Self {
            bridge,
            tx,
            rx,
            hello_id,
            opened: Instant::now(),
            step: Step::Model,
            entered_at: Instant::now(),
            hello: None,
            models: Search::default(),
            model: None,
            source: Source::Hub,
            datasets: Search::default(),
            preview: Lookup::default(),
            preview_repo: None,
            preview_due: None,
            file: Input::default(),
            file_preview: Lookup::default(),
            fields: recipe_fields(),
            focus: 0,
            target: 0,
            run: None,
            fatal: None,
            quit: false,
        };
        app.search_models();
        app.search_datasets();
        Ok(app)
    }

    fn search_models(&mut self) {
        let id = self.bridge.request("search_models", json!({ "query": self.models.input.text }));
        self.models.results.pending = Some(id);
    }

    fn search_datasets(&mut self) {
        let id = self.bridge.request("search_datasets", json!({ "query": self.datasets.input.text }));
        self.datasets.results.pending = Some(id);
    }

    fn request_preview(&mut self, config: Option<&str>) {
        let Some(repo) = &self.preview_repo else { return };
        let id = self.bridge.request("preview_dataset", json!({ "repo_id": repo, "config": config }));
        self.preview.pending = Some(id);
    }

    fn go(&mut self, step: Step) {
        self.step = step;
        self.entered_at = Instant::now();
    }

    // ── Time and backend messages ───────────────────────────────────────────

    pub fn tick(&mut self) {
        while let Ok(incoming) = self.rx.try_recv() {
            self.receive(incoming);
        }

        if self.models.input.settled() {
            self.search_models();
        }
        if self.datasets.input.settled() {
            self.search_datasets();
        }
        if self.file.settled() && !self.file.text.trim().is_empty() {
            let id = self.bridge.request("preview_file", json!({ "path": self.file.text.trim() }));
            self.file_preview.pending = Some(id);
        }

        let highlighted = self.datasets.current().map(|d| d.repo_id.clone());
        if highlighted != self.preview_repo {
            self.preview_repo = highlighted;
            self.preview = Lookup::default();
            self.preview_due = Some(Instant::now() + BROWSING_PAUSE);
        }
        if self.preview_due.is_some_and(|due| Instant::now() >= due) {
            self.preview_due = None;
            self.request_preview(None);
        }

        if let Some(run) = &mut self.run {
            run.shown += (run.fraction() - run.shown) * 0.18;
            if run.outcome.is_none() {
                run.took = run.started.elapsed();
            }
        }
    }

    fn receive(&mut self, incoming: Incoming) {
        match incoming {
            Incoming::Reply { id, result } => {
                if id == self.hello_id {
                    match result.and_then(|v| serde_json::from_value::<Hello>(v).map_err(|e| e.to_string())) {
                        Ok(hello) => {
                            self.target = hello.targets.iter().position(|t| t.ready).unwrap_or(0);
                            self.hello = Some(hello);
                        }
                        Err(error) => self.fatal = Some(format!("Could not read this machine's hardware. {error}")),
                    }
                } else if let Some(run) = self.run.as_mut().filter(|run| run.id == id) {
                    run.outcome = Some(result.and_then(|v| serde_json::from_value(v).map_err(|e| e.to_string())));
                    if run.outcome.as_ref().is_some_and(Result::is_ok) {
                        run.active = run.phases.len();
                    }
                } else {
                    let _ = self.models.accept(id, &result)
                        || self.datasets.accept(id, &result)
                        || self.preview.accept(id, &result)
                        || self.file_preview.accept(id, &result);
                }
            }
            Incoming::Event { id, event } => {
                if let Some(run) = self.run.as_mut().filter(|run| run.id == id) {
                    run.on_event(&event);
                }
            }
            Incoming::Log(line) => {
                if let Some(run) = &mut self.run {
                    run.log = line;
                }
            }
            Incoming::Closed { generation } => {
                if generation == self.bridge.generation {
                    self.fatal = Some("The Python backend stopped. Run decide doctor to check the install.".into());
                }
            }
        }
    }

    // ── Keys ────────────────────────────────────────────────────────────────

    pub fn key(&mut self, key: KeyEvent) {
        let ctrl_c = key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL);
        if ctrl_c || self.fatal.is_some() {
            self.quit = true;
            return;
        }
        match self.step {
            Step::Model => self.key_model(key),
            Step::Data => self.key_data(key),
            Step::Recipe => self.key_recipe(key),
            Step::Launch => self.key_launch(key),
            Step::Run => self.key_run(key),
        }
    }

    fn key_model(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up => self.models.step(false),
            KeyCode::Down => self.models.step(true),
            KeyCode::Esc => {
                self.models.input.text.clear();
                self.search_models();
            }
            // While a search is in flight the list still answers the previous query.
            KeyCode::Enter if self.models.busy() => {}
            KeyCode::Enter => {
                let typed = self.models.input.text.trim();
                self.model = match self.models.current() {
                    Some(hit) => Some(ChosenModel {
                        repo_id: hit.info.repo_id.clone(),
                        encoder: hit.info.architecture_family == "encoder",
                        params: hit.info.param_count_b,
                        fits: hit.fits.clone(),
                    }),
                    // Nothing on the Hub matched: take the text as a repo id or a local folder.
                    None if !typed.is_empty() => Some(ChosenModel {
                        repo_id: typed.to_string(),
                        encoder: typed.to_lowercase().contains("bert"),
                        params: None,
                        fits: Vec::new(),
                    }),
                    None => return,
                };
                self.go(Step::Data);
            }
            _ => {
                self.models.input.edit(&key);
            }
        }
    }

    fn key_data(&mut self, key: KeyEvent) {
        let sources = Source::ALL.map(|(source, _)| source);
        let at = sources.iter().position(|s| *s == self.source).expect("listed source");
        match key.code {
            KeyCode::Esc => self.go(Step::Model),
            KeyCode::Tab => self.source = sources[(at + 1) % sources.len()],
            KeyCode::BackTab => self.source = sources[(at + sources.len() - 1) % sources.len()],
            KeyCode::Enter => {
                let ready = match self.source {
                    Source::Hub => self.datasets.current().is_some() && !self.datasets.busy(),
                    Source::File => self.file_preview.value.is_some(),
                    Source::Mix | Source::ZeroShot => true,
                };
                if ready {
                    let first = self.visible_fields().next().unwrap_or(0);
                    self.focus = first;
                    self.go(Step::Recipe);
                }
            }
            _ => match self.source {
                Source::Hub => match key.code {
                    KeyCode::Up => self.datasets.step(false),
                    KeyCode::Down => self.datasets.step(true),
                    KeyCode::Left | KeyCode::Right => self.cycle_subset(key.code == KeyCode::Right),
                    _ => {
                        self.datasets.input.edit(&key);
                    }
                },
                Source::File => {
                    if self.file.edit(&key) {
                        self.file_preview = Lookup::default();
                    }
                }
                Source::Mix | Source::ZeroShot => {}
            },
        }
    }

    fn cycle_subset(&mut self, forward: bool) {
        let Some(preview) = &self.preview.value else { return };
        let count = preview.configs.len();
        let Some(at) = preview.configs.iter().position(|c| *c == preview.config) else { return };
        if count < 2 {
            return;
        }
        let next = preview.configs[(at + if forward { 1 } else { count - 1 }) % count].clone();
        self.request_preview(Some(&next));
    }

    /// Whether the run will skip weight updates.
    pub fn zero_shot(&self) -> bool {
        self.source == Source::ZeroShot || self.fields[0].value() == "zero_shot"
    }

    pub fn visible(&self, field: &Field) -> bool {
        let encoder = self.model.as_ref().is_some_and(|m| m.encoder);
        match field.key {
            "method" | "calibration" => self.source != Source::ZeroShot,
            "attention" => !encoder,
            "epochs" | "lr" | "batch_size" | "max_train_samples" => !self.zero_shot(),
            "bench_limit" => self.field("bench") != "off",
            _ => true,
        }
    }

    pub fn visible_fields(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.fields.len()).filter(|i| self.visible(&self.fields[*i]))
    }

    pub fn field(&self, key: &str) -> &'static str {
        self.fields.iter().find(|f| f.key == key).expect("known field").value()
    }

    fn key_recipe(&mut self, key: KeyEvent) {
        let visible: Vec<usize> = self.visible_fields().collect();
        let at = visible.iter().position(|i| *i == self.focus).unwrap_or(0);
        match key.code {
            KeyCode::Esc => self.go(Step::Data),
            KeyCode::Enter => self.go(Step::Launch),
            KeyCode::Up => self.focus = visible[at.saturating_sub(1)],
            KeyCode::Down => self.focus = visible[(at + 1).min(visible.len() - 1)],
            KeyCode::Left | KeyCode::Right => {
                let field = &mut self.fields[self.focus];
                let last = field.options.len() - 1;
                field.picked = if key.code == KeyCode::Right {
                    (field.picked + 1).min(last)
                } else {
                    field.picked.saturating_sub(1)
                };
                if field.key == "method" {
                    let rate = if field.value() == "full" { "2e-5" } else { "2e-4" };
                    self.fields.iter_mut().find(|f| f.key == "lr").expect("lr field").pick(rate);
                }
            }
            _ => {}
        }
    }

    fn key_launch(&mut self, key: KeyEvent) {
        let Some(hello) = &self.hello else { return };
        match key.code {
            KeyCode::Esc => self.go(Step::Recipe),
            KeyCode::Up => self.target = self.target.saturating_sub(1),
            KeyCode::Down => self.target = (self.target + 1).min(hello.targets.len() - 1),
            KeyCode::Enter if hello.targets[self.target].ready => {
                let target = &hello.targets[self.target];
                let model = self.model.as_ref().expect("model chosen before launch");
                let title = format!("{} on {}", short_name(&model.repo_id), target.label);
                let id = self.bridge.request("run", self.run_arguments(&target.key));
                self.run = Some(Run {
                    id,
                    title,
                    started: Instant::now(),
                    phases: Vec::new(),
                    active: 0,
                    latest: None,
                    first_step_at: None,
                    losses: Vec::new(),
                    shown: 0.0,
                    log: String::new(),
                    outcome: None,
                    took: Duration::ZERO,
                    cancel_armed: false,
                });
                self.go(Step::Run);
            }
            _ => {}
        }
    }

    /// The `decide train` arguments this session amounts to.
    fn run_arguments(&self, target: &str) -> Value {
        let mut args = Map::new();
        let mut set = |key: &str, value: Value| args.insert(key.to_string(), value);
        set("model", json!(self.model.as_ref().expect("model chosen before launch").repo_id));
        set("target", json!(target));
        match self.source {
            Source::Hub => {
                set("dataset", json!(self.preview_repo));
                if let Some(preview) = &self.preview.value {
                    set("split", json!(preview.split));
                    if preview.config != "default" {
                        set("dataset_config", json!(preview.config));
                    }
                }
            }
            Source::File => {
                let path = self.file_preview.value.as_ref().and_then(|p| p.path.clone());
                set("local_file", json!(path));
            }
            Source::Mix => {
                set("general_mix", json!(true));
            }
            Source::ZeroShot => {
                set("zero_shot", json!(true));
            }
        }
        for field in self.fields.iter().filter(|f| self.visible(f)) {
            let value = field.value();
            match field.key {
                "epochs" | "batch_size" | "max_seq_len" | "max_train_samples" => {
                    set(field.key, json!(value.parse::<u64>().expect("integer option")));
                }
                "lr" => {
                    set("lr", json!(value.parse::<f64>().expect("numeric option")));
                }
                "calibration" => {
                    set("no_calibration", json!(value == "off"));
                }
                "bench" => {
                    set("eval_decidebench", json!(value != "off"));
                    if value != "off" {
                        set("bench_version", json!(value));
                    }
                }
                "bench_limit" => {
                    set("bench_limit", json!(value.parse::<u64>().ok()));
                }
                _ => {
                    set(field.key, json!(value));
                }
            }
        }
        Value::Object(args)
    }

    fn key_run(&mut self, key: KeyEvent) {
        let Some(run) = &mut self.run else { return };
        if run.outcome.is_some() {
            match key.code {
                KeyCode::Enter => self.go(Step::Model),
                KeyCode::Esc => self.go(Step::Launch),
                KeyCode::Char('q') => self.quit = true,
                _ => {}
            }
        } else if key.code != KeyCode::Esc {
            run.cancel_armed = false;
        } else if !run.cancel_armed {
            run.cancel_armed = true;
        } else {
            // Training has no cooperative stop, so cancelling replaces the backend process.
            match Bridge::spawn(self.tx.clone()) {
                Ok(bridge) => self.bridge = bridge,
                Err(error) => self.fatal = Some(format!("Could not restart the Python backend. {error}")),
            }
            // Questions the old backend still owed an answer to will never be answered.
            self.models.results.pending = None;
            self.datasets.results.pending = None;
            self.preview.pending = None;
            self.file_preview.pending = None;
            self.run = None;
            self.go(Step::Launch);
        }
    }
}

/// "Qwen/Qwen3-0.6B" → "Qwen3-0.6B"; a local folder path → its last component.
pub fn short_name(repo_id: &str) -> &str {
    repo_id.trim_end_matches('/').rsplit('/').next().unwrap_or(repo_id)
}

/// "Qwen/Qwen3-0.6B" → "Qwen"; empty for a bare name.
pub fn owner(repo_id: &str) -> &str {
    repo_id.rsplit_once('/').map_or("", |(owner, _)| owner)
}
