//! The Python backend (`python -m decision_forge.bridge`), spoken to in JSON lines.
//! The protocol is documented at the top of `src/decision_forge/bridge.py`.

use serde_json::{json, Value};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::thread;

pub enum Incoming {
    Reply {
        id: u64,
        result: Result<Value, String>,
    },
    Event {
        id: u64,
        event: Value,
    },
    /// One line the backend wrote to stderr: library logs and download progress.
    Log(String),
    Closed {
        generation: u64,
    },
}

/// Request ids and bridge generations share one counter, so a reply from a
/// backend that was since replaced can never match a live request.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub struct Bridge {
    child: Child,
    stdin: ChildStdin,
    pub generation: u64,
}

impl Bridge {
    pub fn spawn(tx: Sender<Incoming>) -> io::Result<Self> {
        let python = std::env::var("DECIDE_PYTHON").unwrap_or_else(|_| "python3".into());
        let mut child = Command::new(python)
            .args(["-m", "decision_forge.bridge"])
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let generation = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let mut stderr = child.stderr.take().expect("piped stderr");

        let replies = tx.clone();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                let Ok(message) = serde_json::from_str::<Value>(&line) else { continue };
                let Some(id) = message["id"].as_u64() else { continue };
                let incoming = if message.get("event").is_some() {
                    Incoming::Event { id, event: message["event"].clone() }
                } else if message["ok"] == true {
                    Incoming::Reply { id, result: Ok(message["result"].clone()) }
                } else {
                    let error = message["error"].as_str().unwrap_or("unknown error").to_string();
                    Incoming::Reply { id, result: Err(error) }
                };
                if replies.send(incoming).is_err() {
                    return;
                }
            }
            let _ = replies.send(Incoming::Closed { generation });
        });

        // Progress bars redraw with a carriage return, so both \r and \n end a line.
        thread::spawn(move || {
            let mut chunk = [0u8; 4096];
            let mut partial = String::new();
            while let Ok(read) = stderr.read(&mut chunk) {
                if read == 0 {
                    break;
                }
                partial.push_str(&String::from_utf8_lossy(&chunk[..read]));
                while let Some(end) = partial.find(['\n', '\r']) {
                    let line = strip_ansi(partial[..end].trim());
                    partial.drain(..=end);
                    if !line.is_empty() && tx.send(Incoming::Log(line)).is_err() {
                        return;
                    }
                }
            }
        });

        Ok(Self { child, stdin, generation })
    }

    pub fn request(&mut self, method: &str, params: Value) -> u64 {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        // A dead backend is reported through `Incoming::Closed`, not through this write.
        let _ = writeln!(self.stdin, "{}", json!({ "id": id, "method": method, "params": params }));
        let _ = self.stdin.flush();
        id
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn strip_ansi(text: &str) -> String {
    let mut clean = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // CSI sequences end at the first byte in '@'..='~' after the '['.
            if chars.next() == Some('[') {
                for c in chars.by_ref() {
                    if ('@'..='~').contains(&c) {
                        break;
                    }
                }
            }
        } else if !c.is_control() {
            clean.push(c);
        }
    }
    clean
}
