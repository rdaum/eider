use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
#[ignore = "loads the full local checkpoint and runs the installed Codex CLI"]
fn codex_completes_a_turn_through_live_responses_server() {
    let model_dir = std::env::var_os("QWEN36_MODEL")
        .map(PathBuf::from)
        .map(|path| {
            if path.is_absolute() {
                path
            } else {
                workspace_dir().join(path)
            }
        })
        .unwrap_or_else(default_model_dir);
    assert!(
        model_dir.is_dir(),
        "model directory is missing: {}",
        model_dir.display()
    );
    Command::new("codex")
        .arg("--version")
        .stdout(Stdio::null())
        .status()
        .expect("installed codex CLI");

    let listen = reserve_address();
    let api_key = "eider-integration-test";
    let child = Command::new(env!("CARGO_BIN_EXE_eider-serve"))
        .arg(&model_dir)
        .args(["--listen", &listen.to_string()])
        .args(["--served-model-name", "eider-test"])
        .args(["--decode-capacity", "1"])
        .args(["--prefill-sequence-capacity", "1"])
        .args(["--max-active-sequences", "1"])
        .args(["--max-context-tokens", "32768"])
        .env("EIDER_API_KEY", api_key)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("start Eider server");
    let _server = Server(child);
    wait_until_ready(listen, Duration::from_secs(180));

    let work_dir = temporary_directory();
    std::fs::create_dir_all(&work_dir).expect("create temporary Codex work directory");
    let base_url = format!("http://{listen}/v1");
    let output = Command::new("codex")
        .args([
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            "-C",
        ])
        .arg(&work_dir)
        .args(["-m", "eider-test"])
        .args(["-c", "model_provider=\"eider\""])
        .args(["-c", "model_providers.eider.name=\"Eider\""])
        .arg("-c")
        .arg(format!("model_providers.eider.base_url=\"{base_url}\""))
        .args(["-c", "model_providers.eider.env_key=\"EIDER_API_KEY\""])
        .args(["-c", "model_providers.eider.wire_api=\"responses\""])
        .args(["-c", "model_providers.eider.request_max_retries=0"])
        .args(["-c", "model_providers.eider.stream_max_retries=0"])
        .arg(
            "Use exec_command to run `printf EIDER_TOOL_OK > proof.txt`, then reply with exactly EIDER_CODEX_OK.",
        )
        .env("EIDER_API_KEY", api_key)
        .stdin(Stdio::null())
        .output()
        .expect("run Codex against Eider");
    assert!(
        output.status.success(),
        "Codex failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("EIDER_CODEX_OK"),
        "Codex did not receive the expected response:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        std::fs::read_to_string(work_dir.join("proof.txt")).expect("Codex-created proof file"),
        "EIDER_TOOL_OK"
    );
    let _ = std::fs::remove_dir_all(&work_dir);
}

fn default_model_dir() -> PathBuf {
    workspace_dir().join("models/qwen3.6-35b-a3-nvfp4")
}

fn workspace_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn reserve_address() -> SocketAddr {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("reserve local port");
    listener.local_addr().expect("reserved address")
}

fn wait_until_ready(address: SocketAddr, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_secs(1)) {
            let _ = stream.write_all(
                b"GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
            );
            let mut response = String::new();
            if stream.read_to_string(&mut response).is_ok() && response.contains("200 OK") {
                return;
            }
        }
        thread::sleep(Duration::from_millis(250));
    }
    panic!("Eider server did not become ready within {timeout:?}");
}

fn temporary_directory() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("eider-codex-{nonce}"))
}
