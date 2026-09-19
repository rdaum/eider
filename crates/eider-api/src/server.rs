//! Axum HTTP and SSE surface for the OpenAI-compatible APIs.

use crate::actor::{ActorRequestId, InferenceActor};
use crate::chat_completions::{ChatCompletionRequest, ChatCompletionStream};
use crate::decisions::{DecisionApiRequest, DecisionApiResponse, DecisionCalibration};
use crate::metrics::{ServerEndpoint, StreamingMode, metrics as server_metrics};
use crate::protocol::{ApiError, ErrorEnvelope, ResponseRequest, ResponseStream};
use axum::Json;
use axum::Router;
use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use eider_inference::metrics::metrics as infer_metrics;
use serde_json::{Value, json};
use std::convert::Infallible;
use std::future::Future;
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;

const MIN_REQUEST_BODY_BYTES: usize = 2 * 1024 * 1024;
const REQUEST_BODY_BYTES_PER_CONTEXT_TOKEN: usize = 32;
const DECISION_MAX_LOGICAL_INPUT_TOKENS: usize = 64 * 1024;
const DECISION_MAX_BRANCH_INPUT_TOKENS: usize = 32 * 1024;
const DECISION_TIMEOUT: Duration = Duration::from_secs(120);

/// HTTP-facing configuration independent of model execution limits.
#[derive(Clone, Debug)]
pub struct ApiConfig {
    pub listen: SocketAddr,
    pub model: String,
    pub decision_model: String,
    pub decision_calibration: Option<DecisionCalibration>,
    pub bearer_token: Option<String>,
    pub context_window: usize,
}

impl ApiConfig {
    pub fn local(model: impl Into<String>) -> Self {
        Self {
            listen: SocketAddr::from(([127, 0, 0, 1], 8080)),
            model: model.into(),
            decision_model: String::new(),
            decision_calibration: None,
            bearer_token: None,
            context_window: 32_768,
        }
        .with_default_decision_model()
    }

    fn with_default_decision_model(mut self) -> Self {
        if self.decision_model.is_empty() {
            self.decision_model.clone_from(&self.model);
        }
        self
    }
}

#[derive(Clone)]
struct ApiState {
    actor: InferenceActor,
    config: ApiConfig,
}

/// Serves until the listener fails or the process is shut down.
pub async fn serve(actor: InferenceActor, config: ApiConfig) -> Result<(), std::io::Error> {
    let listener = TcpListener::bind(config.listen).await?;
    serve_listener(listener, actor, config).await
}

/// Serves until the listener fails or `shutdown` resolves.
pub async fn serve_with_shutdown(
    actor: InferenceActor,
    config: ApiConfig,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), std::io::Error> {
    let listener = TcpListener::bind(config.listen).await?;
    axum::serve(listener, router(actor, config))
        .with_graceful_shutdown(shutdown)
        .await
}

/// Serves an already-bound listener, allowing tests to use an ephemeral port.
pub async fn serve_listener(
    listener: TcpListener,
    actor: InferenceActor,
    config: ApiConfig,
) -> Result<(), std::io::Error> {
    axum::serve(listener, router(actor, config)).await
}

fn router(actor: InferenceActor, config: ApiConfig) -> Router {
    let request_body_limit = request_body_limit(config.context_window);
    let decision_capable = actor.capabilities().decisions;
    let mut router = Router::new()
        .route("/healthz", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(models))
        .route(
            "/v1/responses",
            post(responses).layer(DefaultBodyLimit::max(request_body_limit)),
        )
        .route(
            "/v1/chat/completions",
            post(chat_completions).layer(DefaultBodyLimit::max(request_body_limit)),
        );
    if decision_capable {
        router = router.route(
            "/v1/decisions",
            post(decisions).layer(DefaultBodyLimit::max(request_body_limit)),
        );
    }
    router.with_state(ApiState { actor, config })
}

fn request_body_limit(context_window: usize) -> usize {
    context_window
        .saturating_mul(REQUEST_BODY_BYTES_PER_CONTEXT_TOKEN)
        .max(MIN_REQUEST_BODY_BYTES)
}

async fn health() -> Json<Value> {
    server_metrics().requests.inc(ServerEndpoint::Healthz);
    let _request_duration = RequestDuration::start();
    Json(json!({"status": "ok"}))
}

/// Prometheus text exposition endpoint.
///
/// Always enabled. Returns all registered metrics from the eider-api and infer
/// crates in Prometheus text format (`text/plain; version=0.0.4`).
async fn metrics() -> Response {
    server_metrics().requests.inc(ServerEndpoint::Metrics);
    let _request_duration = RequestDuration::start();
    let mut output = String::new();
    server_metrics().export_prometheus(&mut output);
    infer_metrics().export_prometheus(&mut output);
    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        output,
    )
        .into_response()
}

async fn models(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiFailure> {
    server_metrics().requests.inc(ServerEndpoint::Models);
    let _request_duration = RequestDuration::start();
    authorise(&state.config, &headers)?;
    let model = &state.config.model;
    let context_window = state.config.context_window;
    let auto_compact_token_limit = context_window.saturating_mul(9) / 10;
    let openai_model = json!({
        "id": model,
        "object": "model",
        "created": 0,
        "owned_by": "eider"
    });
    let codex_model = json!({
            "slug": model,
            "display_name": model,
            "description": "Eider local model",
            "default_reasoning_level": "none",
            "supported_reasoning_levels": [{"effort": "none", "description": "Model default"}],
            "shell_type": "unified_exec",
            "visibility": "list",
            "supported_in_api": true,
            "priority": 1,
            "additional_speed_tiers": [],
            "service_tiers": [],
            "default_service_tier": null,
            "availability_nux": null,
            "upgrade": null,
            "base_instructions": "You are a coding agent. Work carefully in the supplied repository, use the available tools when needed, and report the completed result concisely.",
            "model_messages": null,
            "include_skills_usage_instructions": false,
            "supports_reasoning_summary_parameter": false,
            "default_reasoning_summary": "none",
            "support_verbosity": false,
            "default_verbosity": null,
            "apply_patch_tool_type": null,
            "web_search_tool_type": "text",
            "truncation_policy": {"mode": "tokens", "limit": context_window},
            "supports_parallel_tool_calls": false,
            "supports_image_detail_original": false,
            "context_window": context_window,
            "max_context_window": context_window,
            "auto_compact_token_limit": auto_compact_token_limit,
            "comp_hash": null,
            "effective_context_window_percent": 95,
            "experimental_supported_tools": [],
            "input_modalities": ["text"],
            "supports_search_tool": false,
            "use_responses_lite": false,
            "auto_review_model_override": null,
            "tool_mode": null,
            "multi_agent_version": null
    });
    Ok(Json(json!({
        "object": "list",
        "data": [openai_model],
        "models": [codex_model]
    })))
}

async fn responses(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<ResponseRequest>,
) -> Result<Response, ApiFailure> {
    server_metrics().requests.inc(ServerEndpoint::Responses);
    let _request_duration = RequestDuration::start();
    authorise(&state.config, &headers)?;
    if request.model != state.config.model {
        server_metrics()
            .request_errors
            .inc(ServerEndpoint::Responses);
        return Err(ApiFailure::bad_request(ApiError::invalid(
            "model",
            format!(
                "model {:?} is not served; expected {:?}",
                request.model, state.config.model
            ),
        )));
    }
    let stream_requested = request.stream;
    let model = request.model.clone();
    let chat = request
        .into_chat_request(state.actor.generation_defaults())
        .map_err(|e| {
            server_metrics()
                .request_errors
                .inc(ServerEndpoint::Responses);
            ApiFailure::bad_request(e)
        })?;
    let response = state.actor.submit(chat).map_err(|e| {
        server_metrics()
            .request_errors
            .inc(ServerEndpoint::Responses);
        ApiFailure::server(e)
    })?;
    server_metrics()
        .responses_submitted
        .inc(if stream_requested {
            StreamingMode::Stream
        } else {
            StreamingMode::NonStream
        });
    let stream = ResponseStream::new(model);

    if stream_requested {
        return Ok(streaming_response(
            state.actor,
            response.id,
            response.events,
            stream,
        ));
    }
    non_streaming_response(state.actor, response.id, response.events, stream).await
}

async fn chat_completions(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiFailure> {
    server_metrics()
        .requests
        .inc(ServerEndpoint::ChatCompletions);
    let _request_duration = RequestDuration::start();
    authorise(&state.config, &headers)?;
    if request.model != state.config.model {
        server_metrics()
            .request_errors
            .inc(ServerEndpoint::ChatCompletions);
        return Err(ApiFailure::bad_request(ApiError::invalid(
            "model",
            format!(
                "model {:?} is not served; expected {:?}",
                request.model, state.config.model
            ),
        )));
    }
    let stream_requested = request.stream;
    let include_usage = request.include_usage();
    let model = request.model.clone();
    let chat = request
        .into_chat_request(state.actor.generation_defaults())
        .map_err(|error| {
            server_metrics()
                .request_errors
                .inc(ServerEndpoint::ChatCompletions);
            ApiFailure::bad_request(error)
        })?;
    let response = state.actor.submit(chat).map_err(|error| {
        server_metrics()
            .request_errors
            .inc(ServerEndpoint::ChatCompletions);
        ApiFailure::server(error)
    })?;
    server_metrics()
        .chat_completions_submitted
        .inc(if stream_requested {
            StreamingMode::Stream
        } else {
            StreamingMode::NonStream
        });
    let stream = ChatCompletionStream::new(model, include_usage);

    if stream_requested {
        return Ok(streaming_chat_completion(
            state.actor,
            response.id,
            response.events,
            stream,
        ));
    }
    non_streaming_chat_completion(state.actor, response.id, response.events, stream).await
}

async fn decisions(
    State(state): State<ApiState>,
    headers: HeaderMap,
    request: Result<Json<DecisionApiRequest>, JsonRejection>,
) -> Result<Response, ApiFailure> {
    server_metrics().requests.inc(ServerEndpoint::Decisions);
    let _request_duration = RequestDuration::start();
    authorise(&state.config, &headers)?;
    let Json(request) = request
        .map_err(|error| ApiFailure::unprocessable(ApiError::invalid("body", error.body_text())))?;
    if request.model != state.config.model && request.model != "decision-latest" {
        server_metrics()
            .request_errors
            .inc(ServerEndpoint::Decisions);
        return Err(ApiFailure::unprocessable(ApiError::invalid(
            "model",
            format!(
                "model {:?} is not served; expected {:?} or \"decision-latest\"",
                request.model, state.config.model
            ),
        )));
    }
    let include_raw_logits = request.include_raw_logits;
    let prompt_started = Instant::now();
    let prompt = request
        .into_prompt_request()
        .map_err(ApiFailure::unprocessable)?;
    let decision = state
        .actor
        .prepare_decision(prompt)
        .map_err(ApiFailure::unprocessable)?;
    let prompt_preparation = prompt_started.elapsed();
    server_metrics()
        .decision_prompt_preparation_us
        .record(duration_us(prompt_preparation));
    let logical_tokens = decision.logical_input_tokens();
    if logical_tokens > DECISION_MAX_LOGICAL_INPUT_TOKENS {
        return Err(ApiFailure::unprocessable(ApiError::invalid(
            "questions",
            format!(
                "decision input has {logical_tokens} logical tokens; the limit is {DECISION_MAX_LOGICAL_INPUT_TOKENS}"
            ),
        )));
    }
    let longest_branch = decision.longest_branch_tokens();
    let branch_limit = DECISION_MAX_BRANCH_INPUT_TOKENS.min(state.config.context_window);
    if longest_branch > branch_limit {
        return Err(ApiFailure::unprocessable(ApiError::invalid(
            "state",
            format!(
                "longest decision branch has {longest_branch} tokens; the limit is {branch_limit}"
            ),
        )));
    }
    let branches = decision.branches.len();
    let shared_prefix_tokens = decision.prefix_tokens.len();
    let branch_input_tokens = decision
        .branches
        .iter()
        .map(|branch| branch.suffix_tokens.len())
        .sum::<usize>();
    let response = state
        .actor
        .submit_decision(decision)
        .map_err(ApiFailure::server)?;
    server_metrics().decision_requests_submitted.inc();
    server_metrics().decision_branches.add(branches as isize);
    server_metrics()
        .decision_shared_prefix_tokens
        .add(shared_prefix_tokens as isize);
    server_metrics()
        .decision_branch_input_tokens
        .add(branch_input_tokens as isize);
    let mut cancellation = CancellationGuard::new(state.actor, response.id);
    let completion = tokio::time::timeout(DECISION_TIMEOUT, response.completion)
        .await
        .map_err(|_| {
            ApiFailure::gateway_timeout(ApiError::server(
                "decision request exceeded the 120 second timeout",
            ))
        })?
        .map_err(|_| {
            ApiFailure::server(ApiError::server(
                "inference actor closed the decision before completion",
            ))
        })?
        .map_err(|message| ApiFailure::server(ApiError::server(message)))?;
    cancellation.disarm();
    let timings = completion.timings;
    server_metrics()
        .decision_shared_prefill_us
        .record(duration_us(timings.shared_prefill));
    server_metrics()
        .decision_branch_fork_us
        .record(duration_us(timings.branch_fork));
    server_metrics()
        .decision_branch_inference_us
        .record(duration_us(timings.branch_inference));
    server_metrics()
        .decision_sequence_device_bytes_released
        .add(completion.released_sequence_device_bytes as isize);
    let response = DecisionApiResponse::from_completion(
        state.config.decision_model,
        completion,
        state.config.decision_calibration.as_ref(),
        include_raw_logits,
    )
    .map_err(ApiFailure::server)?;
    let server_timing = format!(
        "prompt;dur={:.3}, shared_prefill;dur={:.3}, fork;dur={:.3}, branches;dur={:.3}",
        duration_ms(prompt_preparation),
        duration_ms(timings.shared_prefill),
        duration_ms(timings.branch_fork),
        duration_ms(timings.branch_inference),
    );
    let mut response = Json(response).into_response();
    response.headers_mut().insert(
        HeaderName::from_static("server-timing"),
        HeaderValue::from_str(&server_timing)
            .expect("decision server timing contains only finite decimal values"),
    );
    Ok(response)
}

fn streaming_response(
    actor: InferenceActor,
    request_id: ActorRequestId,
    mut events: tokio::sync::mpsc::Receiver<crate::protocol::InferenceEvent>,
    mut response: ResponseStream,
) -> Response {
    let output = async_stream::stream! {
        let mut cancellation = CancellationGuard::new(actor, request_id);
        yield Ok::<_, Infallible>(sse_event(response.created()));
        while let Some(inference) = events.recv().await {
            for event in response.push(inference) {
                yield Ok(sse_event(event));
            }
            if response.is_completed() {
                cancellation.disarm();
                break;
            }
        }
    };
    Sse::new(output)
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response()
}

fn streaming_chat_completion(
    actor: InferenceActor,
    request_id: ActorRequestId,
    mut events: tokio::sync::mpsc::Receiver<crate::protocol::InferenceEvent>,
    mut response: ChatCompletionStream,
) -> Response {
    let output = async_stream::stream! {
        let mut cancellation = CancellationGuard::new(actor, request_id);
        yield Ok::<_, Infallible>(chat_sse_event(response.initial_chunk()));
        while let Some(inference) = events.recv().await {
            for chunk in response.push(inference) {
                yield Ok(chat_sse_event(chunk));
            }
            if response.is_completed() {
                cancellation.disarm();
                yield Ok(chat_sse_done());
                break;
            }
        }
    };
    Sse::new(output)
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response()
}

async fn non_streaming_response(
    actor: InferenceActor,
    request_id: ActorRequestId,
    mut events: tokio::sync::mpsc::Receiver<crate::protocol::InferenceEvent>,
    mut response: ResponseStream,
) -> Result<Response, ApiFailure> {
    let mut cancellation = CancellationGuard::new(actor, request_id);
    while let Some(inference) = events.recv().await {
        for event in response.push(inference) {
            match event["type"].as_str() {
                Some("response.completed" | "response.incomplete") => {
                    cancellation.disarm();
                    return Ok(Json(event["response"].clone()).into_response());
                }
                Some("error") => {
                    cancellation.disarm();
                    let message = event["error"]["message"]
                        .as_str()
                        .unwrap_or("inference failed");
                    return Err(ApiFailure::server(ApiError::server(message)));
                }
                _ => {}
            }
        }
    }
    Err(ApiFailure::server(ApiError::server(
        "inference actor closed the response before completion",
    )))
}

async fn non_streaming_chat_completion(
    actor: InferenceActor,
    request_id: ActorRequestId,
    mut events: tokio::sync::mpsc::Receiver<crate::protocol::InferenceEvent>,
    mut response: ChatCompletionStream,
) -> Result<Response, ApiFailure> {
    let mut cancellation = CancellationGuard::new(actor, request_id);
    while let Some(inference) = events.recv().await {
        response.push(inference);
        if let Some(message) = response.error() {
            let message = message.to_string();
            cancellation.disarm();
            return Err(ApiFailure::server(ApiError::server(message)));
        }
        if let Some(completion) = response.response() {
            cancellation.disarm();
            return Ok(Json(completion).into_response());
        }
    }
    Err(ApiFailure::server(ApiError::server(
        "inference actor closed the chat completion before completion",
    )))
}

fn sse_event(value: Value) -> Event {
    let kind = value["type"].as_str().unwrap_or("error").to_string();
    Event::default()
        .event(kind)
        .json_data(value)
        .expect("Responses event is valid JSON")
}

fn chat_sse_event(value: Value) -> Event {
    Event::default()
        .json_data(value)
        .expect("Chat Completions chunk is valid JSON")
}

fn chat_sse_done() -> Event {
    Event::default().data("[DONE]")
}

fn authorise(config: &ApiConfig, headers: &HeaderMap) -> Result<(), ApiFailure> {
    let Some(expected) = config.bearer_token.as_deref() else {
        return Ok(());
    };
    let supplied = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    if supplied == Some(expected) {
        return Ok(());
    }
    Err(ApiFailure {
        status: StatusCode::UNAUTHORIZED,
        error: ApiError {
            message: "invalid bearer token".to_string(),
            kind: "authentication_error",
            param: None,
            code: Some("invalid_api_key".to_string()),
        },
    })
}

struct CancellationGuard {
    actor: Option<InferenceActor>,
    request_id: ActorRequestId,
}

impl CancellationGuard {
    fn new(actor: InferenceActor, request_id: ActorRequestId) -> Self {
        Self {
            actor: Some(actor),
            request_id,
        }
    }

    fn disarm(&mut self) {
        self.actor = None;
    }
}

impl Drop for CancellationGuard {
    fn drop(&mut self) {
        if let Some(actor) = &self.actor {
            actor.cancel(self.request_id);
        }
    }
}

pub struct ApiFailure {
    status: StatusCode,
    error: ApiError,
}

impl ApiFailure {
    fn bad_request(error: ApiError) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error,
        }
    }

    fn server(error: ApiError) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            error,
        }
    }

    fn unprocessable(error: ApiError) -> Self {
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            error,
        }
    }

    fn gateway_timeout(error: ApiError) -> Self {
        Self {
            status: StatusCode::GATEWAY_TIMEOUT,
            error,
        }
    }
}

impl IntoResponse for ApiFailure {
    fn into_response(self) -> Response {
        (self.status, Json(ErrorEnvelope { error: self.error })).into_response()
    }
}

struct RequestDuration {
    started: Instant,
}

impl RequestDuration {
    fn start() -> Self {
        Self {
            started: Instant::now(),
        }
    }
}

impl Drop for RequestDuration {
    fn drop(&mut self) {
        server_metrics()
            .request_duration_us
            .record(duration_us(self.started.elapsed()));
    }
}

fn duration_us(elapsed: Duration) -> u64 {
    elapsed.as_micros().min(u128::from(u64::MAX)) as u64
}

fn duration_ms(elapsed: Duration) -> f64 {
    elapsed.as_secs_f64() * 1_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_body_limit_scales_with_context_window() {
        assert_eq!(request_body_limit(32_768), 2 * 1024 * 1024);
        assert_eq!(request_body_limit(131_072), 4 * 1024 * 1024);
        assert_eq!(request_body_limit(262_144), 8 * 1024 * 1024);
    }
}
