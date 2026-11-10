use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use axum::extract::{DefaultBodyLimit, Path, State, rejection::JsonRejection};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::{StreamExt, wrappers::ReceiverStream};

use crate::generation::{Engine, Finish, Generation, Message, Sampling, THINK_MARKERS};

const REQUEST_BODY_BYTES: usize = 1024 * 1024;
const STREAM_BUFFER_EVENTS: usize = 16;
const PENDING_JOBS: usize = 4;
const DEFAULT_OUTPUT_TOKENS: usize = 1024;

/// The caller validates `bind` and `model_id` before loading the model.
pub async fn serve(engine: Engine, bind: SocketAddr, model_id: String) -> Result<()> {
    let listener = TcpListener::bind(bind)
        .await
        .with_context(|| format!("bind {bind}"))?;
    let (jobs, receiver) = mpsc::channel(PENDING_JOBS);
    std::thread::Builder::new()
        .name("oryndral-inference".into())
        .spawn(move || worker(engine, receiver))
        .context("start inference worker")?;

    let address = listener.local_addr()?;
    eprintln!("Oryndral serving {model_id} at http://{address}/v1");
    eprintln!("Open browser chat at http://{address}/");
    axum::serve(listener, router(model_id, jobs))
        .with_graceful_shutdown(async {
            if let Err(error) = tokio::signal::ctrl_c().await {
                eprintln!("Failed to listen for Ctrl+C: {error}");
            }
        })
        .await
        .context("serve HTTP")
}

#[derive(Clone)]
struct App {
    model: Arc<str>,
    created: u64,
    sequence: Arc<AtomicU64>,
    jobs: mpsc::Sender<Job>,
}

fn router(model: String, jobs: mpsc::Sender<Job>) -> Router {
    Router::new()
        .route("/", get(|| async { Html(include_str!("web_chat.html")) }))
        .route("/v1", get(open_chat))
        .route("/v1/", get(open_chat))
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/models/{id}", get(retrieve_model))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(text_completion))
        .fallback(|| async {
            ApiError::new(
                StatusCode::NOT_FOUND,
                "route_not_found",
                "Endpoint not found",
            )
        })
        .method_not_allowed_fallback(|| async {
            ApiError::new(
                StatusCode::METHOD_NOT_ALLOWED,
                "method_not_allowed",
                "Method not allowed",
            )
        })
        .layer(DefaultBodyLimit::max(REQUEST_BODY_BYTES))
        .with_state(App {
            model: model.into(),
            created: unix_seconds(),
            sequence: Arc::new(AtomicU64::new(1)),
            jobs,
        })
}

impl App {
    fn ensure_ready(&self) -> Result<(), ApiError> {
        if self.jobs.is_closed() {
            return Err(ApiError::unavailable());
        }
        Ok(())
    }

    fn check_model(&self, id: Option<&str>) -> Result<(), ApiError> {
        if id.is_some_and(|id| id != self.model.as_ref()) {
            return Err(ApiError::new(
                StatusCode::NOT_FOUND,
                "model_not_found",
                "Requested model is not loaded",
            )
            .param("model"));
        }
        Ok(())
    }

    fn model_info(&self) -> Value {
        json!({"id": self.model.as_ref(), "object": "model", "created": self.created,
            "owned_by": "oryndral"})
    }
}

async fn open_chat() -> Redirect {
    Redirect::temporary("/")
}

async fn health(State(app): State<App>) -> Result<Json<Value>, ApiError> {
    app.ensure_ready()?;
    Ok(Json(json!({"status": "ok"})))
}

async fn models(State(app): State<App>) -> Result<Json<Value>, ApiError> {
    app.ensure_ready()?;
    Ok(Json(json!({"object": "list", "data": [app.model_info()]})))
}

async fn retrieve_model(
    State(app): State<App>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    app.ensure_ready()?;
    app.check_model(Some(&id))?;
    Ok(Json(app.model_info()))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompletionRequest {
    model: Option<String>,
    messages: Option<Vec<ChatMessage>>,
    prompt: Option<String>,
    #[serde(alias = "max_completion_tokens")]
    max_tokens: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    seed: Option<i64>,
    stop: Option<Stop>,
    stream: Option<bool>,
    stream_options: Option<StreamOptions>,
    n: Option<usize>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    logprobs: Option<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatMessage {
    role: String,
    content: Content,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Parts(Vec<Part>),
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum Part {
    Text { text: String },
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Stop {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamOptions {
    include_usage: Option<bool>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Chat,
    Text,
}

enum Input {
    Messages(Vec<Message>),
    Prompt(String),
}

impl CompletionRequest {
    fn sampling(&mut self, kind: Kind) -> Result<Sampling, ApiError> {
        if self.n.is_some_and(|n| n != 1) {
            return Err(ApiError::invalid("n must be 1").param("n"));
        }
        if self.frequency_penalty.unwrap_or(0.0) != 0.0
            || self.presence_penalty.unwrap_or(0.0) != 0.0
        {
            return Err(ApiError::invalid(
                "nonzero frequency and presence penalties are unsupported",
            ));
        }
        if self
            .logprobs
            .as_ref()
            .is_some_and(|value| kind == Kind::Text || value != &Value::Bool(false))
        {
            return Err(ApiError::invalid("logprobs are unsupported").param("logprobs"));
        }
        if self.stream_options.is_some() && !self.stream.unwrap_or(false) {
            return Err(
                ApiError::invalid("stream_options requires stream=true").param("stream_options")
            );
        }

        let defaults = Sampling::default();
        let stop = match self.stop.take() {
            Some(Stop::One(stop)) => vec![stop],
            Some(Stop::Many(stops)) => stops,
            None => defaults.stop,
        };
        let sampling = Sampling {
            max_tokens: self.max_tokens.unwrap_or(DEFAULT_OUTPUT_TOKENS),
            temperature: self.temperature.unwrap_or(defaults.temperature),
            top_p: self.top_p.unwrap_or(defaults.top_p),
            top_k: self.top_k.or(defaults.top_k),
            // Like the OpenAI API, a request without a seed gets a fresh one.
            seed: self.seed.map_or_else(rand::random, |seed| seed as u64),
            stop,
        };
        sampling
            .validate()
            .map_err(|error| ApiError::invalid(error.to_string()))?;
        Ok(sampling)
    }

    fn into_input(self, kind: Kind) -> Result<Input, ApiError> {
        if kind == Kind::Text {
            if self.messages.is_some() {
                return Err(ApiError::invalid(
                    "text completions require prompt instead of messages",
                )
                .param("messages"));
            }
            return self
                .prompt
                .map(Input::Prompt)
                .ok_or_else(|| ApiError::invalid("a string prompt is required").param("prompt"));
        }
        if self.prompt.is_some() {
            return Err(
                ApiError::invalid("chat completions require messages instead of prompt")
                    .param("prompt"),
            );
        }
        let messages = self
            .messages
            .filter(|messages| !messages.is_empty())
            .ok_or_else(|| ApiError::invalid("messages cannot be empty").param("messages"))?;
        messages
            .into_iter()
            .map(|message| {
                if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
                    return Err(ApiError::invalid(
                        "message roles must be system, user, or assistant",
                    )
                    .param("messages"));
                }
                let content = match message.content {
                    Content::Text(text) => text,
                    Content::Parts(parts) => {
                        parts.into_iter().map(|Part::Text { text }| text).collect()
                    }
                };
                Ok(Message {
                    role: message.role,
                    content,
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Input::Messages)
    }
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    code: &'static str,
    param: Option<&'static str>,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            param: None,
        }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request", message)
    }

    fn unavailable() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_unavailable",
            "Inference worker unavailable",
        )
    }

    fn inference(error: anyhow::Error) -> Self {
        eprintln!("Inference failed: {error:#}");
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "inference_error",
            "Model generation failed",
        )
    }

    fn param(mut self, param: &'static str) -> Self {
        self.param = Some(param);
        self
    }

    fn body(&self) -> Value {
        let kind = match self.status {
            StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
            status if status.is_server_error() => "server_error",
            _ => "invalid_request_error",
        };
        json!({"error": {"message": self.message, "type": kind, "param": self.param, "code": self.code}})
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body())).into_response()
    }
}

struct Job {
    input: Input,
    sampling: Sampling,
    stream: bool,
    include_usage: bool,
    completion: Completion,
    reply: oneshot::Sender<Result<Reply, ApiError>>,
}

enum Reply {
    Full(Value),
    Stream(mpsc::Receiver<Event>),
}

async fn chat(
    State(app): State<App>,
    request: Result<Json<CompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    submit(app, request, Kind::Chat).await
}

async fn text_completion(
    State(app): State<App>,
    request: Result<Json<CompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    submit(app, request, Kind::Text).await
}

async fn submit(
    app: App,
    request: Result<Json<CompletionRequest>, JsonRejection>,
    kind: Kind,
) -> Result<Response, ApiError> {
    let Json(mut request) = request.map_err(|error| {
        let status = match error.status() {
            StatusCode::PAYLOAD_TOO_LARGE => StatusCode::PAYLOAD_TOO_LARGE,
            _ => StatusCode::BAD_REQUEST,
        };
        ApiError::new(status, "invalid_json", error.body_text())
    })?;
    app.check_model(request.model.as_deref())?;
    let sampling = request.sampling(kind)?;
    let stream = request.stream.unwrap_or(false);
    let include_usage = request
        .stream_options
        .as_ref()
        .and_then(|options| options.include_usage)
        .unwrap_or(false);
    let input = request.into_input(kind)?;
    let sequence = app.sequence.fetch_add(1, Ordering::Relaxed);
    let prefix = match kind {
        Kind::Chat => "chatcmpl",
        Kind::Text => "cmpl",
    };
    let (reply, response) = oneshot::channel();
    app.jobs
        .try_send(Job {
            input,
            sampling,
            stream,
            include_usage,
            completion: Completion {
                id: format!("{prefix}-{}-{sequence}", std::process::id()),
                created: unix_seconds(),
                model: app.model,
                kind,
            },
            reply,
        })
        .map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => ApiError::new(
                StatusCode::TOO_MANY_REQUESTS,
                "model_busy",
                "The inference queue is full",
            ),
            mpsc::error::TrySendError::Closed(_) => ApiError::unavailable(),
        })?;

    match response.await.map_err(|_| ApiError::unavailable())?? {
        Reply::Full(value) => Ok(Json(value).into_response()),
        Reply::Stream(events) => Ok(
            Sse::new(ReceiverStream::new(events).map(Ok::<_, Infallible>))
                .keep_alive(KeepAlive::default())
                .into_response(),
        ),
    }
}

fn worker(mut engine: Engine, mut jobs: mpsc::Receiver<Job>) {
    while let Some(job) = jobs.blocking_recv() {
        if !job.reply.is_closed() {
            serve_job(&mut engine, job);
        }
    }
}

fn serve_job(engine: &mut Engine, job: Job) {
    let Job {
        input,
        sampling,
        stream,
        include_usage,
        completion,
        reply,
    } = job;
    let (prompt, param) = match input {
        Input::Messages(messages) => (engine.chat_prompt(&messages), "messages"),
        Input::Prompt(prompt) => (Ok(prompt), "prompt"),
    };
    let ids = match prompt.and_then(|prompt| engine.tokenize(&prompt)) {
        Ok(ids) => ids,
        Err(error) => {
            let _ = reply.send(Err(ApiError::invalid(error.to_string()).param(param)));
            return;
        }
    };
    let separate_reasoning = completion.kind == Kind::Chat && engine.separates_reasoning();

    if !stream {
        if !reply.is_closed() {
            let result = engine
                .generate(ids, &sampling, |_| !reply.is_closed())
                .map_err(ApiError::inference)
                .and_then(|generation| completion.full(generation, separate_reasoning));
            let _ = reply.send(result.map(Reply::Full));
        }
        return;
    }

    let (sender, events) = mpsc::channel(STREAM_BUFFER_EVENTS);
    if reply.send(Ok(Reply::Stream(events))).is_err() {
        return;
    }
    let send = |body| sender.blocking_send(event(body)).is_ok();
    if completion.kind == Kind::Chat && !send(completion.opening_chunk(include_usage)) {
        return;
    }

    let mut split = ReasoningSplit::new(separate_reasoning);
    let mut emit = |field, piece: &str| send(completion.chunk(field, piece, None, include_usage));
    let result = engine.generate(ids, &sampling, |piece| {
        !sender.is_closed() && split.push(piece, &mut emit)
    });
    match result {
        Ok(generation) => {
            let Some(reason) = finish_reason(&generation.finish) else {
                return;
            };
            if !split.finish(emit)
                || !send(completion.chunk(ChatField::Content, "", Some(reason), include_usage))
            {
                return;
            }
            if include_usage && !send(completion.usage_chunk(&generation)) {
                return;
            }
        }
        Err(error) => {
            if !send(ApiError::inference(error).body()) {
                return;
            }
        }
    }
    let _ = sender.blocking_send(Event::default().data("[DONE]"));
}

struct Completion {
    id: String,
    created: u64,
    model: Arc<str>,
    kind: Kind,
}

impl Completion {
    fn envelope(&self, stream: bool, choices: Value) -> Value {
        let object = match (self.kind, stream) {
            (Kind::Chat, false) => "chat.completion",
            (Kind::Chat, true) => "chat.completion.chunk",
            (Kind::Text, _) => "text_completion",
        };
        json!({"id": self.id, "object": object, "created": self.created,
            "model": self.model.as_ref(), "choices": choices})
    }

    fn full(&self, generation: Generation, separate_reasoning: bool) -> Result<Value, ApiError> {
        let reason = finish_reason(&generation.finish).ok_or_else(ApiError::unavailable)?;
        let choice = match self.kind {
            Kind::Chat => json!({
                "index": 0,
                "message": assistant_message(&generation.text, separate_reasoning),
                "finish_reason": reason,
            }),
            Kind::Text => Self::text_choice(&generation.text, Some(reason)),
        };
        let mut response = self.envelope(false, json!([choice]));
        response["usage"] = usage(&generation);
        Ok(response)
    }

    fn text_choice(text: &str, reason: Option<&str>) -> Value {
        json!({"index": 0, "text": text, "logprobs": null, "finish_reason": reason})
    }

    fn opening_chunk(&self, include_usage: bool) -> Value {
        self.stream_choice(json!({"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}), include_usage)
    }

    fn chunk(
        &self,
        field: ChatField,
        text: &str,
        reason: Option<&str>,
        include_usage: bool,
    ) -> Value {
        let choice = match self.kind {
            Kind::Text => Self::text_choice(text, reason),
            Kind::Chat => {
                let delta = if reason.is_none() {
                    json!({(field.key()): text})
                } else {
                    json!({})
                };
                json!({"index": 0, "delta": delta, "finish_reason": reason})
            }
        };
        self.stream_choice(choice, include_usage)
    }

    fn stream_choice(&self, choice: Value, include_usage: bool) -> Value {
        let mut response = self.envelope(true, json!([choice]));
        if include_usage {
            response["usage"] = Value::Null;
        }
        response
    }

    fn usage_chunk(&self, generation: &Generation) -> Value {
        let mut response = self.envelope(true, json!([]));
        response["usage"] = usage(generation);
        response
    }
}

#[derive(Clone, Copy)]
enum ChatField {
    Content,
    Reasoning,
}

impl ChatField {
    fn key(self) -> &'static str {
        match self {
            Self::Content => "content",
            Self::Reasoning => "reasoning_content",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReasoningPhase {
    Opening(usize),
    Trace,
    Answer,
}

struct ReasoningSplit {
    phase: ReasoningPhase,
    pending: String,
}

impl ReasoningSplit {
    const OPEN: &str = THINK_MARKERS[0];
    const CLOSE: &str = THINK_MARKERS[1];

    fn new(enabled: bool) -> Self {
        Self {
            phase: if enabled {
                ReasoningPhase::Opening(0)
            } else {
                ReasoningPhase::Answer
            },
            pending: String::new(),
        }
    }

    fn push(&mut self, piece: &str, mut emit: impl FnMut(ChatField, &str) -> bool) -> bool {
        if piece.is_empty() {
            return true;
        }
        if self.phase == ReasoningPhase::Answer {
            return emit(ChatField::Content, piece);
        }

        self.pending.push_str(piece);
        if let ReasoningPhase::Opening(offset) = self.phase {
            let trimmed = self.pending[offset..].trim_start();
            let start = self.pending.len() - trimmed.len();
            if trimmed.starts_with(Self::OPEN) {
                self.pending.drain(..start + Self::OPEN.len());
                self.phase = ReasoningPhase::Trace;
            } else if Self::OPEN.starts_with(trimmed) {
                self.phase = ReasoningPhase::Opening(start);
                return true;
            } else {
                self.phase = ReasoningPhase::Answer;
                return emit(ChatField::Content, &std::mem::take(&mut self.pending));
            }
        }

        if let Some(end) = self.pending.find(Self::CLOSE) {
            self.phase = ReasoningPhase::Answer;
            let pending = std::mem::take(&mut self.pending);
            let answer = &pending[end + Self::CLOSE.len()..];
            return (end == 0 || emit(ChatField::Reasoning, &pending[..end]))
                && (answer.is_empty() || emit(ChatField::Content, answer));
        }

        // Hold a possible closing tag until the next chunk confirms it.
        let suffix = (1..Self::CLOSE.len())
            .rev()
            .find(|&len| self.pending.ends_with(&Self::CLOSE[..len]))
            .unwrap_or(0);
        let end = self.pending.len() - suffix;
        let sent = end == 0 || emit(ChatField::Reasoning, &self.pending[..end]);
        self.pending.drain(..end);
        sent
    }

    fn finish(self, mut emit: impl FnMut(ChatField, &str) -> bool) -> bool {
        let field = if self.phase == ReasoningPhase::Trace {
            ChatField::Reasoning
        } else {
            ChatField::Content
        };
        self.pending.is_empty() || emit(field, &self.pending)
    }
}

fn assistant_message(text: &str, separate_reasoning: bool) -> Value {
    if !separate_reasoning {
        return json!({"role": "assistant", "content": text});
    }

    let (mut content, mut reasoning) = (String::new(), String::new());
    let mut emit = |field, piece: &str| {
        match field {
            ChatField::Content => content.push_str(piece),
            ChatField::Reasoning => reasoning.push_str(piece),
        }
        true
    };
    let mut split = ReasoningSplit::new(true);
    split.push(text, &mut emit);
    split.finish(emit);
    let mut message = json!({"role": "assistant", "content": content});
    if !reasoning.is_empty() {
        message[ChatField::Reasoning.key()] = json!(reasoning);
    }
    message
}

fn finish_reason(finish: &Finish) -> Option<&'static str> {
    match finish {
        Finish::Stop => Some("stop"),
        Finish::Length => Some("length"),
        Finish::Cancelled => None,
    }
}

fn usage(generation: &Generation) -> Value {
    json!({
        "prompt_tokens": generation.prompt_tokens,
        "completion_tokens": generation.completion_tokens,
        "total_tokens": generation.prompt_tokens + generation.completion_tokens,
    })
}

fn event(value: Value) -> Event {
    Event::default().data(value.to_string())
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use axum::body::{Body, to_bytes};
    use axum::http::{Request, header};
    use tower::ServiceExt;

    use super::*;

    async fn post(app: Router, payload: Value) -> Response {
        post_at(app, "/v1/chat/completions", payload).await
    }

    async fn post_at(app: Router, path: &str, payload: Value) -> Response {
        app.oneshot(
            Request::post(path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
    }

    async fn get_at(app: &Router, path: &str) -> Response {
        app.clone()
            .oneshot(Request::get(path).body(Body::empty()).unwrap())
            .await
            .unwrap()
    }

    async fn body(response: Response) -> String {
        let bytes = to_bytes(response.into_body(), REQUEST_BODY_BYTES)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    async fn sse_chunks(response: Response) -> Vec<Value> {
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        let stream = body(response).await;
        let mut frames: Vec<_> = stream
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .collect();
        assert_eq!(frames.pop(), Some("[DONE]"));
        frames
            .into_iter()
            .map(|frame| serde_json::from_str(frame).unwrap())
            .collect()
    }

    fn request(stream: bool) -> Value {
        json!({
            "model": "example",
            "messages": [{"role":"user", "content":[{"type":"text", "text":"Hello"}]}],
            "stream": stream,
            "temperature": 0.0,
            "top_p": 0.9,
            "max_completion_tokens": 8,
            "stop": "END"
        })
    }

    #[tokio::test]
    async fn models_and_api_errors_are_json() {
        let (sender, receiver) = mpsc::channel(1);
        let app = router("example".into(), sender);
        let response = get_at(&app, "/v1/models").await;
        assert_eq!(response.status(), StatusCode::OK);
        let list: Value = serde_json::from_str(&body(response).await).unwrap();
        assert_eq!(list["object"], "list");
        assert_eq!(list["data"][0]["id"], "example");
        let response = get_at(&app, "/v1/models/example").await;
        assert_eq!(response.status(), StatusCode::OK);
        let model: Value = serde_json::from_str(&body(response).await).unwrap();
        assert_eq!(model, list["data"][0]);
        assert_eq!(
            get_at(&app, "/v1/models/missing").await.status(),
            StatusCode::NOT_FOUND
        );

        let mut wrong_model = request(false);
        wrong_model["model"] = json!("absent");
        let response = post(app.clone(), wrong_model).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let error: Value = serde_json::from_str(&body(response).await).unwrap();
        assert_eq!(error["error"]["code"], "model_not_found");

        let mut wrong_role = request(false);
        wrong_role["messages"][0]["role"] = json!("tool");
        let response = post(app.clone(), wrong_role).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let error: Value = serde_json::from_str(&body(response).await).unwrap();
        assert_eq!(error["error"]["param"], "messages");

        let response = post_at(app.clone(), "/v1/completions", json!({})).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let error: Value = serde_json::from_str(&body(response).await).unwrap();
        assert_eq!(error["error"]["param"], "prompt");

        drop(receiver);
        let response = post(app, request(false)).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn home_and_health_routes() {
        let (sender, receiver) = mpsc::channel(1);
        let app = router("example".into(), sender);
        let page = get_at(&app, "/").await;
        assert_eq!(page.status(), StatusCode::OK);
        assert_eq!(
            page.headers()[header::CONTENT_TYPE],
            "text/html; charset=utf-8"
        );
        for path in ["/v1", "/v1/"] {
            let response = get_at(&app, path).await;
            assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
            assert_eq!(response.headers()[header::LOCATION], "/");
        }
        for path in ["/health", "/v1/health"] {
            let response = get_at(&app, path).await;
            assert_eq!(response.status(), StatusCode::OK);
            let health: Value = serde_json::from_str(&body(response).await).unwrap();
            assert_eq!(health, json!({"status": "ok"}));
        }
        drop(receiver);
        let response = get_at(&app, "/health").await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let error: Value = serde_json::from_str(&body(response).await).unwrap();
        assert_eq!(error["error"]["code"], "model_unavailable");
    }

    #[tokio::test]
    async fn sse_framing_and_busy_response() {
        let (sender, mut jobs) = mpsc::channel(1);
        let app = router("example".into(), sender);
        let mut payload = request(true);
        payload["stream_options"] = json!({"include_usage": true});
        let task = tokio::spawn(post(app.clone(), payload));
        let job = jobs.recv().await.unwrap();
        assert!(job.stream && job.include_usage);
        let Input::Messages(messages) = &job.input else {
            panic!("expected a chat job")
        };
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "Hello");
        assert_eq!(job.sampling.max_tokens, 8);
        assert_eq!(job.sampling.temperature, 0.0);
        assert_eq!(job.sampling.top_p, 0.9);
        assert_eq!(job.sampling.stop, ["END"]);
        let queued = tokio::spawn(post_at(
            app.clone(),
            "/v1/completions",
            json!({"prompt":"queued"}),
        ));
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while jobs.is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(get_at(&app, "/health").await.status(), StatusCode::OK);
        let response = post(app, request(false)).await;
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let error: Value = serde_json::from_str(&body(response).await).unwrap();
        assert_eq!(error["error"]["code"], "model_busy");
        queued.abort();
        assert!(queued.await.is_err());
        let cancelled = jobs.recv().await.unwrap();
        assert!(matches!(cancelled.input, Input::Prompt(prompt) if prompt == "queued"));
        assert!(cancelled.reply.is_closed());

        let (events, receiver) = mpsc::channel(3);
        assert!(job.reply.send(Ok(Reply::Stream(receiver))).is_ok());
        let response = task.await.unwrap();
        let chunk = job
            .completion
            .chunk(ChatField::Content, "Hi\nthere", None, true);
        events.send(event(chunk)).await.unwrap();
        events
            .send(event(job.completion.chunk(
                ChatField::Content,
                "",
                Some("length"),
                true,
            )))
            .await
            .unwrap();
        events.send(Event::default().data("[DONE]")).await.unwrap();
        drop(events);
        let chunks = sse_chunks(response).await;
        assert_eq!(chunks.len(), 2);
        let first = &chunks[0];
        assert_eq!(first["choices"][0]["delta"]["content"], "Hi\nthere");
        assert!(first["usage"].is_null());
        let last = &chunks[1];
        assert_eq!(last["choices"][0]["finish_reason"], "length");
        assert_eq!(last["id"], first["id"]);
    }

    #[tokio::test]
    async fn dropping_response_signals_stream_cancellation() {
        let (sender, mut jobs) = mpsc::channel(1);
        let task = tokio::spawn(post(router("example".into(), sender), request(true)));
        let job = jobs.recv().await.unwrap();
        let (events, receiver) = mpsc::channel(1);
        assert!(job.reply.send(Ok(Reply::Stream(receiver))).is_ok());
        let response = task.await.unwrap();
        assert!(!events.is_closed());
        drop(response);
        assert!(events.is_closed());
    }

    #[test]
    fn reasoning_chunks_match_full_message_at_all_boundaries() {
        let output = " \n<think>test π</think>Final <think>literal</think> text";
        let full = assistant_message(output, true);
        assert_eq!(full["reasoning_content"], "test π");
        assert_eq!(full["content"], "Final <think>literal</think> text");
        let completion = Completion {
            id: "chatcmpl-test".into(),
            created: 0,
            model: "example".into(),
            kind: Kind::Chat,
        };

        for boundary in output
            .char_indices()
            .map(|(index, _)| index)
            .chain([output.len()])
        {
            let mut split = ReasoningSplit::new(true);
            let mut chunks = Vec::new();
            let mut emit = |field, piece: &str| {
                chunks.push(completion.chunk(field, piece, None, false));
                true
            };
            assert!(split.push(&output[..boundary], &mut emit));
            assert!(split.push(&output[boundary..], &mut emit));
            assert!(split.finish(emit));
            for key in ["content", "reasoning_content"] {
                let joined: String = chunks
                    .iter()
                    .filter_map(|chunk| chunk["choices"][0]["delta"][key].as_str())
                    .collect();
                assert_eq!(joined, full[key].as_str().unwrap());
            }
        }
    }

    #[test]
    fn ordinary_output_and_unclosed_reasoning_preserve_text() {
        for ordinary in [
            "Here is <think>literal</think>",
            " \n<thi",
            "</think> literal",
        ] {
            assert_eq!(
                assistant_message(ordinary, true),
                json!({"role": "assistant", "content": ordinary})
            );
        }
        assert_eq!(
            assistant_message("<think>literal</think>", false),
            json!({"role": "assistant", "content": "<think>literal</think>"})
        );
        assert_eq!(
            assistant_message("<think>unfinished</thi", true),
            json!({"role": "assistant", "content": "", "reasoning_content": "unfinished</thi"})
        );
        assert_eq!(
            assistant_message("<think></think>answer", true),
            json!({"role": "assistant", "content": "answer"})
        );
    }

    #[test]
    fn seed_is_random_unless_the_request_sets_one() {
        let seed = |payload: Value| {
            let mut request: CompletionRequest = serde_json::from_value(payload).unwrap();
            request.sampling(Kind::Text).unwrap().seed
        };
        assert_eq!(seed(json!({"prompt": "Hi", "seed": 7})), 7);
        let drawn: std::collections::HashSet<_> =
            (0..4).map(|_| seed(json!({"prompt": "Hi"}))).collect();
        assert!(drawn.len() > 1, "unseeded requests repeated seed {drawn:?}");
    }

    #[test]
    fn cancelled_reasoning_does_not_emit_answer() {
        let mut split = ReasoningSplit::new(true);
        assert!(split.push(" \n<thi", |_, _| panic!(
            "partial opening tag must stay buffered"
        )));
        let mut emitted = Vec::new();
        assert!(!split.push("nk>trace</think>answer", |field, piece| {
            emitted.push((field.key(), piece.to_owned()));
            false
        }));
        assert_eq!(emitted, [("reasoning_content", "trace".to_owned())]);
    }

    #[tokio::test]
    #[ignore = "set ORYNDRAL_TEST_GGUF to stories15M-q4_0.gguf"]
    async fn real_model_chat_round_trip() {
        use std::path::PathBuf;

        use crate::model::ModelFile;
        use crate::planner::{PlanOptions, plan};

        let path = PathBuf::from(std::env::var_os("ORYNDRAL_TEST_GGUF").unwrap());
        let context = 128;
        let file = ModelFile::open(&path).unwrap();
        let placement = plan(&file.memory(context, 0).unwrap(), PlanOptions::default()).unwrap();
        let engine =
            Engine::load(file, placement, context, 0, Default::default(), None, None).unwrap();
        let (sender, receiver) = mpsc::channel(1);
        let model_worker = std::thread::spawn(move || worker(engine, receiver));
        let app = router("example".into(), sender);

        let mut payload = request(false);
        payload["messages"][0]["content"] = json!("Once upon a time");
        payload["max_completion_tokens"] = json!(24);
        let response = post(app.clone(), payload.clone()).await;
        assert_eq!(response.status(), StatusCode::OK);
        let full: Value = serde_json::from_str(&body(response).await).unwrap();
        let text = full["choices"][0]["message"]["content"].as_str().unwrap();
        assert_eq!(
            text,
            ", there was a little girl named Lily. She loved to play outside and explore the world around her. One day"
        );
        assert_eq!(full["usage"]["prompt_tokens"], 5);
        assert_eq!(full["usage"]["completion_tokens"], 24);

        payload["stream"] = json!(true);
        payload["stream_options"] = json!({"include_usage": true});
        let response = post(app.clone(), payload.clone()).await;
        let chunks = sse_chunks(response).await;
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        let streamed_text: String = chunks
            .iter()
            .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(streamed_text, text);
        assert_eq!(
            chunks[chunks.len() - 2]["choices"][0]["finish_reason"],
            full["choices"][0]["finish_reason"]
        );
        let final_chunk = chunks.last().unwrap();
        assert_eq!(final_chunk["choices"], json!([]));
        assert_eq!(final_chunk["usage"], full["usage"]);

        payload["messages"][0]["content"] = json!(" long".repeat(context * 2));
        let response = post(app.clone(), payload).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let error: Value = serde_json::from_str(&body(response).await).unwrap();
        assert_eq!(error["error"]["param"], "messages");

        let route = "/v1/completions";
        let mut payload =
            json!({"prompt": "Once upon a time", "temperature": 0.0, "max_tokens": 24});
        let response = post_at(app.clone(), route, payload.clone()).await;
        assert_eq!(response.status(), StatusCode::OK);
        let completion: Value = serde_json::from_str(&body(response).await).unwrap();
        assert_eq!(completion["object"], "text_completion");
        assert_eq!(completion["choices"][0]["text"], text);
        assert_eq!(completion["choices"][0]["finish_reason"], "length");
        assert_eq!(completion["usage"], full["usage"]);
        payload["stream"] = json!(true);
        payload["stream_options"] = json!({"include_usage": true});
        let response = post_at(app, route, payload).await;
        let chunks = sse_chunks(response).await;
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk["object"] == "text_completion")
        );
        let streamed_text: String = chunks
            .iter()
            .filter_map(|chunk| chunk["choices"][0]["text"].as_str())
            .collect();
        assert_eq!(streamed_text, text);
        assert_eq!(
            chunks[chunks.len() - 2]["choices"][0]["finish_reason"],
            "length"
        );
        assert_eq!(chunks.last().unwrap()["usage"], full["usage"]);
        model_worker.join().unwrap();
    }
}
