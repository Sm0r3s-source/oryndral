use crate::model::{
    Architecture, ExpertOptions, IoStats, Model, ModelFile, PREFILL_BATCH, Session, get_usize,
};
use crate::planner::Plan;
use anyhow::{Context, Result, anyhow, ensure};
use candle::quantized::gguf_file::Value;
use minijinja::{Environment, Error, ErrorKind};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::Serialize;
use std::{collections::HashSet, path::Path, time::Instant};
use tokenizers::Tokenizer;

/// Opening and closing tags of a reasoning block.
pub const THINK_MARKERS: [&str; 2] = ["<think>", "</think>"];
const THINK_OPEN: &str = "<think>";
const THINK_SKIP: &str = "<think>\n\n</think>\n\n";

#[derive(Clone, Debug, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Per-request settings for rendering a chat prompt.
#[derive(Clone, Debug, Default)]
pub struct ChatOptions {
    /// Extra chat template variables, such as `enable_thinking`.
    pub template: serde_json::Map<String, serde_json::Value>,
    /// Whether a reasoning model thinks before it answers. `None` leaves that to the model.
    pub thinking: Option<bool>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Sampling {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: Option<usize>,
    pub seed: u64,
    pub stop: Vec<String>,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            max_tokens: 32,
            temperature: 0.8,
            top_p: 0.95,
            top_k: None,
            seed: 42,
            stop: Vec::new(),
        }
    }
}

impl Sampling {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.max_tokens > 0, "max_tokens must be positive");
        ensure!(
            self.temperature.is_finite() && (0.0..=2.0).contains(&self.temperature),
            "temperature must be between 0 and 2"
        );
        ensure!(
            self.top_p.is_finite() && self.top_p > 0.0 && self.top_p <= 1.0,
            "top_p must be in (0, 1]"
        );
        ensure!(self.top_k != Some(0), "top_k must be positive");
        ensure!(
            self.stop.len() <= 4 && self.stop.iter().all(|s| !s.is_empty() && s.len() <= 1024),
            "stop accepts up to four nonempty strings of up to 1024 bytes each"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Finish {
    Stop,
    Length,
    Cancelled,
}

#[derive(Debug, Serialize)]
pub struct Metrics {
    pub prefill_seconds: f64,
    pub prefill_tokens_per_second: f64,
    pub first_token_seconds: Option<f64>,
    pub decode_seconds: f64,
    pub decode_tokens_per_second: Option<f64>,
    pub elapsed_seconds: f64,
    pub io: IoStats,
}

#[derive(Debug, Serialize)]
pub struct Generation {
    pub text: String,
    pub prompt_tokens: usize,
    /// Leading prompt tokens whose attention state was reused from the previous generation.
    pub cached_prompt_tokens: usize,
    pub completion_tokens: usize,
    pub prompt_token_ids: Vec<u32>,
    pub completion_token_ids: Vec<u32>,
    pub finish: Finish,
    pub metrics: Metrics,
}

pub struct Engine {
    model: Model,
    tokenizer: Tokenizer,
    bos: Option<u32>,
    eos: HashSet<u32>,
    think_ids: [Option<u32>; THINK_MARKERS.len()],
    bos_text: String,
    eos_text: String,
    template: Option<String>,
    add_bos: bool,
    context: usize,
    /// The tokens of the last generation and the attention state they left behind.
    /// A prompt that starts with the same tokens continues from that state.
    cache: Option<(Vec<u32>, Session)>,
}

impl Engine {
    pub fn load(
        file: ModelFile,
        plan: Plan,
        context: usize,
        gpu_index: usize,
        experts: ExpertOptions,
        tokenizer_path: Option<&Path>,
        template_path: Option<&Path>,
    ) -> Result<Self> {
        let tokenizer = crate::tokenizer::load(&file.gguf, tokenizer_path)?;
        let meta = &file.gguf.metadata;
        let special = |name: &str| -> Result<Option<u32>> {
            get_usize(meta, &format!("tokenizer.ggml.{name}_token_id"))?
                .map(|id| {
                    ensure!(id < file.config.vocab, "invalid {name} token ID");
                    Ok(id as u32)
                })
                .transpose()
        };
        let bos = special("bos")?;
        let eos_id = special("eos")?;
        let mut eos: HashSet<_> = [eos_id, special("eot")?, special("eom")?]
            .into_iter()
            .flatten()
            .collect();
        let mut think_ids = [None; THINK_MARKERS.len()];
        if file.config.architecture == Architecture::Qwen2 {
            for (id, token) in tokenizer.get_added_tokens_decoder() {
                if !token.special {
                    continue;
                }
                if matches!(token.content.as_str(), "<|im_end|>" | "<|endoftext|>") {
                    eos.insert(id);
                }
                if let Some(index) = THINK_MARKERS.iter().position(|m| *m == token.content) {
                    think_ids[index] = Some(id);
                }
            }
        }
        let bos_text = bos
            .and_then(|id| tokenizer.id_to_token(id))
            .unwrap_or_default();
        let eos_text = eos_id
            .and_then(|id| tokenizer.id_to_token(id))
            .unwrap_or_default();
        let add_bos = match meta.get("tokenizer.ggml.add_bos_token") {
            Some(Value::Bool(v)) => *v,
            None => meta
                .get("tokenizer.ggml.model")
                .and_then(|v| v.to_string().ok())
                .is_some_and(|v| v == "llama"),
            _ => return Err(anyhow!("invalid tokenizer.ggml.add_bos_token")),
        };
        ensure!(
            !add_bos || bos.is_some(),
            "tokenizer requests BOS but the GGUF lacks its ID"
        );
        let template = if let Some(path) = template_path {
            Some(
                std::fs::read_to_string(path)
                    .with_context(|| format!("read {}", path.display()))?,
            )
        } else {
            meta.get("tokenizer.chat_template")
                .map(Value::to_string)
                .transpose()?
                .cloned()
        };
        if let Some(template) = &template {
            let mut env = template_environment();
            env.add_template("chat", template)
                .context("parse chat template")?;
        }
        let model = Model::load(file, plan, context, gpu_index, experts)?;
        Ok(Self {
            model,
            tokenizer,
            bos,
            eos,
            think_ids,
            bos_text,
            eos_text,
            template,
            add_bos,
            context,
            cache: None,
        })
    }

    /// A session holding the longest reusable prefix of `ids`, and that prefix's length.
    /// At least one token is left to evaluate, since its logits start the reply.
    fn resume(&mut self, ids: &[u32]) -> Result<(Session, usize)> {
        let Some((seen, mut session)) = self.cache.take() else {
            return Ok((self.model.session(), 0));
        };
        let shared = seen.iter().zip(ids).take_while(|(a, b)| a == b).count();
        let shared = shared.min(ids.len() - 1);
        session.truncate(shared)?;
        Ok((session, shared))
    }

    pub fn separates_reasoning(&self) -> bool {
        self.think_ids.iter().all(Option::is_some)
    }

    /// Callers supply a nonempty list of system, user and assistant messages.
    pub fn chat_prompt(&self, messages: &[Message], options: &ChatOptions) -> Result<String> {
        if let Some(template) = &self.template {
            let mut env = template_environment();
            env.add_template("chat", template)?;
            let mut context = options.template.clone();
            context.insert("messages".into(), serde_json::to_value(messages)?);
            context.insert("bos_token".into(), self.bos_text.as_str().into());
            context.insert("eos_token".into(), self.eos_text.as_str().into());
            context.insert("add_generation_prompt".into(), true.into());
            let mut prompt = env
                .get_template("chat")?
                .render(context)
                .context("render chat template")?;
            // A template that reads `enable_thinking` ends the prompt with a think tag itself.
            // For the others, start the reply with an open or an empty reasoning block.
            let tagged = THINK_MARKERS.iter().any(|m| prompt.trim_end().ends_with(m));
            if let (Some(thinking), true, false) =
                (options.thinking, self.separates_reasoning(), tagged)
            {
                prompt.push_str(if thinking { THINK_OPEN } else { THINK_SKIP });
            }
            return Ok(prompt);
        }
        ensure!(
            messages.len() == 1 && messages[0].role == "user",
            "base model without a chat template accepts a single user message"
        );
        Ok(messages[0].content.clone())
    }

    /// Encode a prompt, adding BOS where the model expects it, and check it fits the context.
    pub fn tokenize(&self, prompt: &str) -> Result<Vec<u32>> {
        let mut ids = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode prompt: {e}"))?
            .get_ids()
            .to_vec();
        if self.add_bos && ids.first().copied() != self.bos {
            ids.insert(0, self.bos.context("missing BOS token")?);
        }
        ensure!(!ids.is_empty(), "prompt has no tokens");
        ensure!(
            ids.len() < self.context,
            "prompt uses {} tokens in a {}-token context",
            ids.len(),
            self.context
        );
        Ok(ids)
    }

    /// Generate from `tokenize` output with sampling settings that passed `Sampling::validate`.
    pub fn generate(
        &mut self,
        ids: Vec<u32>,
        sampling: &Sampling,
        mut emit: impl FnMut(&str) -> bool,
    ) -> Result<Generation> {
        let started = Instant::now();
        let before = self.model.stats();
        // The cache is empty from here until the generation ends without an error, so a
        // failed forward pass cannot leave a half-updated session behind.
        let (mut state, cached) = self.resume(&ids)?;
        let mut logits = None;
        let mut finish = Finish::Length;
        let mut prefilled = 0;
        for batch in ids[cached..].chunks(PREFILL_BATCH) {
            if !emit("") {
                finish = Finish::Cancelled;
                break;
            }
            prefilled += batch.len();
            let last = cached + prefilled == ids.len();
            logits = self.model.forward(batch, &mut state, last)?;
        }
        let prefill = started.elapsed().as_secs_f64();
        let limit = sampling.max_tokens.min(self.context - ids.len());
        let mut tokens = Vec::with_capacity(limit);
        let mut text = String::new();
        let mut pending = String::new();
        let mut rng = StdRng::seed_from_u64(sampling.seed);
        let mut first_token = None;
        let mut last_token = None;
        let mut decoder = self.tokenizer.decode_stream(true);
        for id in &ids {
            decoder
                .step(*id)
                .map_err(|e| anyhow!("initialize streaming decoder: {e}"))?;
        }
        for i in 0..limit {
            if finish == Finish::Cancelled || !emit("") {
                finish = Finish::Cancelled;
                break;
            }
            let scores = logits.take().context("missing model logits")?;
            let token = sample(&scores, sampling, &mut rng)?;
            tokens.push(token);
            let now = started.elapsed().as_secs_f64();
            first_token.get_or_insert(now);
            last_token = Some(now);
            if self.eos.contains(&token) {
                finish = Finish::Stop;
                break;
            }
            let piece = decoder
                .step(token)
                .map_err(|e| anyhow!("decode token: {e}"))?;
            // Qwen2 reasoning markers are visible text even when the tokenizer marks them special.
            let marker = self
                .think_ids
                .iter()
                .zip(THINK_MARKERS)
                .find_map(|(id, marker)| (*id == Some(token)).then_some(marker));
            if piece.is_some() || marker.is_some() {
                pending.push_str(piece.as_deref().unwrap_or_default());
                pending.push_str(marker.unwrap_or_default());
                let (visible, stop) = stop_boundary(&pending, &sampling.stop);
                let visible = pending.drain(..visible).collect::<String>();
                if !visible.is_empty() {
                    text.push_str(&visible);
                    if !emit(&visible) {
                        finish = Finish::Cancelled;
                        break;
                    }
                }
                if stop {
                    pending.clear();
                    finish = Finish::Stop;
                    break;
                }
            }
            if i + 1 < limit {
                logits = self.model.forward(&[token], &mut state, true)?;
            }
        }
        if finish != Finish::Cancelled && !pending.is_empty() {
            text.push_str(&pending);
            if !emit(&pending) {
                finish = Finish::Cancelled;
            }
        }
        let elapsed = started.elapsed().as_secs_f64();
        let decode = last_token
            .zip(first_token)
            .map_or(0., |(last, first)| last - first);
        let decode_rate = (decode > 0.).then(|| (tokens.len() - 1) as f64 / decode);
        let after = self.model.stats();
        // The last sampled token is only evaluated when another one follows it.
        let mut seen = ids.clone();
        seen.extend(&tokens);
        seen.truncate(state.tokens());
        self.cache = Some((seen, state));
        Ok(Generation {
            text,
            prompt_tokens: ids.len(),
            cached_prompt_tokens: cached,
            completion_tokens: tokens.len(),
            prompt_token_ids: ids,
            completion_token_ids: tokens,
            finish,
            metrics: Metrics {
                prefill_seconds: prefill,
                prefill_tokens_per_second: prefilled as f64 / prefill,
                first_token_seconds: first_token,
                decode_seconds: decode,
                decode_tokens_per_second: decode_rate,
                elapsed_seconds: elapsed,
                io: IoStats {
                    weight_read_bytes: after.weight_read_bytes - before.weight_read_bytes,
                    expert_hits: after.expert_hits - before.expert_hits,
                    expert_misses: after.expert_misses - before.expert_misses,
                },
            },
        })
    }

    pub fn load_io(&self) -> IoStats {
        self.model.stats()
    }
}

fn template_environment() -> Environment<'static> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |message: String| -> std::result::Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, message))
        },
    );
    env
}

fn sample(scores: &[f32], cfg: &Sampling, rng: &mut StdRng) -> Result<u32> {
    ensure!(
        !scores.is_empty() && scores.iter().all(|s| s.is_finite()),
        "invalid model logits"
    );
    let mut ranked = scores.iter().copied().enumerate().collect::<Vec<_>>();
    let cmp = |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0));
    if cfg.temperature == 0. {
        return Ok(ranked
            .iter()
            .min_by(|a, b| cmp(a, b))
            .context("empty logits")?
            .0 as u32);
    }
    if let Some(k) = cfg.top_k.filter(|k| *k < ranked.len()) {
        ranked.select_nth_unstable_by(k, cmp);
        ranked.truncate(k);
    }
    ranked.sort_unstable_by(cmp);
    let max = f64::from(ranked[0].1);
    let mut weights = ranked
        .iter()
        .map(|(_, s)| ((f64::from(*s) - max) / cfg.temperature).exp())
        .collect::<Vec<_>>();
    let mut total = weights.iter().sum::<f64>();
    if cfg.top_p < 1.0 {
        let cutoff = total * cfg.top_p;
        let mut sum = 0.0;
        let keep = weights
            .iter()
            .position(|w| {
                sum += w;
                sum >= cutoff
            })
            .map_or(weights.len(), |i| i + 1);
        weights.truncate(keep);
        total = sum;
    }
    let mut draw = rng.random::<f64>() * total;
    for (i, weight) in weights.iter().enumerate() {
        if draw < *weight {
            return Ok(ranked[i].0 as u32);
        }
        draw -= weight;
    }
    Ok(ranked[weights.len() - 1].0 as u32)
}

fn stop_boundary(text: &str, stops: &[String]) -> (usize, bool) {
    if let Some(index) = stops.iter().filter_map(|s| text.find(s)).min() {
        return (index, true);
    }
    for (index, _) in text.char_indices() {
        if stops.iter().any(|s| s.starts_with(&text[index..])) {
            return (index, false);
        }
    }
    (text.len(), false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_prefixes_never_leak_into_stream() {
        let stops = vec!["🐈stop".to_string(), "END".to_string()];
        assert_eq!(stop_boundary("text🐈st", &stops), (4, false));
        assert_eq!(stop_boundary("🐈stop afterwards", &stops), (0, true));
        assert_eq!(stop_boundary("keep END ignore", &stops), (5, true));
    }

    #[test]
    #[ignore = "requires the pinned TinyStories Q4_0 GGUF"]
    fn reused_prompt_prefix_generates_the_same_tokens() {
        let path = std::env::var_os("ORYNDRAL_TEST_GGUF").expect("pinned GGUF path");
        let load = || {
            let file = ModelFile::open(Path::new(&path)).unwrap();
            let memory = file.memory(128, 0).unwrap();
            let plan = crate::planner::plan(&memory, Default::default()).unwrap();
            Engine::load(file, plan, 128, 0, Default::default(), None, None).unwrap()
        };
        let greedy = Sampling {
            temperature: 0.,
            max_tokens: 12,
            ..Sampling::default()
        };
        let run = |engine: &mut Engine, prompt: &str| {
            let ids = engine.tokenize(prompt).unwrap();
            engine.generate(ids, &greedy, |_| true).unwrap()
        };
        let (mut warm, mut cold) = (load(), load());
        let first = run(&mut warm, "Once upon a time");
        assert_eq!(first.cached_prompt_tokens, 0);

        // The same prompt again reuses all but its last token.
        let again = run(&mut warm, "Once upon a time");
        assert_eq!(again.cached_prompt_tokens, first.prompt_tokens - 1);
        assert_eq!(again.completion_token_ids, first.completion_token_ids);

        // This prompt follows the previous reply for a few tokens and then departs from it,
        // so the session is cut back in the middle of what the model generated.
        let longer = "Once upon a time, there was a little dog who";
        let reused = run(&mut warm, longer);
        assert!(reused.cached_prompt_tokens > first.prompt_tokens);
        assert!(reused.cached_prompt_tokens < reused.prompt_tokens);
        assert_eq!(
            reused.completion_token_ids,
            run(&mut cold, longer).completion_token_ids
        );

        // An unrelated prompt shares at most the BOS token.
        let other = "The sun was bright and";
        let fresh = run(&mut warm, other);
        assert!(fresh.cached_prompt_tokens <= 1);
        assert_eq!(
            fresh.completion_token_ids,
            run(&mut cold, other).completion_token_ids
        );
    }

    #[test]
    fn greedy_and_top_k_choose_the_best_token() {
        let mut rng = StdRng::seed_from_u64(42);
        for s in [
            Sampling {
                temperature: 0.,
                ..Sampling::default()
            },
            Sampling {
                top_k: Some(1),
                ..Sampling::default()
            },
        ] {
            assert_eq!(sample(&[1., 9., 3.], &s, &mut rng).unwrap(), 1);
        }
    }
}
