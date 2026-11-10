use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use candle::quantized::{
    gguf_file::{Content, Value},
    tokenizer::TokenizerFromGguf,
};
use tokenizers::{
    AddedToken, Tokenizer,
    decoders::{byte_fallback::ByteFallback, fuse::Fuse, sequence::Sequence, strip::Strip},
    models::bpe::{BPE, Vocab},
    normalizers::{Prepend, Replace},
};

const VOCAB: &str = "tokenizer.ggml.tokens";
const TOKEN_TYPES: &str = "tokenizer.ggml.token_type";

pub fn load(content: &Content, path: Option<&Path>) -> Result<Tokenizer> {
    let tokenizer = if let Some(path) = path {
        Tokenizer::from_file(path)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("reading tokenizer {}", path.display()))?
    } else {
        match metadata(content, "tokenizer.ggml.model")?
            .to_string()?
            .as_str()
        {
            "llama" => sentencepiece(content)?,
            "gpt2" => byte_bpe(content)?,
            model => bail!("GGUF tokenizer {model:?} requires a matching tokenizer.json"),
        }
    };
    let vocab = metadata(content, VOCAB)?.to_vec()?;
    ensure!(
        tokenizer.get_vocab_size(true) == vocab.len(),
        "tokenizer vocabulary size differs from GGUF"
    );
    for (id, token) in vocab.iter().enumerate() {
        ensure!(
            tokenizer.token_to_id(token.to_string()?) == Some(u32::try_from(id)?),
            "tokenizer vocabulary differs from GGUF at token ID {id}"
        );
    }
    Ok(tokenizer)
}

fn metadata<'a>(content: &'a Content, key: &str) -> Result<&'a Value> {
    content
        .metadata
        .get(key)
        .with_context(|| format!("missing GGUF metadata {key}"))
}

fn byte_bpe(content: &Content) -> Result<Tokenizer> {
    let pre = metadata(content, "tokenizer.ggml.pre")?.to_string()?;
    // Candle recognizes `llama3` while GGUF producers also use `llama-bpe`.
    match pre.as_str() {
        "qwen2" => Ok(Tokenizer::from_gguf(content)?),
        "llama-bpe" | "llama-v3" | "llama3" => {
            let mut settings = Content {
                magic: content.magic,
                metadata: content.metadata.clone(),
                tensor_infos: Default::default(),
                tensor_data_offset: 0,
            };
            settings
                .metadata
                .insert("tokenizer.ggml.pre".into(), Value::String("llama3".into()));
            settings
                .metadata
                .insert("tokenizer.ggml.ignore_merges".into(), Value::Bool(true));
            Ok(Tokenizer::from_gguf(&settings)?)
        }
        _ => bail!("GGUF byte BPE pre-tokenizer {pre:?} requires a matching tokenizer.json"),
    }
}

fn sentencepiece(content: &Content) -> Result<Tokenizer> {
    let tokens = metadata(content, VOCAB)?
        .to_vec()?
        .iter()
        .map(Value::to_string)
        .collect::<candle::Result<Vec<_>>>()?;
    let scores = metadata(content, "tokenizer.ggml.scores")?
        .to_vec()?
        .iter()
        .map(Value::to_f32)
        .collect::<candle::Result<Vec<_>>>()?;
    let types = metadata(content, TOKEN_TYPES)?
        .to_vec()?
        .iter()
        .map(Value::to_i32)
        .collect::<candle::Result<Vec<_>>>()?;
    ensure!(
        scores.len() == tokens.len() && types.len() == tokens.len(),
        "GGUF tokenizer scores and types must cover the vocabulary"
    );
    ensure!(
        scores.iter().all(|score| score.is_finite()),
        "non-finite GGUF token score"
    );
    let unk = metadata(content, "tokenizer.ggml.unknown_token_id")?.to_u32()?;
    let unk_token = tokens
        .get(unk as usize)
        .context("unknown token ID is out of range")?;
    let vocab: Vocab = tokens
        .iter()
        .enumerate()
        .map(|(id, token)| Ok(((*token).clone(), u32::try_from(id)?)))
        .collect::<Result<_>>()?;
    ensure!(
        vocab.len() == tokens.len(),
        "duplicate GGUF vocabulary entries"
    );

    // Legacy Llama GGUF omits explicit merges. Recover candidate pairs from
    // normal pieces, ordered by the resulting piece's SentencePiece score.
    let mut ranked: Vec<_> = (0..tokens.len()).filter(|&id| types[id] == 1).collect();
    ranked.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    let is_normal = |part: &str| vocab.get(part).is_some_and(|&id| types[id as usize] == 1);
    let mut merges = Vec::new();
    for id in ranked {
        let token = tokens[id];
        for (boundary, _) in token.char_indices().skip(1) {
            let (left, right) = token.split_at(boundary);
            if is_normal(left) && is_normal(right) {
                merges.push((left.to_owned(), right.to_owned()));
            }
        }
    }

    let model = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .unk_token((*unk_token).clone())
        .byte_fallback(true)
        .build()
        .map_err(anyhow::Error::msg)?;
    let mut tokenizer = Tokenizer::new(model);
    let prefix_space = content
        .metadata
        .get("tokenizer.ggml.add_space_prefix")
        .map(Value::to_bool)
        .transpose()?
        .unwrap_or(true);
    let mut normalizers = Vec::new();
    if prefix_space {
        normalizers.push(Prepend::new("▁".into()).into());
    }
    normalizers.push(Replace::new(" ", "▁").map_err(anyhow::Error::msg)?.into());
    tokenizer
        .with_normalizer(Some(tokenizers::normalizers::Sequence::new(normalizers)))
        .map_err(anyhow::Error::msg)?;

    let mut decoders = vec![
        Replace::new("▁", " ").map_err(anyhow::Error::msg)?.into(),
        ByteFallback::default().into(),
        Fuse::default().into(),
    ];
    if prefix_space {
        decoders.push(Strip::new(' ', 1, 0).into());
    }
    tokenizer.with_decoder(Some(Sequence::new(decoders)));
    let special_tokens: Vec<_> = tokens
        .iter()
        .zip(types)
        .filter(|(_, kind)| matches!(kind, 2..=5))
        .map(|(token, _)| AddedToken::from((*token).clone(), true))
        .collect();
    tokenizer
        .add_special_tokens(special_tokens)
        .map_err(anyhow::Error::msg)?;
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::gguf_file::VersionedMagic;

    #[test]
    fn sentencepiece_uses_scores_and_byte_fallback() -> Result<()> {
        // `▁a` competes with `ab`. The higher-scoring merge wins, and a
        // character outside the vocabulary falls back to its UTF-8 bytes.
        let pieces = ["<unk>", "▁", "a", "b", "▁a", "ab", "<0xC3>", "<0xA9>"];
        let content = Content {
            magic: VersionedMagic::GgufV3,
            metadata: [
                ("tokenizer.ggml.model", Value::String("llama".into())),
                (
                    VOCAB,
                    Value::Array(pieces.map(|s| Value::String(s.into())).into()),
                ),
                (
                    TOKEN_TYPES,
                    Value::Array([2, 1, 1, 1, 1, 1, 6, 6].map(Value::I32).into()),
                ),
                (
                    "tokenizer.ggml.scores",
                    Value::Array([0., 0., 0., 0., -2., -1., 0., 0.].map(Value::F32).into()),
                ),
                ("tokenizer.ggml.unknown_token_id", Value::U32(0)),
            ]
            .into_iter()
            .map(|(key, value)| (key.into(), value))
            .collect(),
            tensor_infos: Default::default(),
            tensor_data_offset: 0,
        };
        let tokenizer = load(&content, None)?;
        let encoding = tokenizer.encode("abé", false).map_err(anyhow::Error::msg)?;
        assert_eq!(encoding.get_ids(), &[1, 5, 6, 7]);
        assert_eq!(
            tokenizer
                .decode(encoding.get_ids(), true)
                .map_err(anyhow::Error::msg)?,
            "abé"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires ORYNDRAL_TEST_GGUF pointing to stories15M-q4_0.gguf"]
    fn tinystories_matches_llama_cpp_token_ids() -> Result<()> {
        let path = std::env::var_os("ORYNDRAL_TEST_GGUF").context("set ORYNDRAL_TEST_GGUF")?;
        let mut model = std::io::BufReader::new(std::fs::File::open(path)?);
        let tokenizer = load(&Content::read(&mut model)?, None)?;
        // llama-tokenize from a4d4f2fd4 with --no-bos --ids.
        let cases: &[(&str, &[u32])] = &[
            (
                "Once upon a time there was a",
                &[9038, 2501, 263, 931, 727, 471, 263],
            ),
            (
                "There was a little girl.",
                &[1670, 471, 263, 2217, 7826, 29889],
            ),
            ("apple 🍎", &[26163, 29871, 243, 162, 144, 145]),
            ("  extra   spaces", &[259, 4805, 259, 8162]),
            (
                "Line one.\nLine two.",
                &[7407, 697, 29889, 13, 3542, 1023, 29889],
            ),
        ];
        for (prompt, expected) in cases {
            let encoding = tokenizer
                .encode(*prompt, false)
                .map_err(anyhow::Error::msg)?;
            assert_eq!(encoding.get_ids(), *expected, "{prompt:?}");
            assert_eq!(
                tokenizer
                    .decode(expected, true)
                    .map_err(anyhow::Error::msg)?,
                *prompt
            );
        }
        Ok(())
    }
}
