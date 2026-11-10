use anyhow::{Context, Result, ensure};
use clap::Args;
use hf_hub::{Cache, Repo, RepoType, api::sync::ApiBuilder};
use std::path::PathBuf;

#[derive(Args)]
pub struct HubOptions {
    /// Download from a Hugging Face model repository (owner/name).
    #[arg(long, requires = "hf_file", conflicts_with = "model")]
    hf: Option<String>,
    /// Exact GGUF filename in the Hugging Face repository.
    #[arg(long, requires = "hf")]
    hf_file: Option<String>,
    /// Revision to use for the Hugging Face model. Defaults to main.
    #[arg(long, requires = "hf")]
    hf_revision: Option<String>,
    /// Override the standard Hugging Face cache location.
    #[arg(long, requires = "hf")]
    hf_cache: Option<PathBuf>,
    /// Use the Hugging Face cache without contacting the network.
    #[arg(long, requires = "hf")]
    offline: bool,
}

impl HubOptions {
    pub fn resolve(self, local: Option<PathBuf>) -> Result<PathBuf> {
        if let Some(path) = local {
            return Ok(path);
        }
        let id = self.hf.context("specify a local GGUF path or --hf")?;
        let file = self.hf_file.context("specify --hf-file")?;
        let revision = self.hf_revision.unwrap_or_else(|| "main".to_owned());
        ensure!(
            id.split('/').count() == 2 && safe_path(&id),
            "--hf must be a Hugging Face owner/name"
        );
        ensure!(
            safe_path(&file) && file.to_ascii_lowercase().ends_with(".gguf"),
            "--hf-file must be a repository-relative GGUF filename"
        );
        ensure!(safe_path(&revision), "invalid Hugging Face revision");
        let cache = self.hf_cache.map_or_else(Cache::default, Cache::new);
        let repo = Repo::with_revision(id.clone(), RepoType::Model, revision.clone());
        if let Some(path) = cache.repo(repo.clone()).get(&file) {
            return Ok(path);
        }
        ensure!(!self.offline, "{id}/{file} at {revision} is not cached");
        eprintln!("Downloading {id}/{file} at {revision}");
        ApiBuilder::from_cache(cache)
            .with_retries(2)
            .build()
            .context("initialize Hugging Face client")?
            .repo(repo)
            .get(&file)
            .with_context(|| format!("fetch {id}/{file} at {revision}"))
    }
}

fn safe_path(path: &str) -> bool {
    path.split('/').all(|part| {
        !part.is_empty()
            && !part.starts_with('.')
            && !part.ends_with('.')
            && part
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
    })
}

#[cfg(test)]
mod tests {
    use super::safe_path;

    #[test]
    fn repository_paths_stay_inside_the_cache() {
        assert!(safe_path("models/qwen2.5-0.5b-Q4_K_M.gguf"));
        for path in [
            "",
            "../model.gguf",
            "/model.gguf",
            "a//m.gguf",
            "a\\m.gguf",
            "m.gguf?x",
        ] {
            assert!(!safe_path(path), "{path:?}");
        }
    }
}
