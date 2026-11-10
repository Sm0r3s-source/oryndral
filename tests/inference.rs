use serde_json::Value;
use std::process::Command;

/// Run a greedy 24-token benchmark on four threads and return its JSON report.
fn run(fixture_env: &str, args: &[&str]) -> Value {
    let file = std::env::var_os(fixture_env)
        .unwrap_or_else(|| panic!("set {fixture_env} to the pinned GGUF"));
    let output = Command::new(env!("CARGO_BIN_EXE_oryndral"))
        .arg("benchmark")
        .arg(file)
        .args(["--threads", "4", "--max-tokens", "24", "--temperature", "0"])
        .args(args)
        .output()
        .expect("launch inference benchmark");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("benchmark JSON")
}

fn benchmark(fixture_env: &str, mode: &str, cache_mib: &str, policy: &str) -> Value {
    run(
        fixture_env,
        &[
            "--mode",
            mode,
            "--expert-cache-mib",
            cache_mib,
            "--expert-cache-policy",
            policy,
            "--context",
            "128",
            "--prompt",
            "Once upon a time",
        ],
    )
}

fn assert_reference(result: &Value, expected: &[u32]) {
    let output = &result["generation"];
    assert_eq!(
        output["prompt_token_ids"],
        serde_json::json!([1, 9038, 2501, 263, 931])
    );
    assert_eq!(output["completion_token_ids"], serde_json::json!(expected));
    assert!(
        output["metrics"]["decode_tokens_per_second"]
            .as_f64()
            .unwrap()
            > 0.
    );
}

#[test]
#[ignore = "requires the pinned TinyStories Q4_0 GGUF"]
fn dense_modes_match_llama_cpp_greedy_reference() {
    let resident = benchmark("ORYNDRAL_TEST_GGUF", "resident", "0", "lru");
    let disk = benchmark("ORYNDRAL_TEST_GGUF", "dense-stream", "0", "lru");
    let expected = [
        29892, 727, 471, 263, 2217, 7826, 4257, 365, 2354, 29889, 2296, 18012, 304, 1708, 5377,
        322, 26987, 278, 3186, 2820, 902, 29889, 3118, 2462,
    ];
    assert_reference(&resident, &expected);
    assert_reference(&disk, &expected);
    assert_eq!(resident["generation"]["text"], disk["generation"]["text"]);
    assert_eq!(
        resident["generation"]["metrics"]["io"]["weight_read_bytes"],
        0
    );
    assert!(
        disk["generation"]["metrics"]["io"]["weight_read_bytes"]
            .as_u64()
            .unwrap()
            > 0
    );
}

#[test]
#[ignore = "requires the pinned TinyStories MoE Q8_0 GGUF"]
fn moe_stream_cache_preserves_generation_and_avoids_reads() {
    let uncached = benchmark("ORYNDRAL_TEST_MOE_GGUF", "moe-stream", "0", "lru");
    let cached = benchmark("ORYNDRAL_TEST_MOE_GGUF", "moe-stream", "8", "lru");
    let lfu = benchmark("ORYNDRAL_TEST_MOE_GGUF", "moe-stream", "8", "lfu");
    let expected = [
        29892, 727, 471, 263, 2217, 7826, 4257, 365, 2354, 29889, 2296, 750, 263, 4802, 9883, 4257,
        14008, 29889, 365, 2354, 471, 1407, 5051, 322,
    ];
    assert_reference(&uncached, &expected);
    assert_reference(&cached, &expected);
    assert_reference(&lfu, &expected);
    let cold_io = &uncached["generation"]["metrics"]["io"];
    let cached_io = &cached["generation"]["metrics"]["io"];
    assert_eq!(cold_io["expert_hits"], 0);
    assert!(cached_io["expert_hits"].as_u64().unwrap() > 0);
    assert!(cached_io["weight_read_bytes"].as_u64() < cold_io["weight_read_bytes"].as_u64());
}

#[test]
#[ignore = "requires the pinned 1.93 GB Mythos-nano Q4_K_M GGUF"]
fn mythos_nano_matches_llama_cpp() {
    let result = run(
        "ORYNDRAL_TEST_MYTHOS_GGUF",
        &[
            "--chat",
            "--prompt",
            "What is 2 + 2? Answer briefly.",
            "--context",
            "512",
        ],
    );
    assert_eq!(
        result["generation"]["prompt_token_ids"],
        serde_json::json!([
            151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198,
            3838, 374, 220, 17, 488, 220, 17, 30, 21806, 26753, 13, 151645, 198, 151644, 77091,
            198,
        ])
    );
    assert_eq!(
        result["generation"]["completion_token_ids"],
        serde_json::json!([
            151665, 785, 1196, 17064, 25, 330, 3838, 374, 220, 17, 488, 220, 17, 30, 21806, 26753,
            1189, 1096, 374, 264, 4285, 6888, 3405, 13,
        ])
    );
    assert_eq!(
        result["generation"]["text"],
        r#"<think>The user asks: "What is 2 + 2? Answer briefly." This is a simple math question."#
    );
}
