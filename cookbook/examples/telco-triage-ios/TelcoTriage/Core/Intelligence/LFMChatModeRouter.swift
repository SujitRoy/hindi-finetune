import Foundation

/// Production `ChatModeRouter` backed by an LFM2.5-350M prompt over
/// the base model — no LoRA adapter, no training data, no fine-tune.
/// If validation shows the base model is not discriminative enough
/// across the 4 modes, upgrade to a small dedicated LoRA (~1,500
/// examples of query → mode pairs) without changing the protocol.
///
/// Why base model first:
///   1. No training data to generate, no H100 time to spend.
///   2. A 4-way classification over short queries is within the
///      competence envelope of a decently-prompted 350M model.
///   3. If we end up needing a fine-tune, validation lets us size
///      the dataset from measured confusion pairs — not guesses.
///
/// Inference dispatches through `AdapterInferenceBackend` so the
/// router works identically against on-device llama.cpp, a sidecar
/// backend, or any future transport. `adapterPath: ""` signals "use
/// the loaded base model, do not apply a LoRA."
///
/// Errors are caught and surfaced as `.outOfScope` with confidence
/// 0.0 — a fail-safe default that keeps the chat flow alive even when
/// inference fails. Never crashes the chat flow.
public struct LFMChatModeRouter: ChatModeRouter {
    private let backend: AdapterInferenceBackend
    private let adapterPath: String
    private let maxTokens: Int

    /// - Parameter adapterPath: path to the ChatModeRouter LoRA GGUF
    ///   (e.g. `chat-mode-router-v2.gguf`). Pass the empty string to
    ///   run against the base LFM2.5-350M only — accuracy drops to
    ///   ~23 % without the LoRA per the Phase B validation run.
    public init(
        backend: AdapterInferenceBackend,
        adapterPath: String,
        maxTokens: Int = 64
    ) {
        self.backend = backend
        self.adapterPath = adapterPath
        // 64 is sized for the JSON schema below: 4-mode enum + short
        // reasoning + confidence. 40 tokens is the measured upper
        // bound for well-formed output plus margin.
        self.maxTokens = maxTokens
    }

    public func classify(query: String) async -> ChatModePrediction {
        // ADR-024 — single-turn only. ADR-023 Phase 1's history-aware
        // overload was retired after device testing proved that
        // prompt-stuffing amplified parse failures on the 350M base.
        // Multi-turn signal now flows through the pairwise relational
        // heads on cached hidden states (see ADR-024 §4.5), NOT
        // through the chat_mode prompt.
        let start = Date()
        let prompt = Self.buildPrompt(query: query)

        let raw: String
        do {
            raw = try await backend.generate(
                messages: [.user(prompt)],
                adapterPath: adapterPath,
                maxTokens: maxTokens
            )
        } catch {
            AppLog.intelligence.error("chat mode router backend failed: \(error.localizedDescription, privacy: .public)")
            return ChatModePrediction(
                mode: .outOfScope,
                confidence: 0.0,
                reasoning: "inference error — defaulting to out-of-scope",
                runtimeMS: Int(Date().timeIntervalSince(start) * 1000)
            )
        }

        guard let parsed = Self.parseAssistantJSON(raw) else {
            let preview = raw.prefix(200).replacingOccurrences(of: "\n", with: "⏎")
            AppLog.intelligence.error("chat mode router produced unparseable output (len=\(raw.count, privacy: .public)) preview=\(preview, privacy: .private)")
            return ChatModePrediction(
                mode: .outOfScope,
                confidence: 0.0,
                reasoning: "unparseable model output",
                runtimeMS: Int(Date().timeIntervalSince(start) * 1000)
            )
        }

        return ChatModePrediction(
            mode: parsed.mode,
            confidence: parsed.confidence,
            reasoning: parsed.reasoning,
            runtimeMS: Int(Date().timeIntervalSince(start) * 1000)
        )
    }

    // MARK: - Prompt

    /// Short, schema-locked prompt. The 4-mode taxonomy is explicit so
    /// the model doesn't invent a 5th class; the JSON example pins the
    /// output format. Examples cover the interesting modality-sensitive
    /// pairs ("how do I restart my router" vs "restart my router") —
    /// the exact distinction a 17-intent classifier can't make.
    ///
    /// **Single-turn only**: ADR-023 Phase 1's history-aware variant
    /// was retired by ADR-024 — see the class-level doc above for the
    /// rationale.
    static func buildPrompt(query: String) -> String {
        return """
        Classify this telco home internet customer support message into exactly one mode.

        Modes:
        - kb_question: the user is asking for information or how-to. Answered from the knowledge base.
          Examples: "how do I restart my router", "what are my WPS options", "where is the reset button"
        - tool_action: the user wants the assistant to do something right now. Answered by running a tool.
          Examples: "restart my router", "run a speed test", "pause internet for my son"
        - personal_summary: the user wants a summary of their own account, network, plan, or usage.
          Examples: "summarize my home network", "tell me about my plan", "how is my wifi doing"
        - out_of_scope: the query is not about telco home internet support.
          Examples: "what's the weather", "tell me a joke", "who won the game"

        Query: "\(query)"

        Respond with a JSON object containing:
          mode: one of kb_question, tool_action, personal_summary, out_of_scope
          confidence: a number between 0.0 and 1.0
          reasoning: one short sentence justifying the mode

        JSON:
        """
    }

    // MARK: - Parsing

    /// Extracts mode + confidence + reasoning. Reuses the shared JSON
    /// fence-stripper and balanced-brace walker so the same emission
    /// quirks (markdown fences, trailing prose) are tolerated in all
    /// LFM-backed classifiers.
    ///
    /// **YAML-fallback path (2026-05-27):** on real iPhones the chat-mode-
    /// router LoRA (Q4-trained, now composed with Q8 base after Phase δ's
    /// quant-mismatch fix) occasionally emits its answer in YAML-style
    /// `key: value` lines instead of JSON, then loops back to `JSON:` and
    /// starts again. The model picks the RIGHT mode — it just emits in
    /// the wrong format. Rather than fall back to `outOfScope` (which
    /// contradicts the topic_gate signal), parse the YAML.
    ///
    /// Proper fix is a retrain of chat-mode-router on Q8 base. This
    /// parser-side tolerance is insurance against the same format drift
    /// recurring under future LoRA composition changes.
    static func parseAssistantJSON(_ raw: String) -> (
        mode: ChatMode,
        confidence: Double,
        reasoning: String
    )? {
        let trimmed = JSONExtract.stripFences(raw).trimmingCharacters(in: .whitespacesAndNewlines)

        // Primary path: JSON object.
        if let jsonSlice = JSONExtract.firstJSONObject(in: trimmed),
           let data = jsonSlice.data(using: .utf8),
           let any = try? JSONSerialization.jsonObject(with: data),
           let dict = any as? [String: Any],
           let modeString = dict["mode"] as? String,
           let mode = ChatMode(rawValue: modeString) {
            let confidence = (dict["confidence"] as? Double)
                ?? (dict["confidence"] as? NSNumber)?.doubleValue
                ?? 0.0
            let reasoning = (dict["reasoning"] as? String) ?? ""
            return (mode, confidence, reasoning)
        }

        // Fallback path: YAML-style `key: value` lines.
        return parseYAMLStyle(trimmed)
    }

    /// Tolerant parser for `mode: <value>` / `confidence: <number>` /
    /// `reasoning: <text>` line-based output. Handles the loop-back
    /// pattern where the model emits `JSON:` again followed by another
    /// set of lines — we take the FIRST occurrence of each field and
    /// stop at the next `JSON:` marker.
    ///
    /// Returns nil if `mode` isn't recoverable; tolerates missing
    /// confidence (defaults to 0.5 — neutral, since the model emitted
    /// SOMETHING parseable) and missing reasoning (defaults to "").
    static func parseYAMLStyle(_ text: String) -> (
        mode: ChatMode,
        confidence: Double,
        reasoning: String
    )? {
        var mode: ChatMode?
        var confidence: Double?
        var reasoning: String = ""

        // Stop scanning at a second "JSON:" marker (the loop-back) so
        // we don't pick up trailing repetitions that may have different
        // values.
        var sawJsonMarker = false
        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            // Skip the leading "JSON:" prompt-echo if present; mark on
            // subsequent ones to terminate.
            if line.lowercased().hasPrefix("json:") {
                if sawJsonMarker { break }
                sawJsonMarker = true
                continue
            }
            // Markdown bullets and YAML list-dashes are tolerated.
            let stripped = line.drop(while: { $0 == "-" || $0 == "*" || $0.isWhitespace })
            guard let colonIdx = stripped.firstIndex(of: ":") else { continue }
            let key = stripped[..<colonIdx]
                .trimmingCharacters(in: .whitespaces)
                .lowercased()
            let valueRaw = stripped[stripped.index(after: colonIdx)...]
                .trimmingCharacters(in: CharacterSet(charactersIn: " \t\"',"))

            switch key {
            case "mode":
                // Take the first mode hit; ignore reps.
                if mode == nil, let m = ChatMode(rawValue: valueRaw) {
                    mode = m
                }
            case "confidence":
                if confidence == nil {
                    confidence = Double(valueRaw)
                }
            case "reasoning":
                if reasoning.isEmpty {
                    reasoning = String(valueRaw)
                }
            default:
                continue
            }
        }

        guard let resolvedMode = mode else { return nil }
        return (resolvedMode, confidence ?? 0.5, reasoning)
    }
}
