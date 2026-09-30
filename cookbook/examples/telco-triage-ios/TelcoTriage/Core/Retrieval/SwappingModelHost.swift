import Foundation
import os.log

/// Mode the underlying `LlamaBackend` is currently configured for.
///
/// Per ADR-021 §11.4.3, a single `LlamaBackend` instance holds either
/// the chat backbone (LFM2.5-350M-base) or the ColBERT backbone
/// (LFM2-ColBERT-350M) at any time — never both. The host below owns
/// the transitions between them.
public enum LlamaBackendMode: String, Sendable, Equatable {
    case chat
    case colbert
}

/// Errors raised when the model host can't honor a mode request.
public enum SwappingModelHostError: Error, LocalizedError {
    case missingGGUF(LlamaBackendMode, String)
    case loadFailed(LlamaBackendMode, underlying: Error)

    public var errorDescription: String? {
        switch self {
        case .missingGGUF(let mode, let path):
            return "missing GGUF for \(mode.rawValue): \(path)"
        case .loadFailed(let mode, let underlying):
            return "load failed for \(mode.rawValue): \(underlying.localizedDescription)"
        }
    }
}

/// Configuration for one mode's backbone — the path + the
/// llama.cpp parameters appropriate for that mode.
public struct LlamaBackendModeConfig: Sendable {
    public let path: String
    public let contextLength: UInt32
    public let gpuLayers: Int32

    public init(path: String, contextLength: UInt32, gpuLayers: Int32) {
        self.path = path
        self.contextLength = contextLength
        self.gpuLayers = gpuLayers
    }
}

/// Single-backend swap orchestrator (ADR-021 §11.4.3).
///
/// The chat dispatcher never touches `LlamaBackend.unload()` or
/// `loadModel(...)` directly. It calls `withColBERT { ... }` and the
/// host arranges: ensure ColBERT loaded → run the block → restore
/// chat. After the block returns, the chat backbone is back; adapters
/// must be re-applied by their consumers (Stage A heads,
/// `StageBGenerator`) because unload wipes the LoRA cache.
///
/// **First-principles reason this is an actor**: model state is global
/// to the backend. Two concurrent `withColBERT` calls would race on the
/// unload/load pair. Serializing through actor isolation makes the
/// race structurally impossible, not just "unlikely in practice."
///
/// The actor's "current mode" is cached — repeated requests for the
/// same mode are no-ops, so a multi-turn run that stays in chat never
/// pays the swap cost.
public actor SwappingModelHost {
    private let backend: LlamaBackend
    private let chatConfig: LlamaBackendModeConfig
    private let colbertConfig: LlamaBackendModeConfig

    /// The mode we believe the backend is currently in. `nil` means
    /// the host has never asked the backend to load anything — typical
    /// at boot before the first call.
    private var currentMode: LlamaBackendMode?

    /// Set by the boot path after it kicks off the initial chat load.
    /// If the boot loader sets this to `.chat`, the first
    /// `ensureMode(.chat)` will skip its load call (the backend is
    /// already loaded by buildLFMStack's detached warm-up task).
    public func setInitialMode(_ mode: LlamaBackendMode) {
        currentMode = mode
    }

    private let logger = Logger(
        subsystem: "ai.liquid.demos.telcotriage",
        category: "SwappingModelHost"
    )

    public init(
        backend: LlamaBackend,
        chatConfig: LlamaBackendModeConfig,
        colbertConfig: LlamaBackendModeConfig
    ) {
        self.backend = backend
        self.chatConfig = chatConfig
        self.colbertConfig = colbertConfig
    }

    /// Run `body` with the ColBERT backbone loaded. Restores chat
    /// before returning, even on throw. The backend reference passed
    /// to `body` is the same `LlamaBackend` the host owns — the
    /// dispatcher uses it for `allTokenEmbeddings` inside the block.
    public func withColBERT<T: Sendable>(
        _ body: @Sendable (LlamaBackend) async throws -> T
    ) async throws -> T {
        try await ensureMode(.colbert)
        let result: Result<T, Error>
        do {
            let value = try await body(backend)
            result = .success(value)
        } catch {
            result = .failure(error)
        }
        // Restore chat regardless of body outcome. We don't propagate
        // restoration errors because the body's outcome is what the
        // caller cares about — but we do log them loudly.
        do {
            try await ensureMode(.chat)
        } catch {
            logger.error(
                "swap-back to chat failed: \(error.localizedDescription, privacy: .public)"
            )
        }
        switch result {
        case .success(let v): return v
        case .failure(let e): throw e
        }
    }

    /// Idempotent: if `mode` is already current, no-op. Otherwise
    /// `unload()` + `loadModel(...)` on the backend.
    public func ensureMode(_ mode: LlamaBackendMode) async throws {
        if currentMode == mode { return }

        let config: LlamaBackendModeConfig
        switch mode {
        case .chat: config = chatConfig
        case .colbert: config = colbertConfig
        }

        let t0 = CFAbsoluteTimeGetCurrent()

        // Tear down current state (no-op if backend is already unloaded).
        await backend.unload()

        do {
            try await backend.loadModel(
                path: config.path,
                contextLength: config.contextLength,
                gpuLayers: config.gpuLayers,
                temperature: 0
            )
        } catch {
            currentMode = nil
            throw SwappingModelHostError.loadFailed(mode, underlying: error)
        }

        currentMode = mode
        let elapsedMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        logger.info(
            "swapped to \(mode.rawValue, privacy: .public) in \(String(format: "%.0f", elapsedMs), privacy: .public) ms"
        )
    }

    /// Current mode, for engineering-mode trace / telemetry.
    public func mode() -> LlamaBackendMode? { currentMode }
}
