import CryptoKit
import Foundation
import os.log

/// Runtime drift gate for the bundled RAG corpus (ADR-021 §11.5 L4).
///
/// At Stage B training time, the script that builds the corpus emits
/// `corpus-hash.txt` containing the sha256 of `rag-chunks-v1.json`.
/// At app boot, this struct re-computes the same sha256 over the
/// bundled corpus and compares. Mismatch = the corpus on disk has
/// drifted from what Stage B was trained against.
///
/// Mismatch is a warning, not a fatal — the app still runs, and the
/// retrieval path still works (it doesn't depend on Stage B
/// memorization). But Stage B's grounded generation will now operate
/// against retrieved chunks Stage B hasn't seen during training, so
/// faithfulness gates become the safety net.
///
/// The 73% production-RAG failure mode the web research surfaced
/// ("retrieval is where things go wrong, and the pipeline keeps
/// serving silently") includes corpus drift as a key culprit. This
/// gate catches it loudly.
public struct CorpusManifest: Sendable {
    public let computedHash: String
    public let trainingHash: String
    public let matches: Bool

    private static let logger = Logger(
        subsystem: "ai.liquid.demos.telcotriage",
        category: "CorpusManifest"
    )

    /// Compute the current corpus hash and compare against the
    /// training-time hash from `corpus-hash.txt`. Returns a manifest
    /// even on mismatch (caller decides what to do). Throws only when
    /// either resource is missing.
    public static func bundled(in bundle: Bundle = .main) throws -> CorpusManifest {
        guard let chunksURL = bundle.url(forResource: "rag-chunks-v1", withExtension: "json") else {
            throw ColBERTIndexError.missingResource("rag-chunks-v1.json")
        }
        guard let hashURL = bundle.url(forResource: "corpus-hash", withExtension: "txt") else {
            throw ColBERTIndexError.missingResource("corpus-hash.txt")
        }

        let chunksData = try Data(contentsOf: chunksURL)
        let computed = SHA256.hash(data: chunksData)
            .map { String(format: "%02x", $0) }
            .joined()

        let trainingHash = try String(contentsOf: hashURL, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let manifest = CorpusManifest(
            computedHash: computed,
            trainingHash: trainingHash,
            matches: computed == trainingHash
        )

        if manifest.matches {
            logger.info(
                "corpus hash OK (sha256 prefix \(String(computed.prefix(12)), privacy: .public)…)"
            )
        } else {
            // Loud warning per §11.5: drift detected, downstream
            // faithfulness gates carry the safety burden.
            logger.warning(
                "corpus drift detected. training=\(String(self.trainingHash(trainingHash).prefix(12)), privacy: .public)… computed=\(String(computed.prefix(12)), privacy: .public)… Stage B was trained against a different corpus. Retrieval still works; faithfulness gates are now the only safety net against stale grounding."
            )
        }
        return manifest
    }

    /// Helper used only by the log formatter to avoid escaping
    /// closures over `self` in a struct.
    private static func trainingHash(_ s: String) -> String { s }
}
