import Foundation

/// One composer (or future Stage B polish) output, ready for grading
/// + UI rendering. Mirrors the Python `ComposedAnswer` dataclass.
///
/// `actionFired` is always `false` here — the composer never executes
/// tools; the dispatcher hands the answer to `ToolExecutor` only when
/// the user explicitly confirms. Keeping the field on the envelope
/// makes that invariant testable (`is_action_safe`).
public struct ComposedAnswer: Sendable, Equatable {
    public let text: String
    public let route: ComposerRoute
    public let citedPageID: String?
    public let renderedLinks: [String]
    public let renderedLinkLabels: [String]
    public let expectedLinkURL: String?
    public let requiresConfirmation: Bool?
    public let actionFired: Bool
    public let latencyMs: Double
    public let strategy: String
    public let hasStepChain: Bool
    public let usedFallback: Bool

    public init(
        text: String,
        route: ComposerRoute,
        citedPageID: String? = nil,
        renderedLinks: [String] = [],
        renderedLinkLabels: [String] = [],
        expectedLinkURL: String? = nil,
        requiresConfirmation: Bool? = nil,
        actionFired: Bool = false,
        latencyMs: Double = 0,
        strategy: String = "deterministic_composer",
        hasStepChain: Bool = false,
        usedFallback: Bool = false
    ) {
        self.text = text
        self.route = route
        self.citedPageID = citedPageID
        self.renderedLinks = renderedLinks
        self.renderedLinkLabels = renderedLinkLabels
        self.expectedLinkURL = expectedLinkURL
        self.requiresConfirmation = requiresConfirmation
        self.actionFired = actionFired
        self.latencyMs = latencyMs
        self.strategy = strategy
        self.hasStepChain = hasStepChain
        self.usedFallback = usedFallback
    }
}
