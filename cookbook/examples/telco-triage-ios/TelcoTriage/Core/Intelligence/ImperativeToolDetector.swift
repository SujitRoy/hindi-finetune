import Foundation

/// Deterministic fast-path for unambiguous tool imperatives. Runs INSIDE
/// `runToolProposal` AFTER chatModeRouter has classified `.toolAction`,
/// before the LFM tool selector would otherwise spend ~1.7s picking
/// which of 8 tools to invoke. When the user's phrasing literally names
/// a tool ("run diagnostics", "restart my router", "pause my son's
/// tablet"), the LFM call is pure latency overhead — the answer is
/// already in the words.
///
/// This is NOT a routing pre-filter. It cannot change the lane
/// (chatModeRouter already locked that to `.toolAction`); it only
/// selects WHICH tool inside that lane. Returns nil → fall through to
/// the trained LFMToolSelector for ambiguous phrasings.
///
/// Why not retrain the tool selector to be faster? The tool selector
/// has to handle: argument extraction (target device, location),
/// confidence calibration, multi-tool disambiguation. None of that is
/// needed when the user said "run diagnostics" verbatim. The LFM call
/// is the wrong tool for the obvious cases — same way you don't ask
/// GPT-4 what time it is when there's a clock on the wall.
///
/// Coverage: all 8 `ToolIntent` cases have unambiguous-imperative
/// detection. Question forms ("how do I restart my router") are
/// explicitly REJECTED so KB lookups aren't hijacked.
public enum ImperativeToolDetector {
    /// Returns a tool intent if the query is an unambiguous imperative
    /// for one of the 8 tools. Returns nil otherwise — caller MUST
    /// fall through to the LFM tool selector.
    public static func detect(_ query: String) -> ToolIntent? {
        let q = query.lowercased()

        // Question forms are NEVER overridden — those are real KB
        // lookups even when they mention tool topics ("how do I
        // restart my router" → KB entry on router restart, not the
        // restart-router tool).
        for starter in Self.questionStarters where q.hasPrefix(starter) {
            return nil
        }

        // Order matters — more specific patterns first so e.g.
        // "restart my extender" hits rebootExtender (not restartRouter).
        if matchesRebootExtender(q) { return .rebootExtender }
        if matchesParentalControls(q) { return .toggleParentalControls }
        if matchesRunDiagnostics(q) { return .runDiagnostics }
        if matchesSpeedTest(q) { return .runSpeedTest }
        if matchesRestartRouter(q) { return .restartRouter }
        if matchesCheckConnection(q) { return .checkConnection }
        if matchesWPSPair(q) { return .wpsPair }
        if matchesScheduleTechnician(q) { return .scheduleTechnician }
        return nil
    }

    // MARK: - Question filter

    private static let questionStarters: [String] = [
        "how do", "how can", "how should", "how to", "how would",
        "what is", "what's", "what are", "what does", "what happens",
        "where", "why", "should i", "should we", "when",
        "can you tell", "can you explain", "could you tell",
        "is there a way", "is it possible",
    ]

    // MARK: - Per-tool matchers

    /// `pause/block/stop internet for <device|person>` — must combine
    /// an action verb + an internet noun + a device/person noun. "Block
    /// this site" alone is too ambiguous (could be ad-block, could be
    /// parental controls).
    private static func matchesParentalControls(_ q: String) -> Bool {
        let actions = ["pause", "block", "stop", "shut off", "kill", "disable", "cut off"]
        let internetNouns = ["internet", "wifi", "wi-fi", "wi fi", "network", "connection", "access"]
        let targets = [
            "tablet", "phone", "laptop", "computer", "console", "playstation",
            "ps4", "ps5", "xbox", "switch", "ipad", "iphone", "device",
            "kid", "kids", "son", "daughter", "child", "children", "teen",
            "his", "her", "their", "youtube", "tiktok",
        ]
        return actions.contains { q.contains($0) }
            && internetNouns.contains { q.contains($0) }
            && targets.contains { q.contains($0) }
    }

    /// `restart/reboot/reset extender|mesh node`. Specific because
    /// the extender restart is a different tool from gateway restart.
    private static func matchesRebootExtender(_ q: String) -> Bool {
        let verbs = ["restart", "reboot", "reset", "cycle", "power cycle", "kick"]
        let nouns = ["extender", "mesh node", "mesh point", "satellite", "wifi point"]
        return verbs.contains { q.contains($0) }
            && nouns.contains { q.contains($0) }
    }

    /// `restart/reboot/reset router|modem|gateway|box`. Explicitly
    /// EXCLUDES "extender" matches (handled above by rebootExtender).
    private static func matchesRestartRouter(_ q: String) -> Bool {
        let verbs = ["restart", "reboot", "reset", "cycle", "power cycle"]
        let nouns = ["router", "modem", "gateway", "wifi box", "fios box", "wifi router"]
        let excluded = ["extender", "mesh"]
        return verbs.contains { q.contains($0) }
            && nouns.contains { q.contains($0) }
            && !excluded.contains { q.contains($0) }
    }

    /// `run diagnostics / diagnose / network check`. Closely worded
    /// to the tool name itself.
    private static func matchesRunDiagnostics(_ q: String) -> Bool {
        let phrases = [
            "run diagnostics", "run a diagnostic", "run the diagnostics",
            "run network diagnostics", "diagnose my network",
            "diagnose my wifi", "diagnose my internet",
            "network diagnostic", "diagnostic on my", "diagnostics on my",
            "check network health", "network health check",
        ]
        return phrases.contains { q.contains($0) }
    }

    /// `speed test / test my speed / how fast`. Speed-test is the
    /// most-imperative tool — the phrase IS the tool.
    private static func matchesSpeedTest(_ q: String) -> Bool {
        let phrases = [
            "speed test", "speedtest", "run a speed test", "run speed test",
            "test my speed", "test my internet speed", "test my wifi speed",
            "check my speed", "measure my speed", "how fast is my",
            "what's my speed", "whats my speed",
        ]
        return phrases.contains { q.contains($0) }
    }

    /// `check connection / am I online / is wifi up`. Status checks
    /// that map to the explicit connectivity tool, distinct from
    /// running a full diagnostic.
    private static func matchesCheckConnection(_ q: String) -> Bool {
        let phrases = [
            "check connection", "check my connection", "check connectivity",
            "am i online", "am i connected", "is my wifi up",
            "is my internet up", "is the wifi working", "is the internet working",
            "wifi status", "internet status", "connection status",
        ]
        return phrases.contains { q.contains($0) }
    }

    /// `wps pair / enable wps / connect via wps`. WPS is technical
    /// enough that any mention is almost always a tool request.
    private static func matchesWPSPair(_ q: String) -> Bool {
        let phrases = [
            "wps pair", "pair via wps", "use wps", "enable wps",
            "turn on wps", "start wps", "press wps", "wps button",
            "connect with wps", "connect via wps",
        ]
        return phrases.contains { q.contains($0) }
    }

    /// `schedule a technician / send a tech / book a visit`.
    private static func matchesScheduleTechnician(_ q: String) -> Bool {
        let phrases = [
            "schedule a technician", "schedule technician", "book a technician",
            "book technician", "send a tech", "send out a tech",
            "send a technician", "send out a technician",
            "schedule a service visit", "book a service appointment",
            "have someone come out", "send someone out",
        ]
        return phrases.contains { q.contains($0) }
    }
}
