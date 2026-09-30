import Foundation

/// 4-flag multi-label slot detection from the v2 understanding layer
/// (ADR-022 §4.3). Tells the workflow which kinds of arguments the user
/// has supplied so the tool-action lane can ask a clarification BEFORE
/// firing a tool with missing required slots.
///
/// **Wire contract** (frozen — these indices map directly into the
/// trained head's per-class sigmoid output; reordering would silently
/// invert flag meanings):
///
///  - index 0: `has_device`       — explicit device or person reference
///                                  ("son's tablet", "the router", "my
///                                  iPhone"). Required for parental
///                                  controls + extender reboot.
///  - index 1: `has_location`     — physical room or zone ("upstairs",
///                                  "basement", "garage"). Required by
///                                  the extender-reboot fast-path.
///  - index 2: `has_time`         — explicit time bound ("until bedtime",
///                                  "for an hour", "tomorrow morning").
///                                  Required by `setDowntime` (future
///                                  scope per ADR-021).
///  - index 3: `has_account_ref`  — account-scoped reference ("my plan",
///                                  "my bill", "my line"). Useful for
///                                  routing nav-only lanes and account
///                                  service flows.
///
/// **Why a head and not regex extraction**: the existing
/// `RegexQueryExtractor` already extracts CONCRETE values when patterns
/// match — but that's a downstream concern. The head answers the
/// "is the user TALKING about a device" question for messages where
/// regex can't lift the value but a human reader can tell. E.g. "pause
/// it for the small one" has `has_device=true` (the small one = the
/// kid) but no extractable value. The workflow can then ask which
/// device, instead of firing the tool with a missing arg.
public struct SlotCompleteness: Sendable, Equatable, Hashable, Codable {
    public let hasDevice: Bool
    public let hasLocation: Bool
    public let hasTime: Bool
    public let hasAccountRef: Bool

    public init(
        hasDevice: Bool,
        hasLocation: Bool,
        hasTime: Bool,
        hasAccountRef: Bool
    ) {
        self.hasDevice = hasDevice
        self.hasLocation = hasLocation
        self.hasTime = hasTime
        self.hasAccountRef = hasAccountRef
    }

    /// All flags off. Used for turns where the head wasn't consulted
    /// (chat lanes that don't need slot signal) or where the v2 head
    /// isn't bundled yet.
    public static let none = SlotCompleteness(
        hasDevice: false,
        hasLocation: false,
        hasTime: false,
        hasAccountRef: false
    )

    /// True when at least one slot is present. Used by the trace card
    /// to render "—" vs an actual slot list.
    public var hasAnySlot: Bool {
        hasDevice || hasLocation || hasTime || hasAccountRef
    }

    /// Membership check against a `Slot` enum. Keeps consumers from
    /// thinking in flag-name strings.
    public func contains(_ slot: Slot) -> Bool {
        switch slot {
        case .device:     return hasDevice
        case .location:   return hasLocation
        case .time:       return hasTime
        case .accountRef: return hasAccountRef
        }
    }

    /// The set of slots currently flagged true. Convenience for
    /// `Set`-style operations (e.g., "which required slots are
    /// missing?").
    public var presentSlots: Set<Slot> {
        var slots: Set<Slot> = []
        if hasDevice { slots.insert(.device) }
        if hasLocation { slots.insert(.location) }
        if hasTime { slots.insert(.time) }
        if hasAccountRef { slots.insert(.accountRef) }
        return slots
    }

    /// Decode from a per-class binary vector (the head's
    /// `classifyMultiLabel().binaryVector`). Defensive against vectors
    /// shorter than 4 (returns false for the missing tail), but never
    /// against longer (a length mismatch is a hard contract bug we
    /// want to surface during training-export validation, not
    /// silently swallow at runtime).
    public static func from(binaryVector: [Int]) -> SlotCompleteness {
        func bit(_ idx: Int) -> Bool {
            idx < binaryVector.count && binaryVector[idx] == 1
        }
        return SlotCompleteness(
            hasDevice: bit(0),
            hasLocation: bit(1),
            hasTime: bit(2),
            hasAccountRef: bit(3)
        )
    }
}

/// Typed enumeration of the four slot kinds. Keep this in lockstep with
/// `SlotCompleteness`'s fields — a new slot means a head re-export +
/// schema bump.
public enum Slot: String, Sendable, Equatable, Hashable, CaseIterable, Codable {
    case device      = "has_device"
    case location    = "has_location"
    case time        = "has_time"
    case accountRef  = "has_account_ref"

    public var displayName: String {
        switch self {
        case .device:     return "Device"
        case .location:   return "Location"
        case .time:       return "Time"
        case .accountRef: return "Account ref"
        }
    }
}
