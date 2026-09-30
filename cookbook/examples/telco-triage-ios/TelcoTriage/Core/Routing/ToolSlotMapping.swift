import Foundation

/// Per-intent mapping from the abstract `Slot` taxonomy (ADR-022
/// `slot_completeness` head) to the concrete argument key the tool's
/// catalog entry expects (`target_device`, `extender_name`, etc.).
///
/// **Why this lives next to ToolIntent**: when a new `ToolIntent` case
/// is added, Swift's exhaustive switch forces consideration of every
/// `(intent, slot)` pair below. The compiler tells you to deal with
/// it, instead of a stringly-typed mapping silently returning the
/// wrong key.
///
/// **Consumed by** `ChatViewModel.applyPostDecisionActions` when the
/// router emits `.accumulateSlotsFromAlignment(intent, slots)` —
/// each slot is translated into the per-tool argument key and merged
/// into `ConversationState.slotStore[intent]`.
public extension ToolIntent {
    /// The argument key in the tool catalog for the supplied slot kind.
    /// Exhaustive over `(ToolIntent, Slot)`; a new tool intent forces
    /// the compiler to consider its slot-key mapping.
    ///
    /// **Coverage today**: the only tools that consume clarification-
    /// recovered slots are `toggleParentalControls` (target_device)
    /// and `rebootExtender` (extender_name). Other intents either
    /// take no required slot args or accept the slot's
    /// `displayName.lowercased()` canonical form. When a future intent
    /// needs a specific key, add the case here — do not silently
    /// fall through.
    func argumentKey(for slot: Slot) -> String {
        switch self {
        case .toggleParentalControls:
            switch slot {
            case .device:     return "target_device"
            case .location:   return "location"
            case .time:       return "downtime_until"
            case .accountRef: return "account_ref"
            }

        case .rebootExtender:
            switch slot {
            case .device:     return "device"
            case .location:   return "extender_name"
            case .time:       return "time"
            case .accountRef: return "account_ref"
            }

        case .scheduleTechnician:
            switch slot {
            case .device:     return "device"
            case .location:   return "location"
            case .time:       return "preferred_date"
            case .accountRef: return "account_ref"
            }

        case .restartRouter,
             .runSpeedTest,
             .checkConnection,
             .wpsPair,
             .runDiagnostics:
            // These tools take no required slot arguments. Returning
            // the slot's canonical form is a sensible default for
            // any optional plumbing the catalog might add later.
            // Adding a NEW slot kind to Slot will force this default
            // to be revisited (compiler exhaustiveness on the inner
            // switch).
            switch slot {
            case .device:     return "device"
            case .location:   return "location"
            case .time:       return "time"
            case .accountRef: return "account_ref"
            }
        }
    }
}
