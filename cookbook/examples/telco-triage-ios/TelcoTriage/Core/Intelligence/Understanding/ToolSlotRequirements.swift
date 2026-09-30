import Foundation

/// Per-tool declaration of which `Slot` kinds the tool needs from the
/// understanding layer's `slotCompleteness` head before it can fire
/// safely. Drives the `.toolAction` workflow's clarification step
/// (ADR-022 §6 Phase 4 — `ClarifyMissingSlot` NBA).
///
/// **Why this lives next to the head, not on `Tool`**: argument
/// extraction is a generative LFM (`LFMToolSelector`) or deterministic
/// (`RegexQueryExtractor`) concern — it knows how to pull the VALUE
/// from a query. The `slot_completeness` head answers a different
/// question: "does the user have a value at all?". A tool with a
/// required `device` slot can be asked "which device?" when the head
/// says `has_device=false`, even when the regex would have failed too.
/// Decoupling means the head's signal is reusable across selectors.
///
/// **Closed-world principle**: every `ToolIntent` MUST have an entry
/// here. The `requirementsFor(_:)` lookup is total (returns `[]` for
/// "no slots needed", never nil). Adding a new `ToolIntent` case will
/// surface as a compile-time exhaustiveness warning in the static
/// declaration below — that's intentional, the same way the slot
/// requirements file must change when the tool roster does.
public enum ToolSlotRequirements {

    /// Lookup for "which slots does this tool need?". Total —
    /// returns `[]` when a tool has no required slots, never nil.
    /// Pure function over `ToolIntent` so callers can `let needed =
    /// requirementsFor(intent)` without an actor hop.
    public static func requirementsFor(_ intent: ToolIntent) -> Set<Slot> {
        switch intent {
        case .toggleParentalControls:
            // "Pause the internet" alone is ambiguous — for WHOM? The
            // parental-controls tool fires against a specific managed
            // device or person. Without a device slot, the workflow
            // must clarify before executing the side effect.
            return [.device]

        case .rebootExtender:
            // The user can have multiple extenders (upstairs, basement,
            // garage). Picking the wrong one rebooting the kitchen
            // extender during a Zoom call is a real-world cost.
            return [.location]

        case .scheduleTechnician:
            // The schedule-technician sheet UI can collect time
            // interactively, so the slot isn't strictly required for
            // the assistant to fire the proposal. We do NOT mark
            // `.time` as required — that would force a chat-level
            // clarification before the user even sees the schedule
            // sheet, which feels patronising.
            return []

        case .restartRouter, .runDiagnostics, .runSpeedTest,
             .checkConnection, .wpsPair:
            // These tools operate on the customer's primary router /
            // network. No argument selection needed — the tool's UI
            // confirmation step (per `Tool.requiresConfirmation`)
            // covers the safety boundary.
            return []
        }
    }

    /// The set of required slots that the user has NOT yet supplied.
    /// Used by `ClarifyMissingSlotNBA` and the tool-action workflow
    /// to decide whether to ask a clarification question BEFORE
    /// rendering the tool-decision card.
    ///
    /// When `slotCompleteness` is nil (v2 head not bundled), we
    /// can't know what's missing — the function returns `[]` so
    /// the workflow proceeds as today (tool fires; user sees
    /// regex-extracted args; existing argument-edit UI handles
    /// gaps). This is the "fail OPEN, never silently regress"
    /// principle from ADR-022 §4.3.
    public static func missingSlots(
        for intent: ToolIntent,
        given slots: SlotCompleteness?
    ) -> Set<Slot> {
        let required = requirementsFor(intent)
        guard let slots else {
            // No head signal → no clarification authority. Defer to
            // the existing tool-arg-edit affordance.
            return []
        }
        return required.subtracting(slots.presentSlots)
    }

    /// Convenience: does this intent need clarification given the
    /// current slot vector? True iff at least one required slot is
    /// missing.
    public static func needsClarification(
        for intent: ToolIntent,
        given slots: SlotCompleteness?
    ) -> Bool {
        !missingSlots(for: intent, given: slots).isEmpty
    }

    /// Render the missing-slot set as a single short clarification
    /// question for the chat UI. Pure function — no localisation
    /// layer hooked up to the POC yet, so the strings live here.
    /// One sentence each, ends in a question mark, no model
    /// invocation needed.
    public static func clarificationQuestion(
        for intent: ToolIntent,
        missing: Set<Slot>
    ) -> String? {
        // No missing slots → no question to ask. Caller should
        // check `needsClarification` before calling, but defensive
        // against bad inputs.
        guard !missing.isEmpty else { return nil }

        switch intent {
        case .toggleParentalControls:
            // The most common path: device is missing.
            if missing.contains(.device) {
                return "Which device or person should I pause the internet for?"
            }
        case .rebootExtender:
            if missing.contains(.location) {
                return "Which extender — upstairs, basement, or somewhere else?"
            }
        default:
            break
        }

        // Generic fallback covers any new tool that adds a slot
        // requirement before the per-tool copy lands.
        let slotNames = missing
            .map { $0.displayName.lowercased() }
            .sorted()
            .joined(separator: ", ")
        return "I need a bit more to do that — can you give me the \(slotNames)?"
    }
}
