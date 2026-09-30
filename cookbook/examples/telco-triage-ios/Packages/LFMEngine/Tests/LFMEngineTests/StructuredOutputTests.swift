import Foundation
import Testing
@testable import LFMEngine

@Suite("Structured Output — JSON Early-Stop")
struct StructuredOutputTests {

    private func shouldStop(_ text: String, mode: GenerationParams.OutputMode = .jsonObject) async -> Bool {
        let backend = LlamaBackend()
        return await backend.shouldStop(for: text, mode: mode)
    }

    // MARK: - Text Mode (never stops)

    @Test("Text mode never triggers early stop")
    func textModeNeverStops() async {
        let result = await shouldStop(#"{"intent":"check_balance"}"#, mode: .text)
        #expect(result == false)
    }

    // MARK: - Complete JSON Objects

    @Test("Detects simple complete JSON object")
    func simpleCompleteObject() async {
        #expect(await shouldStop(#"{"intent":"check_balance","confidence":0.95}"#))
    }

    @Test("Detects empty JSON object")
    func emptyObject() async {
        #expect(await shouldStop("{}"))
    }

    @Test("Detects nested JSON object")
    func nestedObject() async {
        #expect(await shouldStop(#"{"a":{"b":1}}"#))
    }

    @Test("Detects deeply nested JSON object")
    func deeplyNestedObject() async {
        #expect(await shouldStop(#"{"a":{"b":{"c":{"d":true}}}}"#))
    }

    @Test("Handles leading whitespace before object")
    func leadingWhitespace() async {
        #expect(await shouldStop(#"  { "key": "value" }"#))
    }

    @Test("Handles newlines and tabs in whitespace")
    func whitespaceVariants() async {
        #expect(await shouldStop("\n\t{\"key\": \"value\"}\n"))
    }

    // MARK: - Incomplete JSON

    @Test("Does not stop on incomplete JSON object")
    func incompleteObject() async {
        #expect(await shouldStop(#"{"intent":"check_balance""#) == false)
    }

    @Test("Does not stop on opening brace only")
    func openingBraceOnly() async {
        #expect(await shouldStop("{") == false)
    }

    @Test("Does not stop on nested incomplete object")
    func nestedIncomplete() async {
        #expect(await shouldStop(#"{"a":{"b":1}"#) == false)
    }

    @Test("Does not stop on empty string")
    func emptyString() async {
        #expect(await shouldStop("") == false)
    }

    @Test("Does not stop on whitespace only")
    func whitespaceOnly() async {
        #expect(await shouldStop("   ") == false)
    }

    // MARK: - Non-Object Input

    @Test("Does not stop on array start")
    func arrayStart() async {
        #expect(await shouldStop(#"[{"a":1}]"#) == false)
    }

    @Test("Does not stop on plain text")
    func plainText() async {
        #expect(await shouldStop("hello world") == false)
    }

    @Test("Does not stop on number")
    func numberInput() async {
        #expect(await shouldStop("42") == false)
    }

    // MARK: - String Escaping

    @Test("Handles escaped quotes in strings")
    func escapedQuotes() async {
        #expect(await shouldStop(#"{"a":"he said \"hi\""}"#))
    }

    @Test("Handles escaped backslash before quote")
    func escapedBackslashBeforeQuote() async {
        // String value is: path\\  (ends with literal backslash)
        // Then the closing " of the value, then }
        #expect(await shouldStop(#"{"path":"C:\\"}"#))
    }

    @Test("Braces inside strings are ignored")
    func bracesInsideStrings() async {
        #expect(await shouldStop(#"{"msg":"use {braces} here"}"#))
    }

    @Test("Incomplete string does not trigger stop")
    func incompleteString() async {
        // Missing closing quote on the string value
        #expect(await shouldStop(#"{"a":"incomplete}"#) == false)
    }

    // MARK: - Multiple Objects

    @Test("Stops after first complete object even with trailing content")
    func stopsAfterFirstObject() async {
        #expect(await shouldStop(#"{"a":1}{"b":2}"#))
    }

    @Test("Stops after first object with trailing text")
    func stopsAfterFirstObjectWithTrailing() async {
        #expect(await shouldStop(#"{"a":1} some trailing text"#))
    }

    // MARK: - Realistic Model Output

    @Test("Handles realistic intent router output")
    func realisticIntentRouterOutput() async {
        let output = """
        {"intent": "check_balance", "routing_target": "/api/v2/accounts/balance", \
        "normalized_query": "what is my balance", "confidence": 0.94}
        """
        #expect(await shouldStop(output))
    }

    @Test("Handles realistic PII detection output")
    func realisticPIIOutput() async {
        let output = """
        {"entities": [{"type": "ssn", "value": "***", "confidence": 0.99}], "has_pii": true}
        """
        #expect(await shouldStop(output))
    }

    @Test("Partial realistic output does not stop")
    func partialRealisticOutput() async {
        let output = #"{"intent": "check_balance", "routing_target""#
        #expect(await shouldStop(output) == false)
    }
}
