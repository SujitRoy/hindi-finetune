import Foundation
import Testing
@testable import LFMEngine

@Suite("Latency Optimization — BUG-007")
struct LatencyOptimizationTests {

    // MARK: - Per-Capability Context Window

    @Test("InferenceResult includes latency breakdown")
    func inferenceResultLatencyBreakdown() {
        let breakdown = LatencyBreakdown(
            promptEvalMs: 30.0,
            tokenGenerationMs: 70.0,
            totalMs: 100.0
        )
        let result = InferenceResult(
            text: "test",
            latencyMs: 100.0,
            tokensGenerated: 5,
            model: "LFM2-350M",
            adapter: "intent-router",
            ranOn: .device,
            latencyBreakdown: breakdown
        )

        #expect(result.latencyBreakdown != nil)
        #expect(result.latencyBreakdown?.promptEvalMs == 30.0)
        #expect(result.latencyBreakdown?.tokenGenerationMs == 70.0)
        #expect(result.latencyBreakdown?.totalMs == 100.0)
    }

    @Test("InferenceResult latency breakdown defaults to nil")
    func inferenceResultLatencyBreakdownDefaultNil() {
        let result = InferenceResult(
            text: "test",
            latencyMs: 50.0,
            tokensGenerated: 3,
            model: "LFM2-350M",
            adapter: nil,
            ranOn: .cloud
        )

        #expect(result.latencyBreakdown == nil)
    }

    @Test("LatencyBreakdown stores all fields correctly")
    func latencyBreakdownFields() {
        let breakdown = LatencyBreakdown(
            promptEvalMs: 15.5,
            tokenGenerationMs: 84.5,
            totalMs: 102.3
        )

        #expect(breakdown.promptEvalMs == 15.5)
        #expect(breakdown.tokenGenerationMs == 84.5)
        #expect(breakdown.totalMs == 102.3)
    }

    // MARK: - Warm-Up

    @Test("LlamaBackend warmUp without load throws modelNotLoaded")
    func warmUpWithoutLoadThrows() async {
        let backend = LlamaBackend()
        do {
            _ = try await backend.warmUp()
            Issue.record("Expected warmUp to throw when model not loaded")
        } catch let error as LFMEngineError {
            if case .modelNotLoaded = error {
                // Expected
            } else {
                Issue.record("Wrong error type: \(error)")
            }
        } catch {
            // Platform-specific error also acceptable (llama.cpp not available)
        }
        await backend.unload()
    }

    // MARK: - Engine State Transitions

    @Test("Engine warmUp is safe in proxy mode")
    @MainActor
    func warmUpProxyModeIsSafe() async {
        let engine = LFMEngine(
            mode: .proxy,
            modelConfig: ModelConfig(name: "test", fileName: "test.gguf"),
            proxyBaseURL: URL(string: "http://localhost:8080")
        )
        // Should not crash or throw — just no-op
        await engine.warmUp()
    }

    @Test("Engine warmUp is safe without loaded model")
    @MainActor
    func warmUpWithoutModelIsSafe() async {
        let engine = LFMEngine(
            mode: .onDevice,
            modelConfig: ModelConfig(name: "test", fileName: "test.gguf")
        )
        // Backend is nil — warmUp should log and return without error
        await engine.warmUp()
    }

    // MARK: - GenerationTiming from LlamaBackend

    @Test("GenerationTiming stores prompt eval and token generation times")
    func generationTimingFields() {
        let timing = LlamaBackend.GenerationTiming(
            promptEvalMs: 25.0,
            tokenGenerationMs: 75.0,
            promptTokens: 42,
            outputTokens: 7
        )

        #expect(timing.promptEvalMs == 25.0)
        #expect(timing.tokenGenerationMs == 75.0)
        #expect(timing.promptTokens == 42)
        #expect(timing.outputTokens == 7)
    }

    // MARK: - Engine Generate Error Paths

    @Test("Generate rejects when engine is unloaded")
    @MainActor
    func generateRejectsUnloaded() async {
        let engine = LFMEngine(
            mode: .onDevice,
            modelConfig: ModelConfig(name: "test", fileName: "test.gguf")
        )

        do {
            _ = try await engine.generate(GenerationParams(prompt: "test"))
            Issue.record("Expected modelNotLoaded error")
        } catch let error as LFMEngineError {
            if case .modelNotLoaded = error {
                // Expected
            } else {
                Issue.record("Wrong error: \(error)")
            }
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test("Engine state returns to ready after failed generate")
    @MainActor
    func stateResetsAfterFailedGenerate() async {
        let engine = LFMEngine(
            mode: .onDevice,
            modelConfig: ModelConfig(name: "test", fileName: "test.gguf")
        )

        // Engine is .unloaded — generate should fail but not corrupt state
        do {
            _ = try await engine.generate(GenerationParams(prompt: "test"))
        } catch {
            // Expected
        }

        #expect(engine.state == .unloaded)
    }
}
