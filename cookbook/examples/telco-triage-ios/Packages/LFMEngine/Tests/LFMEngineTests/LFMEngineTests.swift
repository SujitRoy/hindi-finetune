import Testing
@testable import LFMEngine

@Suite("LFMEngine Types")
struct LFMEngineTypesTests {

    @Test("InferenceResult initializes correctly")
    func inferenceResultInit() {
        let result = InferenceResult(
            text: "hello",
            latencyMs: 25.0,
            tokensGenerated: 5,
            model: "LFM2-350M",
            adapter: "intent-router",
            ranOn: .device
        )

        #expect(result.text == "hello")
        #expect(result.latencyMs == 25.0)
        #expect(result.tokensGenerated == 5)
        #expect(result.ranOn == .device)
    }

    @Test("ModelConfig uses sensible defaults")
    func modelConfigDefaults() {
        let config = ModelConfig(name: "test", fileName: "test.gguf")

        #expect(config.contextLength == 2048)
        #expect(config.gpuLayers == 99)
        #expect(config.defaultTemperature == 0.0)
    }

    @Test("DownloadProgress fraction calculation")
    func downloadProgressFraction() {
        let progress = DownloadProgress(bytesDownloaded: 50, totalBytes: 100, fileName: "test.gguf")
        #expect(progress.fraction == 0.5)

        let zero = DownloadProgress(bytesDownloaded: 0, totalBytes: 0, fileName: "test.gguf")
        #expect(zero.fraction == 0.0)
    }

    @Test("All engine errors have descriptions")
    func engineErrorDescriptions() {
        let errors: [LFMEngineError] = [
            .modelNotLoaded,
            .engineBusy,
            .modelLoadFailed("test"),
            .adapterLoadFailed(adapter: "a", reason: "r"),
            .inferenceTimeout(durationMs: 5000),
            .inferenceFailed("fail"),
            .invalidPromptFormat("bad"),
            .insufficientMemory(requiredMB: 500, availableMB: 200),
        ]

        for error in errors {
            #expect(error.errorDescription != nil)
            #expect(error.errorDescription?.isEmpty == false)
        }
    }

    @Test("GenerationParams uses greedy defaults")
    func generationParamsDefaults() {
        let params = GenerationParams(prompt: "test")

        #expect(params.maxTokens == 256)
        #expect(params.temperature == 0.0)
        #expect(params.stopSequences.isEmpty)
    }
}
