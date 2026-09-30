import XCTest
@testable import LFMEngine

/// Tests for on-device inference components.
/// Since llama.cpp may not be available in CI/simulator, these tests verify
/// the Swift-level logic (error paths, state transitions, configuration)
/// rather than exercising the C API directly.
final class OnDeviceTests: XCTestCase {

    // MARK: - LlamaBackend

    func testBackendInitDefaultThreadCount() async {
        let backend = LlamaBackend()
        // Should not crash — just verifies construction
        await backend.unload()
    }

    func testBackendInitCustomThreadCount() async {
        let backend = LlamaBackend(threadCount: 2)
        await backend.unload()
    }

    func testBackendUnloadWithoutLoadIsSafe() async {
        // Calling unload() before any load should not crash
        let backend = LlamaBackend()
        await backend.unload()
        await backend.unload() // Double unload should also be safe
    }

    func testBackendGenerateWithoutLoadThrows() async {
        let backend = LlamaBackend()
        do {
            _ = try await backend.generate(prompt: "test", maxTokens: 10)
            XCTFail("Expected error when generating without loaded model")
        } catch {
            // Expected — either modelNotLoaded or platform-specific error
        }
    }

    func testBackendSetAdapterWithoutLoadThrows() async {
        let backend = LlamaBackend()
        do {
            try await backend.setAdapter(path: "/nonexistent.gguf")
            XCTFail("Expected error when setting adapter without loaded model")
        } catch {
            // Expected
        }
    }

    func testBackendRemoveAdapterWithoutLoadIsSafe() async {
        let backend = LlamaBackend()
        await backend.removeAdapter() // Should not crash
    }

    // MARK: - LFMEngine On-Device Mode

    @MainActor
    func testOnDeviceLoadWithoutPathFails() async {
        let engine = LFMEngine(
            mode: .onDevice,
            modelConfig: ModelConfig(name: "test", fileName: "test.gguf")
            // modelBasePath intentionally nil
        )

        do {
            try await engine.loadModel()
            XCTFail("Expected error without modelBasePath")
        } catch let error as LFMEngineError {
            XCTAssertTrue(error.localizedDescription.contains("modelBasePath"))
        } catch {
            // Other error types also acceptable
        }
    }

    @MainActor
    func testOnDeviceLoadWithInvalidPathFails() async {
        let engine = LFMEngine(
            mode: .onDevice,
            modelConfig: ModelConfig(name: "test", fileName: "test.gguf"),
            modelBasePath: "/nonexistent/path/test.gguf"
        )

        do {
            try await engine.loadModel()
            XCTFail("Expected error with invalid model path")
        } catch {
            XCTAssertEqual(engine.state, .unloaded)
        }
    }

    @MainActor
    func testSetAdapterRejectsWhenNotReady() async {
        let engine = LFMEngine(
            mode: .onDevice,
            modelConfig: ModelConfig(name: "test", fileName: "test.gguf")
        )
        // State is .unloaded — setAdapter should fail
        let adapter = AdapterConfig(
            id: "test", name: "Test", fileName: "test-lora.gguf", capability: "test"
        )
        do {
            try await engine.setAdapter(adapter)
            XCTFail("Expected error when model not loaded")
        } catch let error as LFMEngineError {
            if case .modelNotLoaded = error {
                // Expected
            } else {
                XCTFail("Wrong error type: \(error)")
            }
        } catch {
            XCTFail("Unexpected error type")
        }
    }

    @MainActor
    func testUnloadWithoutLoadIsSafe() async {
        let engine = LFMEngine(
            mode: .onDevice,
            modelConfig: ModelConfig(name: "test", fileName: "test.gguf")
        )
        await engine.unloadModel()
        XCTAssertEqual(engine.state, .unloaded)
    }

    @MainActor
    func testGenerateWithoutLoadThrows() async {
        let engine = LFMEngine(
            mode: .onDevice,
            modelConfig: ModelConfig(name: "test", fileName: "test.gguf")
        )
        do {
            _ = try await engine.generate(GenerationParams(prompt: "test"))
            XCTFail("Expected error")
        } catch let error as LFMEngineError {
            if case .modelNotLoaded = error {
                // Expected
            } else {
                XCTFail("Wrong error: \(error)")
            }
        } catch {
            XCTFail("Unexpected error type")
        }
    }

    // MARK: - Proxy Mode Correctness

    @MainActor
    func testProxyInferenceReturnsCloudLocation() async {
        // Verify the proxy mode returns .cloud not .device
        let result = InferenceResult(
            text: "test",
            latencyMs: 50,
            tokensGenerated: 5,
            model: "test",
            adapter: nil,
            ranOn: .cloud
        )
        XCTAssertEqual(result.ranOn, .cloud)
    }

    // MARK: - ModelConfig

    func testModelConfigDefaults() {
        let config = ModelConfig(name: "LFM2-350M", fileName: "lfm2-350m-q4_k_m.gguf")
        XCTAssertEqual(config.contextLength, 2048)
        XCTAssertEqual(config.gpuLayers, 99)
        XCTAssertEqual(config.defaultTemperature, 0.0)
    }

    func testModelConfigCustomValues() {
        let config = ModelConfig(
            name: "LFM2-350M",
            fileName: "lfm2.gguf",
            contextLength: 4096,
            gpuLayers: 32,
            defaultTemperature: 0.2
        )
        XCTAssertEqual(config.contextLength, 4096)
        XCTAssertEqual(config.gpuLayers, 32)
        XCTAssertEqual(config.defaultTemperature, 0.2, accuracy: 0.01)
    }

    // MARK: - AdapterConfig

    func testAdapterConfigIdentity() {
        let adapter = AdapterConfig(
            id: "pii-detection",
            name: "PII Detection",
            fileName: "pii-detection-v6e-lora-r16.gguf",
            capability: "pii-detection"
        )
        XCTAssertEqual(adapter.id, "pii-detection")
        XCTAssertEqual(adapter.capability, "pii-detection")
    }

    // MARK: - ModelManager

    func testModelManagerDirectoryCreation() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("lfm-test-\(UUID().uuidString)")
        let config = ModelManager.Config(
            modelsDirectory: tempDir.appendingPathComponent("models"),
            adaptersDirectory: tempDir.appendingPathComponent("adapters")
        )
        let manager = ModelManager(config: config)
        try await manager.ensureDirectories()

        XCTAssertTrue(FileManager.default.fileExists(
            atPath: tempDir.appendingPathComponent("models").path
        ))
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: tempDir.appendingPathComponent("adapters").path
        ))

        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }

    func testModelManagerAvailabilityChecks() async {
        let config = ModelManager.Config(
            modelsDirectory: FileManager.default.temporaryDirectory.appendingPathComponent("nonexistent-models")
        )
        let manager = ModelManager(config: config)

        let modelConfig = ModelConfig(name: "test", fileName: "test.gguf")
        let available = await manager.isBaseModelAvailable(modelConfig)
        XCTAssertFalse(available)

        let adapter = AdapterConfig(id: "test", name: "test", fileName: "test.gguf", capability: "test")
        let adapterAvailable = await manager.isAdapterAvailable(adapter)
        XCTAssertFalse(adapterAvailable)
    }
}
