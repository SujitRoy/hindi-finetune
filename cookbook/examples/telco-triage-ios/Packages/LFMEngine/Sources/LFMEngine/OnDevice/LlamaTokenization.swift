import Foundation

#if LLAMA_CPP_AVAILABLE
import llama
#endif

/// Tokenization, detokenization, and sampler utilities for LlamaBackend.
extension LlamaBackend {

    // MARK: - Sampler

    /// Create or replace the sampler. Greedy for temperature <= 0, chain
    /// (temp + dist) otherwise. When `grammar` is non-nil, a
    /// `llama_sampler_init_grammar` is prepended to the chain so the
    /// decoder enforces the GBNF constraint at every token — invalid
    /// tokens are masked out before sampling. The grammar sampler's
    /// state advances automatically via `llama_sampler_accept` because
    /// it's part of the chain.
    ///
    /// **Caching rule**: with no grammar, the sampler is reused across
    /// calls when temperature hasn't changed (perf win on the chat
    /// hot-path). WITH a grammar, the sampler is always rebuilt — the
    /// grammar parser carries position state from the previous decode
    /// and cannot be replayed across prompts. Cheap to rebuild
    /// (sub-millisecond on small grammars).
    func applySampler(temperature: Float, grammar: String? = nil) {
        #if LLAMA_CPP_AVAILABLE
        let needsRebuild = sampler == nil
            || samplerTemperature != temperature
            || grammar != nil
        if !needsRebuild { return }

        if let s = sampler { llama_sampler_free(s) }

        let chain = llama_sampler_chain_init(llama_sampler_chain_default_params())!

        // Grammar must be FIRST in the chain so its token mask is
        // applied BEFORE temperature/dist sampling. Without grammar,
        // we still wrap in a chain so the code path is uniform.
        if let grammar, let model {
            let vocab = llama_model_get_vocab(model)
            grammar.withCString { gPtr in
                "root".withCString { rPtr in
                    if let grammarSampler = llama_sampler_init_grammar(vocab, gPtr, rPtr) {
                        llama_sampler_chain_add(chain, grammarSampler)
                    }
                }
            }
        }

        if temperature <= 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy())
        } else {
            llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature))
            llama_sampler_chain_add(
                chain,
                llama_sampler_init_dist(UInt32.random(in: 0...UInt32.max))
            )
        }

        self.sampler = chain
        samplerTemperature = temperature
        #endif
    }

    // MARK: - Tokenize

    /// Convert a string to llama tokens.
    func tokenize(_ text: String, addBos: Bool) -> [llama_token] {
        #if LLAMA_CPP_AVAILABLE
        guard let model else { return [] }

        let vocab = llama_model_get_vocab(model)
        let utf8 = Array(text.utf8)
        let maxTokens = utf8.count + (addBos ? 1 : 0) + 1

        var tokens = [llama_token](repeating: 0, count: maxTokens)
        let count = llama_tokenize(vocab, utf8.map { Int8(bitPattern: $0) }, Int32(utf8.count),
                                   &tokens, Int32(maxTokens), addBos, true)

        guard count >= 0 else { return [] }
        return Array(tokens.prefix(Int(count)))
        #else
        return []
        #endif
    }

    // MARK: - Detokenize

    /// Convert llama tokens back to a string.
    func detokenize(_ tokens: [llama_token]) -> String {
        #if LLAMA_CPP_AVAILABLE
        guard let model else { return "" }

        let vocab = llama_model_get_vocab(model)
        var result = ""
        var buf = [CChar](repeating: 0, count: 256)

        for token in tokens {
            let nChars = llama_token_to_piece(vocab, token, &buf, Int32(buf.count), 0, true)
            if nChars > 0 && nChars < Int32(buf.count) {
                buf[Int(nChars)] = 0
                result += String(cString: buf)
            }
        }
        return result
        #else
        return ""
        #endif
    }

    /// Detokenize a single token. Used for incremental stop-sequence checking
    /// to avoid O(N²) re-detokenization of the entire generated sequence.
    func detokenizeSingle(_ token: llama_token) -> String {
        #if LLAMA_CPP_AVAILABLE
        guard let model else { return "" }

        let vocab = llama_model_get_vocab(model)
        var buf = [CChar](repeating: 0, count: 256)
        let nChars = llama_token_to_piece(vocab, token, &buf, Int32(buf.count), 0, true)
        if nChars > 0 && nChars < Int32(buf.count) {
            buf[Int(nChars)] = 0
            return String(cString: buf)
        }
        return ""
        #else
        return ""
        #endif
    }
}
