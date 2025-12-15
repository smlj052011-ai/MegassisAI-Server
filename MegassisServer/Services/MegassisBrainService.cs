using LLama;
using LLama.Abstractions;
using LLama.Common;
using LLama.Exceptions;
using System.Text.Json;
using System.Text;
using System.Threading.Tasks;

namespace MegassisServer.Services
{
    public class MegassisBrainService
    {
        // --- Private Fields for Singleton Instance ---
        private LLamaWeights? _weights;
        private LLama.Common.ModelParams? _modelParams;

        // --- Model Configuration Constants ---
        private readonly string _modelPath;
        private readonly string _knowledgePath;
        private readonly IReadOnlyList<KnowledgeChunk> _knowledgeChunks;
        private readonly string _systemPrompt;

        // CRITICAL: Fixed parameters to avoid the 'NoKvSlot' error
        private readonly int _contextSize = 2048;
        private readonly int _batchSize = 128; // We keep this low, but the environment may ignore it.

        // AGGRESSIVE FIX: Dramatically reduce RAG context size to avoid exceeding the fixed 512 slot initial batch.
        // 150 characters is approximately 30-50 tokens.
        private readonly int _maxRAGContextChars = 150;

        public MegassisBrainService()
        {
            // --- Configuration: Paths and Prompts (Lightweight setup in constructor) ---
            _modelPath = Path.Combine(AppContext.BaseDirectory, "Models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
            _knowledgePath = Path.Combine(AppContext.BaseDirectory, "Data", "megassis_knowledge.json");

            // Load Knowledge Base
            try
            {
                var jsonString = File.ReadAllText(_knowledgePath);
                _knowledgeChunks = JsonSerializer.Deserialize<IReadOnlyList<KnowledgeChunk>>(jsonString) ?? new List<KnowledgeChunk>();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERROR] Failed to load knowledge base: {ex.Message}");
                _knowledgeChunks = new List<KnowledgeChunk>();
            }

            _systemPrompt = "You are Megassis, a friendly, helpful, and concise AI assistant. ALWAYS base your response strictly on the provided context, which is prefixed with 'CONTEXT: '. If the context does not contain the answer, state that you do not have information on that topic. Do not make up answers.";
        }

        /// <summary>
        /// Loads the heavy model weights and defines parameters once (called from Program.cs).
        /// </summary>
        public async Task InitializeAsync()
        {
            Console.WriteLine("[INFO] Initializing LLM...");
            try
            {
                // Define Model/Context Parameters ONCE
                _modelParams = new LLama.Common.ModelParams(_modelPath)
                {
                    ContextSize = (uint)_contextSize,
                    BatchSize = (uint)_batchSize,
                    MainGpu = 0 // Use CPU/main GPU
                };

                // Load Model Weights ONCE
                _weights = LLamaWeights.LoadFromFile(_modelParams);

                Console.WriteLine($"[INFO] LLM Weights loaded successfully. Context Size: {_contextSize}, Batch Size: {_batchSize}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[FATAL] Failed to initialize LLM: {ex.Message}");
            }
            await Task.CompletedTask;
        }

        /// <summary>
        /// Runs inference by creating a new InteractiveExecutor per request.
        /// </summary>
        public async Task<string> AskMegassis(string userQuestion)
        {
            if (_weights == null || _modelParams == null)
            {
                return "The LLM service failed to initialize. Check server logs.";
            }

            // Define Inference Parameters
            var inferenceParams = new InferenceParams
            {
                MaxTokens = 512,
                AntiPrompts = new List<string> { "User:", "</s>", "Human:" },
                // Interactive Executor doesn't stream, so we rely on the return.
            };

            LLamaContext? context = null;
            InteractiveExecutor? executor = null;

            try
            {
                // Create Context (memory/KV cache) for this specific request
                context = _weights.CreateContext(_modelParams);

                // FINAL FIX: Switching to InteractiveExecutor
                executor = new InteractiveExecutor(context);

                // 1. Retrieval Augmented Generation (RAG)
                var relevantChunks = RAG.RetrieveRelevantChunks(_knowledgeChunks, userQuestion, count: 1); // Get only 1 chunk now

                string finalUserQuery;
                if (relevantChunks.Any())
                {
                    var contextBuilder = new StringBuilder();
                    foreach (var chunk in relevantChunks)
                    {
                        // Use the new, smaller RAG context limit
                        string chunkContent = chunk.Content;
                        if (chunkContent.Length > _maxRAGContextChars)
                        {
                            chunkContent = chunkContent.Substring(0, _maxRAGContextChars);
                        }
                        contextBuilder.AppendLine(chunkContent);
                        contextBuilder.AppendLine("---");
                    }

                    string contextText = contextBuilder.ToString().Trim();

                    // PROMPT OPTIMIZATION: Inject RAG context directly before the user question
                    finalUserQuery = $"CONTEXT: {contextText}\n\nUSER QUESTION: {userQuestion}";
                }
                else
                {
                    finalUserQuery = userQuestion;
                }

                // 2. Manually format full prompt (TinyLlama Chat template)
                var fullPrompt = $"<|system|>{_systemPrompt}<|end|>\n<|user|>{finalUserQuery}<|end|>\n<|assistant|>";

                // 3. Run the inference (InteractiveExecutor uses RunAsync/InferAsync in older versions)
                // Use InferAsync, which is typically supported by both executor types for stateless prompts.
                var result = new StringBuilder();
                await foreach (var text in executor.InferAsync(fullPrompt, inferenceParams))
                {
                    result.Append(text);
                }

                return result.ToString().Trim();
            }
            catch (LLamaDecodeError ex)
            {
                Console.WriteLine($"[ERROR] LLama Decode Error (NoKvSlot): {ex.Message}");
                // If this still fails, the prompt is too long even with minimized RAG.
                return "I ran into a persistent context memory issue. The model's context is full. Please try asking a single, very short question.";
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERROR] Unhandled exception in AskMegassis: {ex.GetType().Name}: {ex.Message}");
                Console.WriteLine(ex.ToString());

                return "A server error occurred while trying to generate the response.";
            }
            finally
            {
                // We must dispose of the short-lived context created for this request.
                // Executor disposal is not needed (as per v17 fix).
                context?.Dispose();
            }
        }

        // --- KnowledgeChunk definition and RAG helper class remain the same ---

        public class KnowledgeChunk
        {
            public string Id { get; set; } = Guid.NewGuid().ToString();
            public string SourceFile { get; set; } = string.Empty;
            public string Content { get; set; } = string.Empty;
        }

        private static class RAG
        {
            public static IReadOnlyList<KnowledgeChunk> RetrieveRelevantChunks(IReadOnlyList<KnowledgeChunk> knowledgeBase, string query, int count)
            {
                var queryWords = query.ToLower().Split(new[] { ' ', ',', '.', '?', '!' }, StringSplitOptions.RemoveEmptyEntries)
                                      .Where(w => w.Length > 3)
                                      .Distinct();

                var scoredChunks = knowledgeBase
                    .Select(chunk => new
                    {
                        Chunk = chunk,
                        Score = queryWords.Sum(word => chunk.Content.ToLower().Contains(word) ? 1 : 0)
                    })
                    .Where(sc => sc.Score > 0)
                    .OrderByDescending(sc => sc.Score)
                    .Take(count)
                    .Select(sc => sc.Chunk)
                    .ToList();

                return scoredChunks;
            }
        }
    }
}