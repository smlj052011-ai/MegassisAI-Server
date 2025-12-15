using LLama;
using LLama.Abstractions;
using LLama.Common;
using LLama.Exceptions;
using System.Text.Json;
using System.Text;
using System.Threading.Tasks;

namespace MegassisServer.Services
{
    // Implementation uses a true Singleton pattern for weights, 
    // but creates a new Executor per request to resolve batch size conflicts.
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
        private readonly int _batchSize = 128; // Reduced batch size
        private readonly int _maxRAGContextChars = 500;

        public MegassisBrainService()
        {
            // --- Configuration: Paths and Prompts (Lightweight setup in constructor) ---
            _modelPath = Path.Combine(AppContext.BaseDirectory, "Models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
            _knowledgePath = Path.Combine(AppContext.BaseDirectory, "Data", "megassis_knowledge.json");

            // Load Knowledge Base (This is fast, so it stays in the constructor)
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
        /// Runs inference by creating a new Executor (and temporary Context) per request to enforce parameters.
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
            };

            // Use the weights and params to create a new StatelessExecutor and Context for THIS request.
            // This is the CRITICAL change to force batch size enforcement.
            LLamaContext? context = null;
            StatelessExecutor? executor = null;

            try
            {
                // Create Context (memory/KV cache) for this specific request
                context = _weights.CreateContext(_modelParams);
                executor = new StatelessExecutor(_weights, _modelParams);

                // 1. Retrieval Augmented Generation (RAG)
                var relevantChunks = RAG.RetrieveRelevantChunks(_knowledgeChunks, userQuestion, count: 2);

                string finalUserQuery;
                // ... (RAG context building logic is identical)
                if (relevantChunks.Any())
                {
                    var contextBuilder = new StringBuilder();
                    foreach (var chunk in relevantChunks)
                    {
                        if (contextBuilder.Length + chunk.Content.Length + 50 > _maxRAGContextChars)
                        {
                            break;
                        }
                        contextBuilder.AppendLine(chunk.Content);
                        contextBuilder.AppendLine("---");
                    }

                    string contextText = contextBuilder.ToString().Trim();
                    if (contextText.Length > _maxRAGContextChars)
                    {
                        contextText = contextText.Substring(0, _maxRAGContextChars) + "... [TRUNCATED]";
                    }

                    // PROMPT OPTIMIZATION
                    finalUserQuery = $"CONTEXT: {contextText}\n\nUSER QUESTION: {userQuestion}";
                }
                else
                {
                    finalUserQuery = userQuestion;
                }

                // 2. Manually format full prompt (TinyLlama Chat template)
                var fullPrompt = $"<|system|>{_systemPrompt}<|end|>\n<|user|>{finalUserQuery}<|end|>\n<|assistant|>";

                // 3. Run the inference
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
                // The error now means the prompt itself is too long for the 2048 token context.
                return "I ran into a context limit. The current prompt is too long for the model's 2048 token context. Please try simplifying your question.";
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERROR] Unhandled exception in AskMegassis: {ex.GetType().Name}: {ex.Message}");
                Console.WriteLine(ex.ToString());

                return "A server error occurred while trying to generate the response.";
            }
            finally
            {
               
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