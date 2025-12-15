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
        // Model and path constants (defined once)
        private readonly string _modelPath;
        private readonly string _knowledgePath;
        private readonly IReadOnlyList<KnowledgeChunk> _knowledgeChunks;
        private readonly string _systemPrompt;

        // Context configuration
        // Matching the fixed size reported by the LLama logs to prevent conflicts.
        private readonly int _contextSize = 2048;

        // AGGRESSIVE LIMIT: Hard cap on RAG context.
        private readonly int _maxRAGContextChars = 500;

        // NOTE: The _batchSize constant is removed, as InteractiveExecutor handles batching dynamically.

        public MegassisBrainService()
        {
            // --- Model and Prompt Configuration ---

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
                _knowledgeChunks = new List<KnowledgeChunk>(); // Initialize empty list on failure
            }

            // MODIFIED SYSTEM PROMPT
            _systemPrompt = "You are Megassis, a friendly, helpful, and concise AI assistant. ALWAYS base your response strictly on the provided context, which is prefixed with 'CONTEXT: '. If the context does not contain the answer, state that you do not have information on that topic. Do not make up answers.";
        }

        public async Task<string> AskMegassis(string userQuestion)
        {
            LLamaWeights? weights = null;
            LLamaContext? context = null;

            try
            {
                // --- 1. Load Weights and Create Context ---

                var modelParams = new LLama.Common.ModelParams(_modelPath)
                {
                    ContextSize = (uint)_contextSize,
                    MainGpu = 0 // Use CPU/main GPU
                };

                // Load Model Weights (resource)
                weights = LLamaWeights.LoadFromFile(modelParams);

                // Create Context (memory/KV cache)
                context = weights.CreateContext(modelParams);

                // CRITICAL FIX: Switch to InteractiveExecutor to bypass batching issues
                var executor = new InteractiveExecutor(context);

                // Define Inference Parameters
                var inferenceParams = new InferenceParams
                {
                    MaxTokens = 512,
                    AntiPrompts = new List<string> { "User:", "</s>", "Human:" },
                };

                // 2. Retrieval Augmented Generation (RAG)
                var relevantChunks = RAG.RetrieveRelevantChunks(_knowledgeChunks, userQuestion, count: 2);

                string finalUserQuery;
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

                    // PROMPT OPTIMIZATION: Inject RAG context directly before the user question
                    finalUserQuery = $"CONTEXT: {contextText}\n\nUSER QUESTION: {userQuestion}";
                }
                else
                {
                    // No relevant context found, ask the question directly
                    finalUserQuery = userQuestion;
                }

                // 3. Manually format full prompt (TinyLlama Chat template)
                // The InteractiveExecutor requires the initial prompt to be sent *without* the assistant tag.
                // The prompt is the combination of System and User query.
                var fullPrompt = $"<|system|>{_systemPrompt}<|end|>\n<|user|>{finalUserQuery}<|end|>";

                // 4. Run the inference
                var result = new StringBuilder();

                // Set the initial prompt (this loads the prompt into the context token by token)
                var firstRun = true;
                await foreach (var text in executor.InferAsync(fullPrompt, inferenceParams))
                {
                    // The first result is often the system prompt/prefix, which we ignore.
                    if (firstRun)
                    {
                        firstRun = false;
                        continue;
                    }
                    result.Append(text);
                }

                return result.ToString().Trim();
            }
            catch (LLamaDecodeError ex)
            {
                // This is the error we are specifically targeting to fix.
                Console.WriteLine($"[ERROR] LLama Decode Error (NoKvSlot): {ex.Message}");
                return "I ran into a persistent context memory limitation (NoKvSlot error). I have implemented the last possible fix, but the model's fixed 2048 token context size is extremely restrictive. Please try simplifying or shortening your question significantly.";
            }
            catch (Exception ex)
            {
                // Log all other exceptions
                Console.WriteLine($"[ERROR] Unhandled exception in AskMegassis: {ex.GetType().Name}: {ex.Message}");
                Console.WriteLine(ex.ToString());

                return "A server error occurred while trying to generate the response. Please check the server console for details.";
            }
            finally
            {
                // 5. CRITICAL STEP: Dispose resources in the finally block
                context?.Dispose();
                weights?.Dispose();
            }
        }

        // --- KnowledgeChunk definition ---
        public class KnowledgeChunk
        {
            public string Id { get; set; } = Guid.NewGuid().ToString();
            public string SourceFile { get; set; } = string.Empty;
            public string Content { get; set; } = string.Empty;
        }

        // --- RAG helper class ---
        private static class RAG
        {
            public static IReadOnlyList<KnowledgeChunk> RetrieveRelevantChunks(IReadOnlyList<KnowledgeChunk> knowledgeBase, string query, int count)
            {
                // Simple keyword-based RAG simulation: Find the top 'count' chunks that contain the most query keywords.
                var queryWords = query.ToLower().Split(new[] { ' ', ',', '.', '?', '!' }, StringSplitOptions.RemoveEmptyEntries)
                                      .Where(w => w.Length > 3) // Ignore very short words
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