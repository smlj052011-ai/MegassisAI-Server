using LLama;
using LLama.Abstractions;
using LLama.Common;
using System.Text.Json;
using System.Text;

namespace MegassisServer.Services
{
    public class MegassisBrainService
    {
        // Model and path constants (defined once)
        private readonly string _modelPath;
        private readonly string _knowledgePath;
        private readonly IReadOnlyList<KnowledgeChunk> _knowledgeChunks;
        private readonly string _systemPrompt;
        private readonly string _retrievalPromptTemplate;
        private readonly int _contextSize = 2048; // Based on TinyLlama spec

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

            // System Prompt defining the AI's role and RAG instruction
            _systemPrompt = "You are Megassis, a friendly, helpful, and concise AI assistant. Your goal is to provide accurate answers based on the context provided. If the context does not contain the answer, state that you do not have information on that topic, and suggest searching the web. Do not make up answers.";

            // Template for injecting retrieved context into the prompt
            _retrievalPromptTemplate = "Based *only* on the following context, answer the question. If the context is irrelevant or insufficient, politely state that you cannot answer the question based on the provided data.\n\nContext:\n{0}\n\nQuestion: {1}";
        }

        public async Task<string> AskMegassis(string userQuestion)
        {
            LLamaWeights? weights = null;
            LLamaContext? context = null;

            try
            {
                // --- 1. CRITICAL FIX: Load Weights and Create Context/Executor on every call ---
                // This guarantees a clean KV cache and fixes the 'NoKvSlot' error.

                // Define Model/Context Parameters (Using ModelParams, as ContextParams seems missing in your LLamaSharp version)
                // The model path is often required in the ModelParams constructor in older versions.
                var modelParams = new LLama.Common.ModelParams(_modelPath)
                {
                    ContextSize = (uint)_contextSize,
                    MainGpu = 0 // Use CPU/main GPU
                };

                // Load Model Weights (resource) - Replacing LLamaModelLoader with static LoadFromFile
                weights = LLamaWeights.LoadFromFile(modelParams);

                // Create Context (memory/KV cache). Note: ModelParams implements IContextParams.
                context = weights.CreateContext(modelParams);

                // Create Executor
                // Note: The StatelessExecutor requires an IContextParams, which ModelParams implements.
                var executor = new StatelessExecutor(weights, modelParams);

                // Define Inference Parameters (using the simplest version)
                var inferenceParams = new InferenceParams
                {
                    MaxTokens = 512, // Limit response length
                    AntiPrompts = new List<string> { "User:", "</s>", "Human:" },
                };

                // 2. Retrieval Augmented Generation (RAG)
                var relevantChunks = RAG.RetrieveRelevantChunks(_knowledgeChunks, userQuestion, count: 2);

                string finalPrompt;
                if (relevantChunks.Any())
                {
                    var contextText = string.Join("\n---\n", relevantChunks.Select(c => c.Content));
                    finalPrompt = string.Format(_retrievalPromptTemplate, contextText, userQuestion);
                }
                else
                {
                    // No relevant context found, ask the question directly
                    finalPrompt = userQuestion;
                }

                // 3. Manually format full prompt (TinyLlama Chat template)
                // <|system|>system prompt<|end|>\n<|user|>user query<|end|>\n<|assistant|>
                var fullPrompt = $"<|system|>{_systemPrompt}<|end|>\n<|user|>{finalPrompt}<|end|>\n<|assistant|>";

                // 4. Run the inference
                var result = new StringBuilder();
                await foreach (var text in executor.InferAsync(fullPrompt, inferenceParams))
                {
                    result.Append(text);
                }

                return result.ToString().Trim();
            }
            catch (Exception ex)
            {
                // Log the exception details
                Console.WriteLine($"[ERROR] Unhandled exception in AskMegassis: {ex.GetType().Name}: {ex.Message}");
                Console.WriteLine(ex.ToString());

                // Check for the specific NoKvSlot error in the message if possible
                if (ex.Message.Contains("NoKvSlot") || ex.Message.Contains("llama_decode failed"))
                {
                    return "I ran into a temporary memory limitation during processing. Please try your question again.";
                }

                return "A server error occurred while trying to generate the response. Please check the server console for details.";
            }
            finally
            {
                // 5. CRITICAL STEP: Dispose resources in the finally block
                // This ensures memory is freed even if an error occurs.
                context?.Dispose();
                weights?.Dispose();
            }
        }

        // --- KnowledgeChunk definition included here to fix CS0234 ---
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