using System.Text;
using System.Text.Json;
using System.Net.Http.Headers;

namespace MegassisServer.Services
{
    // This service is now an HTTP client for the Ollama API, 
    // replacing the problematic LLamaSharp library.
    public class MegassisBrainService
    {
        private readonly HttpClient _httpClient;
        private readonly IReadOnlyList<KnowledgeChunk> _knowledgeChunks;

        // --- Configuration Constants ---
        private const string OllamaEndpoint = "http://localhost:11434/api/generate";
        // *** CRITICAL CHANGE: Upgrading to Llama 3 8B Instruct for OCI Deployment ***
        private const string ModelName = "llama3:8b";

        // --- UPDATED SYSTEM PROMPT for Educational Walkthroughs ---
        private const string SystemPrompt = "You are Megassis, an educational AI assistant for KV Class 9 students, specializing in NEP 2020 and Viksit Bharat 2047. Your primary role is to guide students and provide walkthroughs, not direct answers. Use a patient, encouraging tone. Only use the provided 'CONTEXT: '. If the context has no answer, gently suggest where they might find more information (e.g., 'Check your official textbook'). Never give a direct solution or final answer. Focus on guiding principles and steps.";

        private const int MaxRAGContextChars = 500;

        public MegassisBrainService(HttpClient httpClient)
        {
            _httpClient = httpClient;
            // Timeout set to 5 minutes (300s) for robust initial load on OCI.
            _httpClient.Timeout = TimeSpan.FromSeconds(300);

            // Load Knowledge Base (RAG logic remains the same)
            var knowledgePath = Path.Combine(AppContext.BaseDirectory, "Data", "megassis_knowledge.json");
            try
            {
                var jsonString = File.ReadAllText(knowledgePath);
                _knowledgeChunks = JsonSerializer.Deserialize<IReadOnlyList<KnowledgeChunk>>(jsonString) ?? new List<KnowledgeChunk>();
                Console.WriteLine("[INFO] Knowledge base loaded.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERROR] Failed to load knowledge base: {ex.Message}");
                _knowledgeChunks = new List<KnowledgeChunk>();
            }
        }

        public Task InitializeAsync()
        {
            Console.WriteLine("[INFO] Initialization complete (Ollama dependency).");
            return Task.CompletedTask;
        }

        /// <summary>
        /// Runs inference by calling the external Ollama service.
        /// </summary>
        public async Task<string> AskMegassis(string userQuestion)
        {
            try
            {
                // 1. Retrieval Augmented Generation (RAG)
                var relevantChunks = RAG.RetrieveRelevantChunks(_knowledgeChunks, userQuestion, count: 2);
                string finalUserQuery;

                if (relevantChunks.Any())
                {
                    var contextBuilder = new StringBuilder();
                    foreach (var chunk in relevantChunks)
                    {
                        string chunkContent = chunk.Content;
                        if (chunkContent.Length > MaxRAGContextChars)
                        {
                            chunkContent = chunkContent.Substring(0, MaxRAGContextChars) + "...";
                        }
                        contextBuilder.AppendLine(chunkContent);
                        contextBuilder.AppendLine("---");
                    }
                    string contextText = contextBuilder.ToString().Trim();
                    finalUserQuery = $"CONTEXT: {contextText}\n\nSTUDENT QUESTION: {userQuestion}";
                }
                else
                {
                    finalUserQuery = userQuestion;
                }

                // 2. Format the prompt using the chat template (Llama 3 template)
                // Llama 3 uses a specific system/user/assistant template.
                var fullPrompt = $"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{SystemPrompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{finalUserQuery}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n";

                // 3. Construct the JSON payload for Ollama
                var requestBody = new
                {
                    model = ModelName,
                    prompt = fullPrompt,
                    stream = false,
                    options = new
                    {
                        temperature = 0.5,
                        // Increased context window is better for Llama 3
                        num_ctx = 4096,
                        seed = 42
                    }
                };

                var content = new StringContent(JsonSerializer.Serialize(requestBody), Encoding.UTF8, "application/json");

                // 4. Send the request to Ollama
                var response = await _httpClient.PostAsync(OllamaEndpoint, content);

                if (!response.IsSuccessStatusCode)
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    Console.WriteLine($"[OLLAMA ERROR] Status: {response.StatusCode}. Response: {errorContent}");
                    return $"Ollama Error: Status {response.StatusCode}. Details: {errorContent.Trim()}";
                }

                // 5. Process the JSON response
                var responseJson = await response.Content.ReadAsStringAsync();

                // Ollama response is a JSON object with a 'response' field
                using var doc = JsonDocument.Parse(responseJson);
                var answerElement = doc.RootElement.GetProperty("response");

                return answerElement.GetString()?.Trim() ?? "LLM returned an empty response.";
            }
            catch (HttpRequestException ex)
            {
                // --- ENHANCED ERROR REPORTING ---
                Console.WriteLine($"[OLLAMA CONNECTION ERROR] Could not connect to Ollama: {ex.Message}");
                return $"ERROR: Could not connect to Ollama at {OllamaEndpoint}. Please ensure 'ollama.exe serve' is running in a separate terminal. Connection detail: {ex.Message}";
            }
            catch (Exception ex)
            {
                // Catch any other errors (like JSON parsing issues after a long wait)
                Console.WriteLine($"[ERROR] Unhandled exception in AskMegassis: {ex.GetType().Name}: {ex.Message}");
                Console.WriteLine(ex.ToString());

                return "A server error occurred while processing the request.";
            }
        }

        // --- Data Models and RAG Helper (Unchanged) ---

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