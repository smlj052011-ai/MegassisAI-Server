using LLama;
using LLama.Common;
using Newtonsoft.Json;
using System.Text;

namespace MegassisServer.Services
{
    public class KnowledgeChunk
    {
        public string SourceFile { get; set; }
        public string Content { get; set; }
    }

    public class MegassisBrainService
    {
        private LLamaWeights _model;
        private StatelessExecutor _executor;
        private List<KnowledgeChunk> _knowledgeBase;

        private const string ModelFileName = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
        private const string DataFileName = "megassis_knowledge.json";

        public async Task InitializeAsync()
        {
            // 1. Construct safe paths
            string basePath = AppDomain.CurrentDomain.BaseDirectory;
            string modelPath = Path.Combine(basePath, "Models", ModelFileName);
            string dataPath = Path.Combine(basePath, "Data", DataFileName);

            // 2. Load the Knowledge Base (JSON)
            if (File.Exists(dataPath))
            {
                var json = await File.ReadAllTextAsync(dataPath);
                _knowledgeBase = JsonConvert.DeserializeObject<List<KnowledgeChunk>>(json) ?? new List<KnowledgeChunk>();
            }
            else
            {
                _knowledgeBase = new List<KnowledgeChunk>();
                Console.WriteLine($"WARNING: Could not find data at {dataPath}");
            }

            // 3. Load the AI Model
            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"Model file not found at {modelPath}.");
            }

            var modelParams = new ModelParams(modelPath)
            {
                ContextSize = 2048,
                GpuLayerCount = 0
            };

            _model = LLamaWeights.LoadFromFile(modelParams);
            _executor = new StatelessExecutor(_model, modelParams);
        }

        public async Task<string> AskMegassis(string userQuestion)
        {
            // 1. RAG (Retrieval remains the same)
            var keywords = userQuestion.Split(' ')
                .Where(k => k.Length > 3)
                .Select(k => k.Trim())
                .ToList();

            var relevantDocs = _knowledgeBase
                .Where(c => keywords.Any(k => c.Content.Contains(k, StringComparison.OrdinalIgnoreCase)))
                .OrderByDescending(c => keywords.Count(k => c.Content.Contains(k, StringComparison.OrdinalIgnoreCase)))
                .Take(3)
                .ToList();

            string contextText = string.Join("\n\n", relevantDocs.Select(d => $"From {d.SourceFile}: {d.Content}"));

            // 2. Behavioral Fine-Tuning via Prompt Engineering
            var prompt = $@"<|system|>
You are MEGASSIS, an expert, encouraging, and patient tutor for Class 9 students. 
Your goal is to guide the student toward understanding, not just giving a simple answer.

**BEHAVIOR MANDATES:**
1.  **Walkthrough Style:** DO NOT give direct, final answers. Always provide a GUIDED WALKTHROUGH or explanation broken down into numbered steps or bullet points.
2.  **Tone:** Use simple, pedagogical language suitable for a 9th grader. Be highly encouraging.
3.  **Thematic Integration:** If the question relates to education, future goals, or curriculum, you MUST briefly explain how the topic connects to the relevance of NEP 2020 or Vikshit Bharat 2047, using the context provided.
4.  **Sourcing:** Use ONLY the provided context below. If the answer is not in the context, say: 'That's a fantastic question! Based on my current documents, I don't have enough information for a full walkthrough, but let's see what we can find next.'

Context:
{contextText}
</s>
<|user|>
{userQuestion}
</s>
<|assistant|>";

            // 3. Inference
            var inferenceParams = new InferenceParams()
            {
                MaxTokens = 350, // Increased MaxTokens to allow for longer walkthroughs
                AntiPrompts = new List<string> { "<|user|>" }
            };

            var result = new StringBuilder();

            await foreach (var text in _executor.InferAsync(prompt, inferenceParams))
            {
                result.Append(text);
            }

            return result.ToString();
        }
    }
}