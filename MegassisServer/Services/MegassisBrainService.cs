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

            // 2. Prompt (Ready for the next stage of behavioral fine-tuning)
            var prompt = $@"<|system|>
You are MEGASSIS, a helpful assistant for Class 9 students. 
Answer the question using ONLY the provided context. If the answer is not in the context, say 'I don't know based on the documents.'
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
                // *** FIX: Removed the problematic 'SamplingTemperature' property. ***
                // The default temperature of the model will now be used.
                // MaxTokens = 250 is still very important to prevent infinite generation.
                MaxTokens = 250,
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