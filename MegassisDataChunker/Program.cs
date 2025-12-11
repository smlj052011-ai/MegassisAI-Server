using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Xml;
using UglyToad.PdfPig;
using UglyToad.PdfPig.Content;

namespace MegassisDataProcessor
{
    // Fix CS8618: Initialize properties to avoid null warnings
    public class KnowledgeChunk
    {
        public string Id { get; set; } = Guid.NewGuid().ToString();
        public string SourceFile { get; set; } = string.Empty;
        public string Content { get; set; } = string.Empty;
    }

    class Program
    {
        static void Main(string[] args)
        {
            // --- CONFIGURATION ---
            string pdfDirectory = @"E:\MEGASSISAI\MegassisRawData";
            string outputJsonPath = @"E:MEGASSISAI\megassis_knowledge.json";
            // ---------------------

            Console.WriteLine("--- MEGASSIS: Advanced Data Processor ---");

            if (!Directory.Exists(pdfDirectory))
            {
                Console.WriteLine($"Error: Directory not found at {pdfDirectory}");
                return;
            }

            var chunks = new List<KnowledgeChunk>();
            var files = Directory.GetFiles(pdfDirectory, "*.pdf", SearchOption.AllDirectories);

            Console.WriteLine($"Found {files.Length} PDF files. Starting robust processing...");

            foreach (var file in files)
            {
                Console.Write($"Processing: {Path.GetFileName(file)}... ");
                try
                {
                    // 1. Extract text using Y-coordinate analysis to preserve lines
                    string rawText = ExtractTextSmart(file);

                    // 2. Create chunks based on paragraphs
                    var fileChunks = CreateSmartChunks(rawText, Path.GetFileName(file));

                    chunks.AddRange(fileChunks);
                    Console.WriteLine($"[OK] -> {fileChunks.Count} chunks");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[Error]: {ex.Message}");
                }
            }

            // Save to JSON
            var json = JsonConvert.SerializeObject(chunks, Newtonsoft.Json.Formatting.Indented);
            File.WriteAllText(outputJsonPath, json);

            Console.WriteLine($"\nDONE! Saved {chunks.Count} chunks to: {outputJsonPath}");
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }

        // Fixed method: Uses built-in GetWords() and handles layout manually
        // to avoid Delegate/Type compilation errors.
        static string ExtractTextSmart(string path)
        {
            StringBuilder fullText = new StringBuilder();

            using (var pdf = PdfDocument.Open(path))
            {
                foreach (var page in pdf.GetPages())
                {
                    // GetWords() automatically handles spacing (fixes "France0.12")
                    // It separates text based on visual distance.
                    var words = page.GetWords();

                    if (!words.Any()) continue;

                    double currentY = words.First().BoundingBox.Bottom;

                    foreach (var word in words)
                    {
                        // Check if we jumped to a new line (Y coordinate changed significantly)
                        // PDFs use Bottom-Left origin, so different Y means new line.
                        // We use a small threshold (e.g., 4 points) to detect line changes.
                        if (Math.Abs(word.BoundingBox.Bottom - currentY) > 4)
                        {
                            fullText.AppendLine(); // Insert Newline
                            currentY = word.BoundingBox.Bottom;
                        }

                        fullText.Append(word.Text);
                        fullText.Append(" "); // Space between words
                    }

                    // End of page
                    fullText.Append("\n\n");
                }
            }

            return fullText.ToString();
        }

        static List<KnowledgeChunk> CreateSmartChunks(string text, string fileName)
        {
            var results = new List<KnowledgeChunk>();

            // 1. Clean the text
            // Normalize line endings
            string clean = text.Replace("\r\n", "\n").Replace("\r", "\n");

            // Remove Page markers 
            clean = Regex.Replace(clean, @"Page\s+\d+(\s+of\s+\d+)?", "", RegexOptions.IgnoreCase);

            // 2. Split by Double Newlines (Paragraphs)
            string[] paragraphs = clean.Split(new[] { "\n\n" }, StringSplitOptions.RemoveEmptyEntries);

            StringBuilder currentChunk = new StringBuilder();

            // Chunk size limit ~1000 characters
            int targetChunkSize = 1000;

            foreach (var para in paragraphs)
            {
                string p = para.Trim();
                if (string.IsNullOrWhiteSpace(p)) continue;

                // Skip simple page numbers (single digits or small numbers)
                if (p.Length < 5 && char.IsDigit(p[0])) continue;

                // If adding this paragraph makes the chunk too big, save current and start new
                if (currentChunk.Length + p.Length > targetChunkSize && currentChunk.Length > 200)
                {
                    results.Add(new KnowledgeChunk
                    {
                        SourceFile = fileName,
                        Content = currentChunk.ToString().Trim()
                    });
                    currentChunk.Clear();
                }

                currentChunk.AppendLine(p);
                currentChunk.AppendLine();
            }

            if (currentChunk.Length > 0)
            {
                results.Add(new KnowledgeChunk
                {
                    SourceFile = fileName,
                    Content = currentChunk.ToString().Trim()
                });
            }

            return results;
        }
    }
}