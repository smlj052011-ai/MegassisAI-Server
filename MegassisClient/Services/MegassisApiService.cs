using Newtonsoft.Json;
using System.Text;

namespace MegassisClient.Services
{
    public class BotResponse
    {
        public string Answer { get; set; }
    }

    public class MegassisApiService
    {
        // *** IMPORTANT ***
        // FOR LOCAL TESTING: Use the URL of the running ASP.NET Core server (MegassisServer).
        // The endpoint is /api/Chat.

        // Use HTTPS for Windows/Desktop if the server is running on HTTPS:
        // When deploying, this will change to the OCI public URL.
        private const string BaseUrl = "https://localhost:7123/api/Chat";

        private readonly HttpClient _httpClient;

        public MegassisApiService()
        {
            _httpClient = new HttpClient();
            // Increased timeout for LLM inference (Ollama can take a moment)
            _httpClient.Timeout = TimeSpan.FromMinutes(2);
        }

        public async Task<string> GetAnswerAsync(string question)
        {
            try
            {
                var json = JsonConvert.SerializeObject(new { Question = question });
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                // Post directly to the configured BaseUrl (which includes /api/Chat)
                var response = await _httpClient.PostAsync(BaseUrl, content);

                if (response.IsSuccessStatusCode)
                {
                    var responseString = await response.Content.ReadAsStringAsync();

                    // The server response structure is: { "Answer": "..." }
                    var result = JsonConvert.DeserializeObject<BotResponse>(responseString);
                    return result?.Answer ?? "No response.";
                }

                // Handle non-success status codes (400, 500, etc.)
                var errorBody = await response.Content.ReadAsStringAsync();
                return $"Server Error ({response.StatusCode}): API path or payload may be wrong. Details: {errorBody}";
            }
            catch (Exception ex)
            {
                // Catches network connection issues
                return $"Connection Error: Could not reach the server at {BaseUrl}. Ensure the MegassisServer is running and Ollama is active. Detail: {ex.Message}";
            }
        }
    }
}