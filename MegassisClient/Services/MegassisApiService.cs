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
        // REPLACE THIS URL with the actual Codespace Public URL after deployment.
        // It will look something like: https://your-username-megassisa-xxxx.github.dev/api/chat
        // During local testing, you can use: http://10.0.2.2:5000/api/chat
        private const string BaseUrl = "YOUR_CODE_SPACE_PUBLIC_URL_HERE";
        private readonly HttpClient _httpClient;

        public MegassisApiService()
        {
            _httpClient = new HttpClient();
            // Increased timeout for LLM inference
            _httpClient.Timeout = TimeSpan.FromMinutes(2);
        }

        public async Task<string> GetAnswerAsync(string question)
        {
            // ... (rest of the code is the same)
            try
            {
                var json = JsonConvert.SerializeObject(new { Question = question });
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await _httpClient.PostAsync(BaseUrl, content);

                if (response.IsSuccessStatusCode)
                {
                    var responseString = await response.Content.ReadAsStringAsync();
                    var result = JsonConvert.DeserializeObject<BotResponse>(responseString);
                    return result?.Answer ?? "No response.";
                }
                return "Error: Server not responding or API path is wrong.";
            }
            catch (Exception ex)
            {
                return $"Connection Error: Check the URL in MegassisApiService.cs. Detail: {ex.Message}";
            }
        }
    }
}