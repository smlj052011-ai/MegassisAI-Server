using MegassisClient.Services;
using System.Collections.ObjectModel;

namespace MegassisClient
{
    // Data class for our list
    public class ChatMessage
    {
        public string Text { get; set; }
        public LayoutOptions Alignment { get; set; } // Left for AI, Right for User
        public Color BoxColor { get; set; }
    }

    public partial class MainPage : ContentPage
    {
        private readonly MegassisApiService _apiService;
        public ObservableCollection<ChatMessage> Messages { get; set; } = new();

        public MainPage()
        {
            InitializeComponent();
            _apiService = new MegassisApiService();
            ChatList.ItemsSource = Messages; // Bind UI to our list

            AddMessage("Hello! I am MEGASSIS. How can I help you?", false);
        }

        private async void OnSendClicked(object sender, EventArgs e)
        {
            string question = QuestionEntry.Text;
            if (string.IsNullOrWhiteSpace(question)) return;

            // 1. Show User's question
            AddMessage(question, true);
            QuestionEntry.Text = "";

            // 2. Get Answer from Server
            string answer = await _apiService.GetAnswerAsync(question);

            // 3. Show Answer
            AddMessage(answer, false);

            // 4. Speak it aloud
            await TextToSpeech.Default.SpeakAsync(answer);
        }

        private void AddMessage(string text, bool isUser)
        {
            var msg = new ChatMessage
            {
                Text = text,
                Alignment = isUser ? LayoutOptions.End : LayoutOptions.Start,
                BoxColor = isUser ? Color.FromArgb("#6200EE") : Color.FromArgb("#333333")
            };

            Messages.Add(msg);
            // Scroll to bottom
            ChatList.ScrollTo(Messages.Last(), position: ScrollToPosition.End, animate: true);
        }
    }
}