using MegassisServer.Services;
using Microsoft.AspNetCore.Mvc;

namespace MegassisServer.Controllers
{
    // The request object sent by the mobile app
    public class ChatRequest
    {
        public string Question { get; set; }
    }

    [ApiController]
    [Route("api/[controller]")]
    public class ChatController : ControllerBase
    {
        private readonly MegassisBrainService _brain;

        public ChatController(MegassisBrainService brain)
        {
            _brain = brain;
        }

        [HttpPost]
        public async Task<IActionResult> Post([FromBody] ChatRequest request)
        {
            if (string.IsNullOrWhiteSpace(request.Question))
                return BadRequest("Question cannot be empty.");

            var answer = await _brain.AskMegassis(request.Question);
            return Ok(new { Answer = answer });
        }
    }
}