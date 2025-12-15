using MegassisServer.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Register our AI Brain service
builder.Services.AddSingleton<MegassisBrainService>();

// CRITICAL: Add HttpClient service for Ollama communication
builder.Services.AddHttpClient();


var app = builder.Build();

// *** IMPORTANT CHANGE ***
// The LLM model is now loaded by an external service (Ollama), 
// so we no longer need the heavy initialization step here.
// The code below is REMOVED/commented out.
/*
var brain = app.Services.GetRequiredService<MegassisBrainService>();
await brain.InitializeAsync();
*/

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();
app.UseAuthorization();
app.MapControllers();

app.Run();