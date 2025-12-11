using MegassisServer.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Register our AI Brain service
builder.Services.AddSingleton<MegassisBrainService>();

var app = builder.Build();

// Initialize the Brain (Load the heavy model into memory)
var brain = app.Services.GetRequiredService<MegassisBrainService>();
await brain.InitializeAsync();

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