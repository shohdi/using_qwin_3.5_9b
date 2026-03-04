using System.Text.Json;
using Microsoft.Extensions.Options;

var builder = WebApplication.CreateBuilder(args);
const string ChatSessionKey = "chat-history";

builder.Services.AddRazorPages();
builder.Services.AddDistributedMemoryCache();
builder.Services.AddSession(options =>
{
    options.Cookie.Name = "QwenChatWeb.Session";
    options.Cookie.HttpOnly = true;
    options.Cookie.IsEssential = true;
    options.IdleTimeout = TimeSpan.FromHours(8);
});

builder.Services.Configure<PythonApiOptions>(builder.Configuration.GetSection(PythonApiOptions.SectionName));
builder.Services.AddHttpClient<PythonApiClient>((sp, client) =>
{
    var options = sp.GetRequiredService<IOptions<PythonApiOptions>>().Value;
    client.BaseAddress = new Uri(options.BaseUrl);
    client.Timeout = TimeSpan.FromMinutes(5);
});

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseRouting();
app.UseSession();
app.UseAuthorization();

app.MapGet("/api/chat/history", (HttpContext context) =>
{
    var history = GetChatHistory(context.Session);
    return Results.Ok(new ChatHistoryResponse(history));
});

app.MapPost("/api/chat/send", async (
    ChatSendRequest request,
    HttpContext context,
    PythonApiClient client,
    IOptions<PythonApiOptions> options,
    CancellationToken cancellationToken) =>
{
    var message = request.Message?.Trim();
    if (string.IsNullOrWhiteSpace(message))
    {
        return Results.BadRequest(new { error = "Message cannot be empty." });
    }

    var history = GetChatHistory(context.Session);
    history.Add(new ChatMessage("user", message));

    try
    {
        var assistantReply = await client.GetAssistantReplyAsync(history, options.Value.Model, cancellationToken);
        history.Add(new ChatMessage("assistant", assistantReply));

        // Keep session bounded to prevent unlimited memory growth.
        //if (history.Count > 60)
        //{
        //    history = history.Skip(history.Count - 60).ToList();
        //}

        SaveChatHistory(context.Session, history);
        return Results.Ok(new ChatHistoryResponse(history));
    }
    catch (Exception ex)
    {
        if (history.Count > 0)
        {
            history.RemoveAt(history.Count - 1);
        }

        SaveChatHistory(context.Session, history);
        return Results.Problem(
            title: "Python API request failed",
            detail: ex.Message,
            statusCode: StatusCodes.Status502BadGateway);
    }
});

app.MapPost("/api/chat/new", (HttpContext context) =>
{
    context.Session.Remove(ChatSessionKey);
    return Results.Ok(new ChatHistoryResponse(Array.Empty<ChatMessage>()));
});

app.MapStaticAssets();
app.MapRazorPages().WithStaticAssets();

app.Run();

static List<ChatMessage> GetChatHistory(ISession session)
{
    var json = session.GetString(ChatSessionKey);
    if (string.IsNullOrWhiteSpace(json))
    {
        return [];
    }

    return JsonSerializer.Deserialize<List<ChatMessage>>(json) ?? [];
}

static void SaveChatHistory(ISession session, List<ChatMessage> history)
{
    session.SetString(ChatSessionKey, JsonSerializer.Serialize(history));
}

public sealed record ChatMessage(string Role, string Content);
public sealed record ChatSendRequest(string Message);
public sealed record ChatHistoryResponse(IReadOnlyList<ChatMessage> Messages);
