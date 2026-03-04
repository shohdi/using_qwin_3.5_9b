using System.Net.Http.Json;
using System.Text.Json;

public sealed class PythonApiClient(HttpClient httpClient)
{
    public async Task<string> GetAssistantReplyAsync(IReadOnlyList<ChatMessage> messages, string model, CancellationToken cancellationToken)
    {
        var payload = new
        {
            model,
            messages = messages.Select(m => new { role = m.Role, content = m.Content }).ToArray(),
            stream = false
        };

        using var response = await httpClient.PostAsJsonAsync("/v1/chat/completions", payload, cancellationToken);
        var body = await response.Content.ReadAsStringAsync(cancellationToken);

        if (!response.IsSuccessStatusCode)
        {
            throw new InvalidOperationException($"Status {(int)response.StatusCode}: {body}");
        }

        using var document = JsonDocument.Parse(body);
        if (document.RootElement.TryGetProperty("choices", out var choices) && choices.GetArrayLength() > 0)
        {
            var content = choices[0].GetProperty("message").GetProperty("content").GetString();
            if (!string.IsNullOrWhiteSpace(content))
            {
                return content;
            }
        }

        throw new InvalidOperationException("Python API returned an unexpected response shape.");
    }
}
