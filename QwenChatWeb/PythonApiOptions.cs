public sealed class PythonApiOptions
{
    public const string SectionName = "PythonApi";

    public string BaseUrl { get; init; } = "http://127.0.0.1:8001";
    public string Model { get; init; } = "Qwen/Qwen3.5-9B";
}
