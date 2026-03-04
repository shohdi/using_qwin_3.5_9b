const chatLog = document.getElementById("chatLog");
const statusText = document.getElementById("statusText");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
let responseTimerId = null;
let responseStartedAt = 0;

function setStatus(text) {
    statusText.textContent = text;
}

function escapeHtml(value) {
    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function formatMessageBody(content) {
    return `<div class="message-body">${escapeHtml(content || "")}</div>`;
}

function formatAssistantContent(content) {
    const text = content || "";
    const thinkRegex = /<think>([\s\S]*?)<\/think>/gi;
    const thinkBlocks = [];

    let match;
    while ((match = thinkRegex.exec(text)) !== null) {
        const thinkingText = (match[1] || "").trim();
        if (thinkingText.length > 0) {
            thinkBlocks.push(thinkingText);
        }
    }

    const finalText = text.replace(thinkRegex, "").trim();

    let html = "";
    for (const thinkingText of thinkBlocks) {
        html += `<details class="thinking-block"><summary>Thinking</summary><div class="thinking-content">${escapeHtml(thinkingText)}</div></details>`;
    }

    if (finalText.length > 0 || thinkBlocks.length === 0) {
        html += formatMessageBody(finalText.length > 0 ? finalText : text);
    }

    return html;
}

function formatMessageHtml(message) {
    if (message.role === "assistant") {
        return formatAssistantContent(message.content);
    }

    return formatMessageBody(message.content || "");
}

function updateResponseElapsedStatus() {
    if (!responseStartedAt) {
        return;
    }

    const elapsedSeconds = Math.floor((Date.now() - responseStartedAt) / 1000);
    setStatus(`Thinking... ${elapsedSeconds}s`);
}

function startResponseTimer() {
    stopResponseTimer();
    responseStartedAt = Date.now();
    updateResponseElapsedStatus();
    responseTimerId = setInterval(updateResponseElapsedStatus, 1000);
}

function stopResponseTimer(finalStatusText) {
    if (responseTimerId !== null) {
        clearInterval(responseTimerId);
        responseTimerId = null;
    }

    responseStartedAt = 0;
    if (finalStatusText) {
        setStatus(finalStatusText);
    }
}

function renderMessages(messages) {
    if (!messages || messages.length === 0) {
        chatLog.innerHTML = `<div class="message assistant">${formatMessageBody("How can I help you today?")}</div>`;
        return;
    }

    chatLog.innerHTML = messages
        .map(m => `<div class="message ${m.role}">${formatMessageHtml(m)}</div>`)
        .join("");

    chatLog.scrollTop = chatLog.scrollHeight;
}

async function loadHistory() {
    setStatus("Loading...");

    try {
        const response = await fetch("/api/chat/history");
        if (!response.ok) {
            throw new Error(`Failed with ${response.status}`);
        }

        const payload = await response.json();
        renderMessages(payload.messages);
        setStatus("Ready");
    } catch (error) {
        console.error(error);
        setStatus("Offline");
        renderMessages([{ role: "assistant", content: "Unable to load chat history." }]);
    }
}

function resizeInput() {
    messageInput.style.height = "auto";
    messageInput.style.height = `${Math.min(messageInput.scrollHeight, 200)}px`;
}

async function sendMessage(message) {
    startResponseTimer();
    sendBtn.disabled = true;
    let finalStatusText = "Ready";

    try {
        const response = await fetch("/api/chat/send", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message })
        });

        if (!response.ok) {
            const errorBody = await response.text();
            throw new Error(errorBody || `Failed with ${response.status}`);
        }

        const payload = await response.json();
        renderMessages(payload.messages);
    } catch (error) {
        console.error(error);
        finalStatusText = "Error";
        renderMessages([{ role: "assistant", content: "Request failed. Check that Python API is running on port 8001." }]);
    } finally {
        stopResponseTimer(finalStatusText);
        sendBtn.disabled = false;
        messageInput.focus();
    }
}

chatForm.addEventListener("submit", async event => {
    event.preventDefault();

    const message = messageInput.value.trim();
    if (!message) {
        return;
    }

    messageInput.value = "";
    resizeInput();
    await sendMessage(message);
});

newChatBtn.addEventListener("click", async () => {
    setStatus("Resetting...");

    try {
        const response = await fetch("/api/chat/new", { method: "POST" });
        if (!response.ok) {
            throw new Error(`Failed with ${response.status}`);
        }

        const payload = await response.json();
        renderMessages(payload.messages);
        setStatus("Ready");
    } catch (error) {
        console.error(error);
        setStatus("Error");
    }
});

messageInput.addEventListener("input", resizeInput);

loadHistory();
resizeInput();
