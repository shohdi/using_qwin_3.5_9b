const chatLog = document.getElementById("chatLog");
const statusText = document.getElementById("statusText");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");

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

function renderMessages(messages) {
    if (!messages || messages.length === 0) {
        chatLog.innerHTML = '<div class="message assistant">How can I help you today?</div>';
        return;
    }

    chatLog.innerHTML = messages
        .map(m => `<div class="message ${m.role}">${escapeHtml(m.content || "")}</div>`)
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
    setStatus("Thinking...");
    sendBtn.disabled = true;

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
        setStatus("Ready");
    } catch (error) {
        console.error(error);
        setStatus("Error");
        renderMessages([{ role: "assistant", content: "Request failed. Check that Python API is running on port 8001." }]);
    } finally {
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
