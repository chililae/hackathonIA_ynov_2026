from app.schemas import ChatMessage, ChatRequest


def test_chat_request_accepts_empty_assistant_placeholder():
    request = ChatRequest(
        messages=[
            ChatMessage(role="user", content="Bonjour"),
            ChatMessage(role="assistant", content=""),
        ]
    )

    assert request.messages[1].content == ""
