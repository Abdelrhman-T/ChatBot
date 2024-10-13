function sendMessage() {
    const inputField = document.getElementById('userInput');
    const messageText = inputField.value.trim();
    
    if (messageText) {
        // Create a container for the user's message
        const userMessageContainer = document.createElement('div');
        userMessageContainer.classList.add('message-container', 'user-message');

        // User Emoji Avatar
        const userEmoji = document.createElement('div');
        userEmoji.classList.add('emoji-avatar');
        userEmoji.textContent = '🧑🏻'; // User emoji

        // User Message Text
        const userMessage = document.createElement('div');
        userMessage.classList.add('message');
        userMessage.textContent = messageText;

        // Send message to server and handle bot response
        fetch('/groq_answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: messageText })
        })
        .then(response => response.json())
        .then(data => {
            // Get the sentiment emoji from the response
            const sentimentEmoji = data.emoji; // Use the emoji from the response

            // Create a span for the sentiment emoji
            const sentimentEmojiElement = document.createElement('span');
            sentimentEmojiElement.textContent = sentimentEmoji; // Set sentiment emoji
            sentimentEmojiElement.classList.add('emoji'); // Add a class for styling if needed

            // Append user message and sentiment emoji to container
            userMessageContainer.appendChild(userEmoji);
            userMessageContainer.appendChild(userMessage);
            userMessageContainer.appendChild(sentimentEmojiElement); // Add sentiment emoji after user message
            document.getElementById('chatMessages').appendChild(userMessageContainer);

            // Create a container for the bot's response
            const botMessageContainer = document.createElement('div');
            botMessageContainer.classList.add('message-container', 'bot-message');

            // Bot Emoji Avatar
            const botEmoji = document.createElement('div');
            botEmoji.classList.add('emoji-avatar');
            botEmoji.textContent = '🤖'; // Bot emoji

            // Bot Message Text
            const botMessage = document.createElement('div');
            botMessage.classList.add('message');
            botMessage.textContent = data.answer;

            // Append bot emoji and message to container
            botMessageContainer.appendChild(botEmoji);
            botMessageContainer.appendChild(botMessage);
            document.getElementById('chatMessages').appendChild(botMessageContainer);

            // Scroll to the bottom of the messages container
            document.getElementById('chatMessages').scrollTop = document.getElementById('chatMessages').scrollHeight;
        })
        .catch(error => console.error('Error:', error));

        // Clear input field
        inputField.value = '';
    }
}

// Add event listener for the "Enter" key
document.getElementById('userInput').addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // Prevents the default action (like adding a new line)
        sendMessage(); // Call the function to send the message
    }
});
