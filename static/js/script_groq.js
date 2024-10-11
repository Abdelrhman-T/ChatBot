function sendMessage() {
    const inputField = document.getElementById('userInput');
    const messageText = inputField.value.trim();

    if (messageText) {
        // Display the user's message
        const userMessage = document.createElement('div');
        userMessage.classList.add('message');
        userMessage.textContent = "You: " + messageText;
        document.getElementById('chatMessages').appendChild(userMessage);

        // Make a POST request to the server
        fetch('/groq_answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: messageText })
        })
        .then(response => response.json())
        .then(data => {
            // Determine the emoji based on sentiment
            let emoji = '';
            switch (data.sentiment) {
                case 'pos':
                    emoji = '😊';
                    break;
                case 'neg':
                    emoji = '😞';
                    break;
                case 'neu':
                    emoji = '😐';
                    break;
            }

            // Display the response with emoji
            const botMessage = document.createElement('div');
            botMessage.classList.add('message');
            botMessage.textContent = "Bot: " + data.answer + " " + emoji;
            document.getElementById('chatMessages').appendChild(botMessage);

            // Scroll to the bottom of the messages container
            document.getElementById('chatMessages').scrollTop = document.getElementById('chatMessages').scrollHeight;
        })
        .catch(error => console.error('Error:', error));

        // Clear the input field
        inputField.value = '';
    }
}
