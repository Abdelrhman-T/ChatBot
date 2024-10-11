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
        fetch('/new_answer', { // Adjust endpoint as necessary
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: messageText })
        })
        .then(response => response.json())
        .then(data => {
            // Display the response from the server
            const botMessage = document.createElement('div');
            botMessage.classList.add('message');
            botMessage.textContent = "Bot: " + data.answer;
            document.getElementById('chatMessages').appendChild(botMessage);
            
            // Scroll to the bottom of the messages container
            document.getElementById('chatMessages').scrollTop = document.getElementById('chatMessages').scrollHeight;
        })
        .catch(error => console.error('Error:', error));
        
        // Clear the input field
        inputField.value = '';
    }
}
