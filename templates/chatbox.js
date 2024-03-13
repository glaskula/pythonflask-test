import React from 'react';

class Chatbox extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      messages: [] // Initialize an empty array to hold chat messages
    };
  }

  handleSubmit = async (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    const question = formData.get('question');

    // Send the question to the Flask backend
    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question })
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();

      // Update state with the received response
      this.setState(prevState => ({
        messages: [...prevState.messages, { text: data.result, type: 'received' }]
      }));
    } catch (error) {
      console.error('Error:', error);
    }

    // Clear the input field after sending the question
    event.target.reset();
  }

  render() {
    return (
      <div className="chatbox">
        <div className="chatbox-header">
          <h1>Chatbox</h1>
        </div>

        <div className="chatbox-messages">
          {this.state.messages.map((message, index) => (
            <div key={index} className={`message ${message.type}`}>
              {message.text}
            </div>
          ))}
        </div>

        <form onSubmit={this.handleSubmit}>
          <input type="text" id="question" name="question" required placeholder="Type your message..." />
          <button type="submit">Send</button>
        </form>
      </div>
    );
  }
}

export default Chatbox;
