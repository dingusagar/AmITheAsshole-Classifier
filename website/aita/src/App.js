import React, { useState, useRef, useEffect } from 'react';
import { Send, Trash2, Loader2 } from 'lucide-react';

// Loading overlay component
const LoadingOverlay = () => (
  <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
    <div className="bg-white rounded-lg p-8 flex flex-col items-center gap-4 max-w-sm mx-4">
      <Loader2 className="w-12 h-12 text-blue-600 animate-spin" />
      <div className="text-center">
        <h3 className="font-semibold text-lg text-gray-900">Analyzing your story...</h3>
        <p className="text-gray-500 mt-1">This might take a few seconds</p>
      </div>
    </div>
  </div>
);

const Message = ({ content, role }) => {
  const hasExplanation = content.includes("Explanation:");
  const [verdict, explanation] = hasExplanation 
    ? content.split("\n\nExplanation:") 
    : [content, null];

  return (
    <div 
      className={`mb-4 ${role === 'user' ? 'ml-auto max-w-xl' : 'mr-auto max-w-2xl'}`}
    >
      <div 
        className={`rounded-lg p-4 ${
          role === 'user' 
            ? 'bg-blue-600 text-white' 
            : 'bg-white shadow-sm'
        }`}
      >
        <p className={role === 'user' ? 'text-white' : 'text-gray-700'}>
          {verdict}
        </p>
        {explanation && (
          <>
            <div className="my-2 border-t border-gray-200"></div>
            <p className={`mt-2 text-sm ${role === 'user' ? 'text-white' : 'text-gray-600'}`}>
              {explanation.trim()}
            </p>
          </>
        )}
      </div>
    </div>
  );
};

const App = () => {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Welcome! Share your AITA story, and I\'ll analyze it.' }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (message.trim()) {
      setMessages(prev => [...prev, { role: 'user', content: message.trim() }]);
      setIsLoading(true);
      
      try {
        const response = await fetch('http://localhost:5000/message', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message: message.trim() }),
        });
        
        const data = await response.json();
        
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: data.message 
        }]);
      } catch (error) {
        console.error('Error:', error);
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: 'Sorry, there was an error processing your message.' 
        }]);
      } finally {
        setIsLoading(false);
      }
      
      setMessage('');
    }
  };

  const handleClearHistory = () => {
    setMessages([{ role: 'assistant', content: 'Welcome! Share your AITA story, and I\'ll analyze it.' }]);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {isLoading && <LoadingOverlay />}
      
      <header className="fixed top-0 left-0 right-0 bg-white border-b shadow-sm z-10">
        <div className="max-w-3xl mx-auto flex justify-between items-center p-4">
          <h1 className="text-xl font-semibold text-gray-800">AITA Classifier</h1>
          <button
            onClick={handleClearHistory}
            className="flex items-center gap-2 px-4 py-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors duration-200"
          >
            <Trash2 className="w-5 h-5" />
            Clear History
          </button>
        </div>
      </header>

      <main className="flex-1 p-4 mt-16 overflow-hidden">
        <div className="max-w-3xl mx-auto h-full flex flex-col">
          <div className="flex-1 overflow-y-auto">
            {messages.map((msg, index) => (
              <Message key={index} {...msg} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>
      </main>

      <div className="sticky bottom-0 border-t bg-white p-4 shadow-sm">
        <div className="max-w-3xl mx-auto">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Share your AITA story here..."
              className="flex-1 resize-none rounded-lg border border-gray-200 p-3 focus:outline-none focus:ring-2 focus:ring-blue-500 min-h-[56px] max-h-[200px]"
              rows={1}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              disabled={isLoading}
            />
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={!message.trim() || isLoading}
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default App;