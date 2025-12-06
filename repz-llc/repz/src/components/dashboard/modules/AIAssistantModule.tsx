import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Bot, Send, Sparkles, Brain, Activity, 
  Zap, MessageSquare, Mic, MicOff, Volume2
} from 'lucide-react';
import { useTierAccess } from '@/hooks/useTierAccess';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  type: 'text' | 'voice';
}

export const AIAssistantModule: React.FC = () => {
  const { userTier, hasAIAssistant } = useTierAccess();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: "Hi! I'm your AI Fitness Assistant. I can help you with workout modifications, nutrition advice, form checks, and answer any fitness-related questions. What would you like to know?",
      sender: 'ai',
      timestamp: new Date(),
      type: 'text'
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const currentTier = userTier || 'core';
  
  const tierColors = {
    core: '#3B82F6',
    adaptive: '#F15B23', 
    performance: '#A855F7',
    longevity: '#EAB308'
  };

  const tierColor = tierColors[currentTier as keyof typeof tierColors];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (!hasAIAssistant) {
    return (
      <motion.div
        className="glass-tier-card p-8 rounded-xl text-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div 
          className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4"
          style={{ background: `${tierColor}20` }}
        >
          <Bot className="w-8 h-8" style={{ color: tierColor }} />
        </div>
        <h2 className="text-xl font-bold text-white mb-2">AI Fitness Assistant</h2>
        <p className="text-gray-300 mb-6">
          Get instant answers to your fitness questions with our AI-powered assistant
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="p-4 bg-gray-800/30 rounded-lg">
            <Brain className="w-6 h-6 text-purple-400 mx-auto mb-2" />
            <p className="text-sm text-gray-300">Smart Recommendations</p>
          </div>
          <div className="p-4 bg-gray-800/30 rounded-lg">
            <Activity className="w-6 h-6 text-blue-400 mx-auto mb-2" />
            <p className="text-sm text-gray-300">Form Analysis</p>
          </div>
          <div className="p-4 bg-gray-800/30 rounded-lg">
            <Zap className="w-6 h-6 text-orange-400 mx-auto mb-2" />
            <p className="text-sm text-gray-300">24/7 Availability</p>
          </div>
        </div>
        <motion.button
          className="px-6 py-3 rounded-lg text-white font-medium"
          style={{ background: `linear-gradient(135deg, ${tierColor}, ${tierColor}80)` }}
          whileHover={{ scale: 1.02 }}
          onClick={() => window.location.href = '/pricing'}
        >
          Upgrade to Performance Tier
        </motion.button>
      </motion.div>
    );
  }

  const handleSendMessage = async (content: string, type: 'text' | 'voice' = 'text') => {
    if (!content.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      sender: 'user',
      timestamp: new Date(),
      type
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    // Simulate AI response
    setTimeout(() => {
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: getAIResponse(content),
        sender: 'ai',
        timestamp: new Date(),
        type: 'text'
      };
      setMessages(prev => [...prev, aiResponse]);
      setIsLoading(false);
    }, 1500);
  };

  const getAIResponse = (userInput: string): string => {
    const input = userInput.toLowerCase();
    
    if (input.includes('protein')) {
      return "For optimal muscle growth and recovery, aim for 0.8-1g of protein per pound of body weight daily. High-quality sources include lean meats, fish, eggs, dairy, and plant-based options like legumes and quinoa. Timing matters too - try to have 20-30g of protein within 30 minutes post-workout.";
    }
    
    if (input.includes('squat') || input.includes('form')) {
      return "For proper squat form: 1) Keep your feet shoulder-width apart, 2) Engage your core throughout, 3) Lower until your thighs are parallel to the floor, 4) Keep your knees in line with your toes, 5) Drive through your heels to stand up. Start with bodyweight and focus on form before adding weight.";
    }
    
    if (input.includes('cardio')) {
      return "Based on your current program, I recommend 2-3 cardio sessions per week. For fat loss, try HIIT (20-30 minutes) or moderate steady-state (30-45 minutes). For general health and recovery, light walks or easy cycling work great. What's your primary goal?";
    }
    
    if (input.includes('sleep') || input.includes('recovery')) {
      return "Recovery is crucial for progress! Aim for 7-9 hours of quality sleep. Here are key recovery tips: 1) Maintain consistent sleep schedule, 2) Keep room cool and dark, 3) Avoid screens 1 hour before bed, 4) Consider magnesium supplementation, 5) Practice stress management. How's your current sleep quality?";
    }
    
    return "That's a great question! Based on your current program and goals, I'd recommend focusing on consistency first. Every individual responds differently to training, so it's important to track your progress and adjust as needed. Would you like me to elaborate on any specific aspect?";
  };

  const quickQuestions = [
    "How much protein should I eat?",
    "Check my squat form",
    "Best cardio for fat loss?",
    "Help with sleep and recovery"
  ];

  const toggleVoiceInput = () => {
    setIsListening(!isListening);
    // Voice recording logic would go here
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        className="glass-tier-card p-6 rounded-xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white mb-2">AI Fitness Assistant</h1>
            <p className="text-gray-300">Your 24/7 intelligent coaching companion</p>
          </div>
          <div 
            className="w-16 h-16 rounded-full flex items-center justify-center"
            style={{ 
              background: `linear-gradient(135deg, ${tierColor}, ${tierColor}80)`,
              boxShadow: `0 0 30px ${tierColor}40`
            }}
          >
            <Bot className="w-8 h-8 text-white" />
          </div>
        </div>
      </motion.div>

      {/* Chat Interface */}
      <motion.div
        className="glass-tier-card rounded-xl overflow-hidden"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        {/* Messages */}
        <div className="h-96 overflow-y-auto p-6 space-y-4">
          {messages.map((message) => (
            <motion.div
              key={message.id}
              className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className={`flex items-start gap-3 max-w-[80%] ${message.sender === 'user' ? 'flex-row-reverse' : ''}`}>
                <div 
                  className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
                  style={{
                    background: message.sender === 'ai' 
                      ? `linear-gradient(135deg, ${tierColor}, ${tierColor}80)`
                      : '#374151'
                  }}
                >
                  {message.sender === 'ai' ? (
                    <Bot className="w-4 h-4 text-white" />
                  ) : (
                    <div className="w-3 h-3 bg-white rounded-full" />
                  )}
                </div>
                <div
                  className={`p-3 rounded-lg ${
                    message.sender === 'user'
                      ? 'bg-gray-700 text-white'
                      : 'bg-gray-800/50 text-gray-100 border border-gray-700'
                  }`}
                >
                  <p className="text-sm leading-relaxed">{message.content}</p>
                  <p className="text-xs text-gray-400 mt-1">
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
          
          {isLoading && (
            <motion.div
              className="flex justify-start"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <div className="flex items-start gap-3 max-w-[80%]">
                <div 
                  className="w-8 h-8 rounded-full flex items-center justify-center"
                  style={{ background: `linear-gradient(135deg, ${tierColor}, ${tierColor}80)` }}
                >
                  <Bot className="w-4 h-4 text-white" />
                </div>
                <div className="bg-gray-800/50 border border-gray-700 p-3 rounded-lg">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                  </div>
                </div>
              </div>
            </motion.div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Quick Questions */}
        <div className="px-6 pb-4">
          <p className="text-sm text-gray-400 mb-2">Quick questions:</p>
          <div className="flex flex-wrap gap-2">
            {quickQuestions.map((question, index) => (
              <motion.button
                key={index}
                className="px-3 py-1 text-xs bg-gray-700 text-gray-300 rounded-full hover:bg-gray-600 transition-colors"
                onClick={() => handleSendMessage(question)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {question}
              </motion.button>
            ))}
          </div>
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-700 p-4">
          <form 
            onSubmit={(e) => {
              e.preventDefault();
              handleSendMessage(inputMessage);
            }}
            className="flex gap-2"
          >
            <div className="flex-1 relative">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Ask me anything about fitness, nutrition, or your program..."
                className="w-full p-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:border-current focus:outline-none pr-12"
                style={{ borderColor: tierColor }}
                disabled={isLoading}
              />
              <button
                type="button"
                onClick={toggleVoiceInput}
                className={`absolute right-3 top-1/2 transform -translate-y-1/2 p-1 rounded-full transition-colors ${
                  isListening ? 'text-red-400 bg-red-400/20' : 'text-gray-400 hover:text-white'
                }`}
              >
                {isListening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
              </button>
            </div>
            
            <motion.button
              type="submit"
              className="p-3 rounded-lg text-white flex items-center gap-2"
              style={{ background: `linear-gradient(135deg, ${tierColor}, ${tierColor}80)` }}
              disabled={!inputMessage.trim() || isLoading}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Send className="w-4 h-4" />
            </motion.button>
          </form>
        </div>
      </motion.div>

      {/* AI Capabilities */}
      <motion.div
        className="glass-tier-card p-6 rounded-xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="flex items-center gap-3 mb-4">
          <Sparkles className="w-6 h-6" style={{ color: tierColor }} />
          <h3 className="text-lg font-semibold text-white">AI Capabilities</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-gray-800/30 rounded-lg">
            <Brain className="w-6 h-6 text-purple-400 mb-2" />
            <h4 className="font-medium text-white mb-1">Smart Analysis</h4>
            <p className="text-xs text-gray-400">Analyzes your progress and provides personalized recommendations</p>
          </div>
          <div className="p-4 bg-gray-800/30 rounded-lg">
            <Activity className="w-6 h-6 text-blue-400 mb-2" />
            <h4 className="font-medium text-white mb-1">Form Coaching</h4>
            <p className="text-xs text-gray-400">Real-time form corrections and exercise modifications</p>
          </div>
          <div className="p-4 bg-gray-800/30 rounded-lg">
            <MessageSquare className="w-6 h-6 text-green-400 mb-2" />
            <h4 className="font-medium text-white mb-1">24/7 Support</h4>
            <p className="text-xs text-gray-400">Instant answers to your fitness and nutrition questions</p>
          </div>
        </div>
      </motion.div>
    </div>
  );
};