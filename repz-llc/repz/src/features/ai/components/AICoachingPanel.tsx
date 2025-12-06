import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Input } from '@/ui/atoms/Input';
import { Avatar, AvatarFallback } from '@/ui/atoms/Avatar';
import { Bot, Send, Mic, MicOff, Loader2 } from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/ui/atoms/Badge';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  type?: 'text' | 'voice' | 'analysis';
}

export function AICoachingPanel() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: "Hello! I'm your AI fitness assistant. I can help you with workout analysis, nutrition questions, form feedback, and personalized recommendations. What would you like to work on today?",
      role: 'assistant',
      timestamp: new Date(),
      type: 'text'
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      role: 'user',
      timestamp: new Date(),
      type: 'text'
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      // Simulate AI response
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: generateAIResponse(inputMessage),
        role: 'assistant',
        timestamp: new Date(),
        type: 'text'
      };

      setMessages(prev => [...prev, aiResponse]);
    } catch (error) {
      console.error('AI response error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const generateAIResponse = (input: string): string => {
    const responses = {
      workout: "Based on your current program, I recommend adjusting your rep ranges this week. For strength gains, try 3-5 reps at 85-90% 1RM. Make sure to maintain proper form and take 3-4 minutes rest between sets.",
      nutrition: "Your nutrition looks solid! To optimize recovery, try adding 20-30g of casein protein before bed. Also, consider timing your carbs around your workouts for better performance and glycogen replenishment.",
      form: "From your recent session data, I notice your squat depth could improve. Focus on hip mobility exercises and consider adding goblet squats as a warm-up to reinforce the movement pattern.",
      progress: "Great progress this month! Your strength is up 8% and body composition is improving. To maintain momentum, let's add some variation to your program next week with tempo work.",
      default: "I'm here to help with your fitness journey! You can ask me about workouts, nutrition, form analysis, or progress tracking. What specific area would you like to focus on?"
    };

    const lowercaseInput = input.toLowerCase();
    if (lowercaseInput.includes('workout') || lowercaseInput.includes('exercise')) return responses.workout;
    if (lowercaseInput.includes('nutrition') || lowercaseInput.includes('diet')) return responses.nutrition;
    if (lowercaseInput.includes('form') || lowercaseInput.includes('technique')) return responses.form;
    if (lowercaseInput.includes('progress') || lowercaseInput.includes('results')) return responses.progress;
    
    return responses.default;
  };

  const handleVoiceToggle = () => {
    // Voice recognition implementation coming soon
    setIsListening(!isListening);
  };

  const quickActions = [
    { label: 'Analyze Last Workout', action: () => setInputMessage('Analyze my last workout session') },
    { label: 'Nutrition Check', action: () => setInputMessage('Review my nutrition for today') },
    { label: 'Form Feedback', action: () => setInputMessage('Give me form feedback on my squats') },
    { label: 'Progress Update', action: () => setInputMessage('How is my progress this month?') }
  ];

  return (
    <div className="flex flex-col h-[600px] max-w-4xl mx-auto p-6">
      <Card className="glass-tier-card border-border/20 flex-1 flex flex-col">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2">
            <Bot className="w-5 h-5 text-[hsl(var(--tier-performance))]" />
            AI Fitness Assistant
            <Badge variant="secondary" className="ml-auto">Performance+</Badge>
          </CardTitle>
        </CardHeader>
        
        <CardContent className="flex-1 flex flex-col space-y-4">
          {/* Quick Actions */}
          <div className="flex flex-wrap gap-2">
            {quickActions.map((action, index) => (
              <Button
                key={index}
                variant="outline"
                size="sm"
                onClick={action.action}
                className="text-xs"
              >
                {action.label}
              </Button>
            ))}
          </div>

          {/* Messages */}
          <ScrollArea className="flex-1 pr-4">
            <div className="space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${
                    message.role === 'user' ? 'justify-end' : 'justify-start'
                  }`}
                >
                  <div
                    className={`flex items-start space-x-2 max-w-[80%] ${
                      message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                    }`}
                  >
                    <Avatar className="w-8 h-8">
                      <AvatarFallback>
                        {message.role === 'user' ? 'U' : <Bot className="w-4 h-4" />}
                      </AvatarFallback>
                    </Avatar>
                    <div
                      className={`rounded-lg px-3 py-2 ${
                        message.role === 'user'
                          ? 'bg-[hsl(var(--tier-performance))] text-white'
                          : 'bg-muted'
                      }`}
                    >
                      <p className="text-sm">{message.content}</p>
                      <p className="text-xs opacity-70 mt-1">
                        {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="flex items-start space-x-2">
                    <Avatar className="w-8 h-8">
                      <AvatarFallback>
                        <Bot className="w-4 h-4" />
                      </AvatarFallback>
                    </Avatar>
                    <div className="bg-muted rounded-lg px-3 py-2">
                      <div className="flex items-center space-x-2">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <p className="text-sm">Analyzing...</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>

          {/* Input */}
          <div className="flex items-center space-x-2">
            <div className="flex-1 relative">
              <Input
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                placeholder="Ask about workouts, nutrition, form, or progress..."
                disabled={isLoading}
              />
            </div>
            <Button
              variant="outline"
              size="icon"
              onClick={handleVoiceToggle}
              className={isListening ? 'bg-red-500 text-white' : ''}
            >
              {isListening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
            </Button>
            <Button onClick={handleSendMessage} disabled={!inputMessage.trim() || isLoading}>
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}