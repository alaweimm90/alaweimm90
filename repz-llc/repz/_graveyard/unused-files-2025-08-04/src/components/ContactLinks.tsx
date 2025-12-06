import React from 'react';
import { Button } from "@/components/ui/button";
import { MessageSquare, Instagram, ExternalLink, Send, Phone, Mail } from 'lucide-react';

export const ContactLinks = () => {
  const contactMethods = [
    {
      name: "WhatsApp",
      icon: <MessageSquare className="w-5 h-5" />,
      url: "https://wa.me/message/JKZKE5JEYB4MB1",
      description: "Direct messaging"
    },
    {
      name: "Instagram", 
      icon: <Instagram className="w-5 h-5" />,
      url: "https://www.instagram.com/repz",
      description: "Follow for updates"
    },
    {
      name: "Taplink Hub",
      icon: <ExternalLink className="w-5 h-5" />,
      url: "https://taplink.cc/repz", 
      description: "All-in-one link hub"
    },
    {
      name: "Telegram",
      icon: <Send className="w-5 h-5" />,
      url: "https://t.me/repz",
      description: "Secure messaging"
    },
    {
      name: "SMS",
      icon: <Phone className="w-5 h-5" />,
      url: "sms:+14159929792",
      description: "Text directly"
    },
    {
      name: "Email",
      icon: <Mail className="w-5 h-5" />,
      url: "mailto:contact@repz.com",
      description: "Professional contact"
    }
  ];

  const openContact = (url: string) => {
    window.open(url, '_blank');
  };

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
      {contactMethods.map((method, index) => (
        <Button
          key={index}
          variant="outline"
          className="flex flex-col items-center gap-2 h-auto p-4 hover:bg-primary/5 hover:border-primary transition-all duration-300"
          onClick={() => openContact(method.url)}
        >
          {method.icon}
          <span className="font-medium text-xs">{method.name}</span>
          <span className="text-xs text-muted-foreground text-center">{method.description}</span>
        </Button>
      ))}
    </div>
  );
};