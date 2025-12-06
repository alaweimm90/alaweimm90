import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { SparklesIcon } from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

export const ContentGenerator: React.FC = () => {
  const [contentType, setContentType] = useState('social_post');
  const [platform, setPlatform] = useState('instagram');
  const [prompt, setPrompt] = useState('');
  const [generatedContent, setGeneratedContent] = useState<string[]>([]);

  const generateMutation = useMutation({
    mutationFn: async (data: any) => {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/content/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`
        },
        body: JSON.stringify(data)
      });
      return response.json();
    },
    onSuccess: (data) => {
      setGeneratedContent(data.data.content);
      toast.success('Content generated successfully!');
    },
    onError: () => {
      toast.error('Failed to generate content');
    }
  });

  const handleGenerate = () => {
    generateMutation.mutate({
      type: contentType,
      platform,
      prompt,
      parameters: {
        variations: 3,
        includeHashtags: true,
        includeEmojis: true,
        tone: 'professional'
      }
    });
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">AI Content Generator</h1>
        <p className="text-gray-600 mt-2">Create engaging content for all your marketing channels</p>
      </div>

      {/* Generator Form */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Content Type</label>
              <select
                value={contentType}
                onChange={(e) => setContentType(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="social_post">Social Media Post</option>
                <option value="blog_article">Blog Article</option>
                <option value="email">Email Campaign</option>
                <option value="video_script">Video Script</option>
                <option value="ad_copy">Ad Copy</option>
                <option value="product_description">Product Description</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Platform</label>
              <select
                value={platform}
                onChange={(e) => setPlatform(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="instagram">Instagram</option>
                <option value="facebook">Facebook</option>
                <option value="twitter">Twitter/X</option>
                <option value="linkedin">LinkedIn</option>
                <option value="tiktok">TikTok</option>
                <option value="youtube">YouTube</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">What do you want to create?</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe what you want to create... (e.g., 'Create a post about our new product launch')"
              rows={4}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <button
            onClick={handleGenerate}
            disabled={generateMutation.isPending || !prompt}
            className="w-full bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            <SparklesIcon className="w-5 h-5" />
            {generateMutation.isPending ? 'Generating...' : 'Generate Content'}
          </button>
        </div>
      </div>

      {/* Generated Content */}
      {generatedContent.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-xl font-semibold text-gray-900">Generated Variations</h2>
          {generatedContent.map((content, index) => (
            <div key={index} className="bg-white rounded-lg shadow p-6">
              <div className="flex items-start justify-between mb-4">
                <h3 className="font-medium text-gray-900">Variation {index + 1}</h3>
                <button className="text-blue-600 hover:text-blue-700 text-sm font-medium">
                  Use This
                </button>
              </div>
              <div className="prose max-w-none">
                <p className="whitespace-pre-wrap text-gray-700">{content}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
