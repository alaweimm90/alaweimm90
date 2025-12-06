import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Calendar, Camera, Scale, TrendingUp, 
  Star, MessageSquare, Upload, CheckCircle
} from 'lucide-react';
import { useTierAccess } from '@/hooks/useTierAccess';

interface CheckinFormData {
  weight: string;
  bodyFat: string;
  energy: number;
  motivation: number;
  stress: number;
  sleep: number;
  notes: string;
  photos: File[];
}

export const WeeklyCheckinsModule: React.FC = () => {
  const { userTier, hasWeeklyCheckins } = useTierAccess();
  const [formData, setFormData] = useState<CheckinFormData>({
    weight: '',
    bodyFat: '',
    energy: 5,
    motivation: 5,
    stress: 3,
    sleep: 7,
    notes: '',
    photos: []
  });
  const [submitted, setSubmitted] = useState(false);
  
  const currentTier = userTier || 'core';
  
  const tierColors = {
    core: '#3B82F6',
    adaptive: '#F15B23', 
    performance: '#A855F7',
    longevity: '#EAB308'
  };

  const tierColor = tierColors[currentTier as keyof typeof tierColors];

  if (!hasWeeklyCheckins) {
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
          <Calendar className="w-8 h-8" style={{ color: tierColor }} />
        </div>
        <h2 className="text-xl font-bold text-white mb-2">Weekly Check-ins</h2>
        <p className="text-gray-300 mb-6">
          Track your progress with detailed weekly assessments
        </p>
        <motion.button
          className="px-6 py-3 rounded-lg text-white font-medium"
          style={{ background: `linear-gradient(135deg, ${tierColor}, ${tierColor}80)` }}
          whileHover={{ scale: 1.02 }}
          onClick={() => window.location.href = '/pricing'}
        >
          Upgrade to Access Check-ins
        </motion.button>
      </motion.div>
    );
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
    // Here you would submit to your API
    console.log('Checkin submitted:', formData);
  };

  const handlePhotoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFormData({
        ...formData,
        photos: Array.from(e.target.files)
      });
    }
  };

  const ScaleInput = ({ 
    label, 
    value, 
    onChange, 
    min = 1, 
    max = 10, 
    unit = '' 
  }: {
    label: string;
    value: number;
    onChange: (value: number) => void;
    min?: number;
    max?: number;
    unit?: string;
  }) => (
    <div className="space-y-2">
      <div className="flex justify-between">
        <label className="text-white font-medium">{label}</label>
        <span className="text-white">{value}{unit}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
        style={{
          background: `linear-gradient(to right, ${tierColor} 0%, ${tierColor} ${((value - min) / (max - min)) * 100}%, #374151 ${((value - min) / (max - min)) * 100}%, #374151 100%)`
        }}
      />
      <div className="flex justify-between text-xs text-gray-400">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );

  if (submitted) {
    return (
      <motion.div
        className="glass-tier-card p-8 rounded-xl text-center"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
      >
        <div 
          className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4"
          style={{ background: `${tierColor}20` }}
        >
          <CheckCircle className="w-8 h-8 text-green-400" />
        </div>
        <h2 className="text-xl font-bold text-white mb-2">Check-in Submitted!</h2>
        <p className="text-gray-300 mb-6">
          Your coach will review your progress and provide feedback within 24-48 hours.
        </p>
        <motion.button
          className="px-6 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-600"
          onClick={() => setSubmitted(false)}
          whileHover={{ scale: 1.02 }}
        >
          Submit Another Check-in
        </motion.button>
      </motion.div>
    );
  }

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
            <h1 className="text-2xl font-bold text-white mb-2">Weekly Check-in</h1>
            <p className="text-gray-300">Week 8 • Share your progress and insights</p>
          </div>
          <div 
            className="w-16 h-16 rounded-full flex items-center justify-center"
            style={{ 
              background: `linear-gradient(135deg, ${tierColor}, ${tierColor}80)`,
              boxShadow: `0 0 30px ${tierColor}40`
            }}
          >
            <Calendar className="w-8 h-8 text-white" />
          </div>
        </div>
      </motion.div>

      {/* Check-in Form */}
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Body Metrics */}
        <motion.div
          className="glass-tier-card p-6 rounded-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="flex items-center gap-3 mb-4">
            <Scale className="w-6 h-6" style={{ color: tierColor }} />
            <h3 className="text-lg font-semibold text-white">Body Metrics</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-white font-medium mb-2">Current Weight (kg)</label>
              <input
                type="number"
                step="0.1"
                value={formData.weight}
                onChange={(e) => setFormData({ ...formData, weight: e.target.value })}
                className="w-full p-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:border-current focus:outline-none"
                style={{ borderColor: tierColor }}
                placeholder="75.5"
              />
            </div>
            <div>
              <label className="block text-white font-medium mb-2">Body Fat % (optional)</label>
              <input
                type="number"
                step="0.1"
                value={formData.bodyFat}
                onChange={(e) => setFormData({ ...formData, bodyFat: e.target.value })}
                className="w-full p-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:border-current focus:outline-none"
                style={{ borderColor: tierColor }}
                placeholder="15.0"
              />
            </div>
          </div>
        </motion.div>

        {/* Wellness Metrics */}
        <motion.div
          className="glass-tier-card p-6 rounded-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex items-center gap-3 mb-6">
            <TrendingUp className="w-6 h-6" style={{ color: tierColor }} />
            <h3 className="text-lg font-semibold text-white">Wellness Assessment</h3>
          </div>
          <div className="space-y-6">
            <ScaleInput
              label="Energy Levels"
              value={formData.energy}
              onChange={(value) => setFormData({ ...formData, energy: value })}
            />
            <ScaleInput
              label="Motivation"
              value={formData.motivation}
              onChange={(value) => setFormData({ ...formData, motivation: value })}
            />
            <ScaleInput
              label="Stress Levels"
              value={formData.stress}
              onChange={(value) => setFormData({ ...formData, stress: value })}
            />
            <ScaleInput
              label="Sleep Quality (hours)"
              value={formData.sleep}
              onChange={(value) => setFormData({ ...formData, sleep: value })}
              min={4}
              max={12}
              unit=" hrs"
            />
          </div>
        </motion.div>

        {/* Progress Photos */}
        <motion.div
          className="glass-tier-card p-6 rounded-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="flex items-center gap-3 mb-4">
            <Camera className="w-6 h-6" style={{ color: tierColor }} />
            <h3 className="text-lg font-semibold text-white">Progress Photos</h3>
          </div>
          <div 
            className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center hover:border-current transition-colors cursor-pointer"
            style={{ borderColor: formData.photos.length > 0 ? tierColor : undefined }}
          >
            <input
              type="file"
              multiple
              accept="image/*"
              onChange={handlePhotoUpload}
              className="hidden"
              id="photo-upload"
            />
            <label htmlFor="photo-upload" className="cursor-pointer">
              <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
              <p className="text-gray-400">
                {formData.photos.length > 0 
                  ? `${formData.photos.length} photo(s) selected`
                  : 'Click to upload progress photos'
                }
              </p>
              <p className="text-xs text-gray-500 mt-1">
                Front, side, and back views recommended
              </p>
            </label>
          </div>
        </motion.div>

        {/* Notes */}
        <motion.div
          className="glass-tier-card p-6 rounded-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <div className="flex items-center gap-3 mb-4">
            <MessageSquare className="w-6 h-6" style={{ color: tierColor }} />
            <h3 className="text-lg font-semibold text-white">Weekly Notes</h3>
          </div>
          <textarea
            value={formData.notes}
            onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
            className="w-full p-4 bg-gray-800 border border-gray-600 rounded-lg text-white focus:border-current focus:outline-none h-32 resize-none"
            style={{ borderColor: tierColor }}
            placeholder="How are you feeling? Any challenges? Victories? Questions for your coach?"
          />
        </motion.div>

        {/* Submit Button */}
        <motion.div
          className="flex justify-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <motion.button
            type="submit"
            className="px-8 py-3 rounded-lg text-white font-medium text-lg"
            style={{ background: `linear-gradient(135deg, ${tierColor}, ${tierColor}80)` }}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            Submit Weekly Check-in
          </motion.button>
        </motion.div>
      </form>
    </div>
  );
};