import React from 'react';
import { motion } from 'framer-motion';
import { 
  Activity, Target, Calendar, TrendingUp, 
  Clock, Zap, Trophy, Star
} from 'lucide-react';
import { useTierAccess } from '@/hooks/useTierAccess';

interface OverviewModuleProps {
  userName?: string;
}

export const OverviewModule: React.FC<OverviewModuleProps> = ({ userName = 'Athlete' }) => {
  const { userTier } = useTierAccess();
  
  const currentTier = userTier || 'core';
  
  // Mock data - in real app this would come from API
  const stats = {
    currentWeek: 8,
    totalWorkouts: 24,
    adherenceRate: 92,
    weightProgress: -3.2,
    bodyFatProgress: -2.1,
    nextSession: 'Upper Body Power'
  };

  const tierColors = {
    core: '#3B82F6',
    adaptive: '#F15B23', 
    performance: '#A855F7',
    longevity: '#EAB308'
  };

  const tierColor = tierColors[currentTier as keyof typeof tierColors];

  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <motion.div
        className="glass-tier-card p-6 rounded-xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white mb-2">
              Welcome back, {userName}! 👋
            </h1>
            <p className="text-gray-300">
              You're in week {stats.currentWeek} of your transformation journey
            </p>
          </div>
          <div 
            className="w-16 h-16 rounded-full flex items-center justify-center"
            style={{ 
              background: `linear-gradient(135deg, ${tierColor}, ${tierColor}80)`,
              boxShadow: `0 0 30px ${tierColor}40`
            }}
          >
            <Trophy className="w-8 h-8 text-white" />
          </div>
        </div>
      </motion.div>

      {/* Quick Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <motion.div
          className="glass-tier-card p-4 rounded-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="flex items-center gap-3">
            <div 
              className="w-10 h-10 rounded-lg flex items-center justify-center"
              style={{ background: `${tierColor}20` }}
            >
              <Activity className="w-5 h-5" style={{ color: tierColor }} />
            </div>
            <div>
              <p className="text-2xl font-bold text-white">{stats.totalWorkouts}</p>
              <p className="text-sm text-gray-400">Total Workouts</p>
            </div>
          </div>
        </motion.div>

        <motion.div
          className="glass-tier-card p-4 rounded-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex items-center gap-3">
            <div 
              className="w-10 h-10 rounded-lg flex items-center justify-center"
              style={{ background: `${tierColor}20` }}
            >
              <Target className="w-5 h-5" style={{ color: tierColor }} />
            </div>
            <div>
              <p className="text-2xl font-bold text-white">{stats.adherenceRate}%</p>
              <p className="text-sm text-gray-400">Adherence Rate</p>
            </div>
          </div>
        </motion.div>

        <motion.div
          className="glass-tier-card p-4 rounded-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="flex items-center gap-3">
            <div 
              className="w-10 h-10 rounded-lg flex items-center justify-center"
              style={{ background: `${tierColor}20` }}
            >
              <TrendingUp className="w-5 h-5" style={{ color: tierColor }} />
            </div>
            <div>
              <p className="text-2xl font-bold text-white">{stats.weightProgress}kg</p>
              <p className="text-sm text-gray-400">Weight Change</p>
            </div>
          </div>
        </motion.div>

        <motion.div
          className="glass-tier-card p-4 rounded-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <div className="flex items-center gap-3">
            <div 
              className="w-10 h-10 rounded-lg flex items-center justify-center"
              style={{ background: `${tierColor}20` }}
            >
              <Zap className="w-5 h-5" style={{ color: tierColor }} />
            </div>
            <div>
              <p className="text-2xl font-bold text-white">{stats.bodyFatProgress}%</p>
              <p className="text-sm text-gray-400">Body Fat Change</p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Next Session */}
        <motion.div
          className="glass-tier-card p-6 rounded-xl"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.5 }}
        >
          <div className="flex items-center gap-3 mb-4">
            <Calendar className="w-6 h-6" style={{ color: tierColor }} />
            <h3 className="text-lg font-semibold text-white">Next Session</h3>
          </div>
          <div className="space-y-3">
            <div className="p-4 bg-gray-800/50 rounded-lg">
              <h4 className="font-medium text-white mb-1">{stats.nextSession}</h4>
              <p className="text-sm text-gray-400 mb-2">Tomorrow, 6:00 AM</p>
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-400">60 minutes</span>
              </div>
            </div>
            <motion.button
              className="w-full py-2 px-4 rounded-lg text-white font-medium"
              style={{ 
                background: `linear-gradient(135deg, ${tierColor}, ${tierColor}80)` 
              }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              View Workout Details
            </motion.button>
          </div>
        </motion.div>

        {/* Recent Activity */}
        <motion.div
          className="glass-tier-card p-6 rounded-xl"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.6 }}
        >
          <div className="flex items-center gap-3 mb-4">
            <Star className="w-6 h-6" style={{ color: tierColor }} />
            <h3 className="text-lg font-semibold text-white">Recent Activity</h3>
          </div>
          <div className="space-y-3">
            {[
              { action: 'Completed Upper Body Power', time: '2 hours ago', type: 'workout' },
              { action: 'Uploaded progress photos', time: '1 day ago', type: 'progress' },
              { action: 'Coach reviewed form video', time: '2 days ago', type: 'coaching' },
              { action: 'Weekly check-in submitted', time: '3 days ago', type: 'checkin' }
            ].map((activity, index) => (
              <motion.div
                key={index}
                className="flex items-center gap-3 p-3 bg-gray-800/30 rounded-lg"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.7 + index * 0.1 }}
              >
                <div 
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: tierColor }}
                />
                <div className="flex-1">
                  <p className="text-sm text-white">{activity.action}</p>
                  <p className="text-xs text-gray-400">{activity.time}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Progress Chart Placeholder */}
      <motion.div
        className="glass-tier-card p-6 rounded-xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
      >
        <h3 className="text-lg font-semibold text-white mb-4">Progress Overview</h3>
        <div className="h-64 bg-gray-800/30 rounded-lg flex items-center justify-center">
          <p className="text-gray-400">Progress chart will be displayed here</p>
        </div>
      </motion.div>
    </div>
  );
};