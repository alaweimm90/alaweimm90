import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Activity, Play, CheckCircle, Clock, 
  Target, Zap, RotateCcw, Calendar, Mic
} from 'lucide-react';
import { useTierAccess } from '@/hooks/useTierAccess';
import EnhancedVoiceCoaching from '@/components/voice/EnhancedVoiceCoaching';
import VoiceWorkoutGuide from '@/components/voice/VoiceWorkoutGuide';

interface Exercise {
  name: string;
  sets: number;
  reps: string;
  weight: string;
  restTime: number;
  completed: boolean;
}

interface Workout {
  id: string;
  name: string;
  type: string;
  duration: number;
  exercises: Exercise[];
  completed: boolean;
  date: string;
}

export const TrainingModule: React.FC = () => {
  const { userTier } = useTierAccess();
  const [selectedWorkout, setSelectedWorkout] = useState<Workout | null>(null);
  const [currentExerciseIndex, setCurrentExerciseIndex] = useState(0);
  const [activeTab, setActiveTab] = useState<'workouts' | 'voice-coaching' | 'voice-guide'>('workouts');
  
  const currentTier = userTier || 'core';
  
  const tierColors = {
    core: '#3B82F6',
    adaptive: '#F15B23', 
    performance: '#A855F7',
    longevity: '#EAB308'
  };

  const tierColor = tierColors[currentTier as keyof typeof tierColors];

  // Mock workout data
  const workouts: Workout[] = [
    {
      id: '1',
      name: 'Upper Body Power',
      type: 'Strength',
      duration: 60,
      date: 'Today',
      completed: false,
      exercises: [
        { name: 'Bench Press', sets: 4, reps: '6-8', weight: '225 lbs', restTime: 180, completed: false },
        { name: 'Incline Dumbbell Press', sets: 3, reps: '8-10', weight: '85 lbs', restTime: 120, completed: false },
        { name: 'Barbell Rows', sets: 4, reps: '6-8', weight: '185 lbs', restTime: 180, completed: false },
        { name: 'Lat Pulldowns', sets: 3, reps: '10-12', weight: '140 lbs', restTime: 90, completed: false }
      ]
    },
    {
      id: '2',
      name: 'Lower Body Hypertrophy',
      type: 'Hypertrophy',
      duration: 75,
      date: 'Tomorrow',
      completed: false,
      exercises: [
        { name: 'Back Squats', sets: 4, reps: '10-12', weight: '315 lbs', restTime: 150, completed: false },
        { name: 'Romanian Deadlifts', sets: 3, reps: '12-15', weight: '225 lbs', restTime: 120, completed: false },
        { name: 'Bulgarian Split Squats', sets: 3, reps: '12 each', weight: '50 lbs', restTime: 90, completed: false },
        { name: 'Walking Lunges', sets: 3, reps: '20 total', weight: 'Bodyweight', restTime: 60, completed: false }
      ]
    }
  ];

  const handleStartWorkout = (workout: Workout) => {
    setSelectedWorkout(workout);
    setCurrentExerciseIndex(0);
  };

  const handleCompleteExercise = () => {
    if (!selectedWorkout) return;
    
    const updatedExercises = [...selectedWorkout.exercises];
    updatedExercises[currentExerciseIndex].completed = true;
    
    setSelectedWorkout({
      ...selectedWorkout,
      exercises: updatedExercises
    });
    
    if (currentExerciseIndex < selectedWorkout.exercises.length - 1) {
      setCurrentExerciseIndex(currentExerciseIndex + 1);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header with Navigation */}
      <motion.div
        className="glass-tier-card p-6 rounded-xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-white mb-2">Training Program</h1>
            <p className="text-gray-300">Week 8 • Phase 2: Strength & Power</p>
          </div>
          <div 
            className="w-16 h-16 rounded-full flex items-center justify-center"
            style={{ 
              background: `linear-gradient(135deg, ${tierColor}, ${tierColor}80)`,
              boxShadow: `0 0 30px ${tierColor}40`
            }}
          >
            <Activity className="w-8 h-8 text-white" />
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-2 p-1 bg-gray-800/50 rounded-lg">
          <button
            onClick={() => setActiveTab('workouts')}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
              activeTab === 'workouts'
                ? 'text-white'
                : 'text-gray-400 hover:text-white'
            }`}
            style={activeTab === 'workouts' ? { 
              background: `linear-gradient(135deg, ${tierColor}, ${tierColor}80)` 
            } : {}}
          >
            <Activity className="w-4 h-4 inline mr-2" />
            Workouts
          </button>
          
          {(userTier === 'performance' || userTier === 'longevity') && (
            <>
              <button
                onClick={() => setActiveTab('voice-coaching')}
                className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
                  activeTab === 'voice-coaching'
                    ? 'text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
                style={activeTab === 'voice-coaching' ? { 
                  background: `linear-gradient(135deg, ${tierColor}, ${tierColor}80)` 
                } : {}}
              >
                <Mic className="w-4 h-4 inline mr-2" />
                Voice Coach
              </button>
              
              <button
                onClick={() => setActiveTab('voice-guide')}
                className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
                  activeTab === 'voice-guide'
                    ? 'text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
                style={activeTab === 'voice-guide' ? { 
                  background: `linear-gradient(135deg, ${tierColor}, ${tierColor}80)` 
                } : {}}
              >
                <Target className="w-4 h-4 inline mr-2" />
                Guided Training
              </button>
            </>
          )}
        </div>
      </motion.div>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        {activeTab === 'workouts' && !selectedWorkout && (
          <motion.div
            key="workouts"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            {/* Workout Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {workouts.map((workout, index) => (
                <motion.div
                  key={workout.id}
                  className="glass-tier-card p-6 rounded-xl hover:scale-105 transition-transform cursor-pointer"
                  initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  onClick={() => handleStartWorkout(workout)}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="text-lg font-semibold text-white">{workout.name}</h3>
                      <p className="text-sm text-gray-400">{workout.type} • {workout.date}</p>
                    </div>
                    <div 
                      className="w-12 h-12 rounded-lg flex items-center justify-center"
                      style={{ background: `${tierColor}20` }}
                    >
                      <Play className="w-6 h-6" style={{ color: tierColor }} />
                    </div>
                  </div>
                  
                  <div className="space-y-2 mb-4">
                    <div className="flex items-center gap-2">
                      <Clock className="w-4 h-4 text-gray-400" />
                      <span className="text-sm text-gray-300">{workout.duration} minutes</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Target className="w-4 h-4 text-gray-400" />
                      <span className="text-sm text-gray-300">{workout.exercises.length} exercises</span>
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
                    Start Workout
                  </motion.button>
                </motion.div>
              ))}
            </div>

            {/* Weekly Overview */}
            <motion.div
              className="glass-tier-card p-6 rounded-xl"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              <h3 className="text-lg font-semibold text-white mb-4">This Week's Schedule</h3>
              <div className="grid grid-cols-1 md:grid-cols-7 gap-3">
                {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((day, index) => {
                  const hasWorkout = index < 5;
                  const isToday = index === 1;
                  
                  return (
                    <div
                      key={day}
                      className={`p-3 rounded-lg border text-center ${
                        isToday 
                          ? 'border-current' 
                          : hasWorkout 
                            ? 'border-gray-600' 
                            : 'border-gray-800'
                      }`}
                      style={isToday ? { borderColor: tierColor, background: `${tierColor}10` } : {}}
                    >
                      <p className="text-sm font-medium text-white">{day}</p>
                      <p className="text-xs text-gray-400 mt-1">
                        {hasWorkout ? 'Training' : 'Rest'}
                      </p>
                      {hasWorkout && (
                        <div 
                          className="w-2 h-2 rounded-full mx-auto mt-1"
                          style={{ backgroundColor: isToday ? tierColor : '#6B7280' }}
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            </motion.div>
          </motion.div>
        )}

        {activeTab === 'voice-coaching' && (
          <motion.div
            key="voice-coaching"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <EnhancedVoiceCoaching />
          </motion.div>
        )}

        {activeTab === 'voice-guide' && (
          <motion.div
            key="voice-guide"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <VoiceWorkoutGuide />
          </motion.div>
        )}
      </AnimatePresence>

      {selectedWorkout && activeTab === 'workouts' && (
        /* Active Workout View */
        <motion.div
          className="space-y-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          {/* Workout Header */}
          <div className="glass-tier-card p-6 rounded-xl">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-bold text-white">{selectedWorkout.name}</h2>
                <p className="text-gray-300">
                  Exercise {currentExerciseIndex + 1} of {selectedWorkout.exercises.length}
                </p>
              </div>
              <button
                onClick={() => setSelectedWorkout(null)}
                className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600"
              >
                End Workout
              </button>
            </div>
            
            {/* Progress Bar */}
            <div className="mt-4 bg-gray-700 rounded-full h-2">
              <motion.div
                className="h-2 rounded-full"
                style={{ 
                  background: `linear-gradient(90deg, ${tierColor}, ${tierColor}80)`,
                  width: `${((currentExerciseIndex + 1) / selectedWorkout.exercises.length) * 100}%`
                }}
                initial={{ width: 0 }}
                animate={{ 
                  width: `${((currentExerciseIndex + 1) / selectedWorkout.exercises.length) * 100}%` 
                }}
              />
            </div>
          </div>

          {/* Current Exercise */}
          {selectedWorkout.exercises[currentExerciseIndex] && (
            <motion.div
              className="glass-tier-card p-6 rounded-xl"
              key={currentExerciseIndex}
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-white mb-2">
                  {selectedWorkout.exercises[currentExerciseIndex].name}
                </h3>
                <div className="flex items-center justify-center gap-6 text-gray-300">
                  <div className="text-center">
                    <p className="text-2xl font-bold text-white">
                      {selectedWorkout.exercises[currentExerciseIndex].sets}
                    </p>
                    <p className="text-sm">Sets</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-white">
                      {selectedWorkout.exercises[currentExerciseIndex].reps}
                    </p>
                    <p className="text-sm">Reps</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-white">
                      {selectedWorkout.exercises[currentExerciseIndex].weight}
                    </p>
                    <p className="text-sm">Weight</p>
                  </div>
                </div>
              </div>

              <div className="flex gap-4 justify-center">
                <motion.button
                  className="px-6 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-600 flex items-center gap-2"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <RotateCcw className="w-5 h-5" />
                  Rest Timer
                </motion.button>
                
                <motion.button
                  className="px-6 py-3 text-white rounded-lg flex items-center gap-2"
                  style={{ 
                    background: `linear-gradient(135deg, ${tierColor}, ${tierColor}80)` 
                  }}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={handleCompleteExercise}
                >
                  <CheckCircle className="w-5 h-5" />
                  Complete Set
                </motion.button>
              </div>
            </motion.div>
          )}

          {/* Exercise List */}
          <div className="glass-tier-card p-6 rounded-xl">
            <h4 className="text-lg font-semibold text-white mb-4">All Exercises</h4>
            <div className="space-y-3">
              {selectedWorkout.exercises.map((exercise, index) => (
                <div
                  key={index}
                  className={`p-3 rounded-lg flex items-center gap-3 ${
                    index === currentExerciseIndex 
                      ? `border border-current` 
                      : exercise.completed 
                        ? 'bg-green-900/30 border border-green-700'
                        : 'bg-gray-800/30'
                  }`}
                  style={index === currentExerciseIndex ? { borderColor: tierColor, background: `${tierColor}10` } : {}}
                >
                  <div className="w-8 h-8 rounded-full flex items-center justify-center">
                    {exercise.completed ? (
                      <CheckCircle className="w-5 h-5 text-green-400" />
                    ) : index === currentExerciseIndex ? (
                      <div 
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: tierColor }}
                      />
                    ) : (
                      <div className="w-3 h-3 rounded-full bg-gray-500" />
                    )}
                  </div>
                  <div className="flex-1">
                    <p className="text-white font-medium">{exercise.name}</p>
                    <p className="text-sm text-gray-400">
                      {exercise.sets} sets × {exercise.reps} @ {exercise.weight}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};