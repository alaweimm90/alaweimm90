/**
 * REPZ Training Plan PDF Generator
 *
 * Generates professional PDF training plans from client intake data.
 * Uses jsPDF for PDF creation with custom styling.
 */

import jsPDF from 'jspdf';

// Flexible interface that accepts partial or unknown intake data
export interface IntakeData {
  firstName?: string;
  lastName?: string;
  email?: string;
  phone?: string;
  age?: string;
  sex?: string;
  height?: string;
  weight?: string;
  activityLevel?: string;
  healthConditions?: string;
  medications?: string;
  injuries?: string;
  restingHeartRate?: string;
  bloodPressure?: string;
  vo2Max?: string;
  sleepQuality?: string;
  stressLevel?: string;
  trainingYears?: string;
  primaryGoals?: string;
  trainingFrequency?: string;
  availableEquipment?: string;
  trainingStyle?: string;
  dietaryRestrictions?: string;
  eatingSchedule?: string;
  supplements?: string;
  alcoholConsumption?: string;
  smokingStatus?: string;
  sleepSchedule?: string;
  primaryGoal?: string;
  secondaryGoals?: string;
  timeline?: string;
  weeklyCommitment?: string;
  successMetrics?: string;
  paymentType?: string;
  [key: string]: string | undefined; // Allow additional fields
}

interface PlanConfig {
  weeks: number;
  sessionsPerWeek: number;
  includesNutrition: boolean;
  includesCheckIns: boolean;
  revisions: number;
}

// REPZ Brand Colors
const COLORS = {
  orange: '#F15B23',
  darkOrange: '#D94A15',
  dark: '#1A1A1A',
  white: '#FFFFFF',
  gray: '#6B7280',
  lightGray: '#E5E7EB',
};

// Plan configurations by tier - REPZ Canonical Tiers
const PLAN_CONFIGS: Record<string, PlanConfig> = {
  // Core Program - $89/mo
  'core': {
    weeks: 8,
    sessionsPerWeek: 3,
    includesNutrition: true,
    includesCheckIns: false,
    revisions: 1,
  },
  // Adaptive Engine - $149/mo
  'adaptive': {
    weeks: 12,
    sessionsPerWeek: 4,
    includesNutrition: true,
    includesCheckIns: true,
    revisions: 2,
  },
  // Prime Suite - $229/mo
  'performance': {
    weeks: 12,
    sessionsPerWeek: 5,
    includesNutrition: true,
    includesCheckIns: true,
    revisions: 3,
  },
  // Elite Concierge - $349/mo
  'longevity': {
    weeks: 16,
    sessionsPerWeek: 6,
    includesNutrition: true,
    includesCheckIns: true,
    revisions: -1, // unlimited
  },
  // Legacy support
  'one-time-basic': {
    weeks: 8,
    sessionsPerWeek: 3,
    includesNutrition: true,
    includesCheckIns: false,
    revisions: 1,
  },
  'one-time-premium': {
    weeks: 12,
    sessionsPerWeek: 4,
    includesNutrition: true,
    includesCheckIns: true,
    revisions: 2,
  },
  'one-time-concierge': {
    weeks: 12,
    sessionsPerWeek: 5,
    includesNutrition: true,
    includesCheckIns: true,
    revisions: -1,
  },
};

function getPlanConfig(paymentType: string): PlanConfig {
  return PLAN_CONFIGS[paymentType] || PLAN_CONFIGS['core'];
}

function getTierName(paymentType: string): string {
  const names: Record<string, string> = {
    'core': 'Core Program',
    'adaptive': 'Adaptive Engine',
    'performance': 'Prime Suite',
    'longevity': 'Elite Concierge',
    // Legacy support
    'one-time-basic': 'Core Program',
    'one-time-premium': 'Adaptive Engine',
    'one-time-concierge': 'Elite Concierge',
  };
  return names[paymentType] || 'Custom Plan';
}

// Generate workout program based on intake data
function generateWorkoutProgram(intake: IntakeData, config: PlanConfig): string[][] {
  const sessions: string[][] = [];
  const { trainingStyle, availableEquipment, trainingFrequency, primaryGoals } = intake;

  // Basic exercise templates by training style
  const exerciseTemplates: Record<string, string[][]> = {
    strength: [
      ['Barbell Squat', '4x8-10', 'Back, Core'],
      ['Bench Press', '4x8-10', 'Chest, Triceps'],
      ['Deadlift', '3x6-8', 'Back, Glutes'],
      ['Overhead Press', '3x8-10', 'Shoulders'],
      ['Barbell Row', '4x8-10', 'Back, Biceps'],
      ['Pull-ups', '3xAMRAP', 'Back, Biceps'],
      ['Lunges', '3x10/leg', 'Legs, Balance'],
      ['Plank', '3x60s', 'Core'],
    ],
    cardio: [
      ['Treadmill Intervals', '20min', 'Cardiovascular'],
      ['Rowing Machine', '15min', 'Full Body'],
      ['Jump Rope', '3x3min', 'Coordination'],
      ['Cycling', '30min', 'Legs, Cardio'],
      ['Stair Climber', '15min', 'Legs, Cardio'],
      ['Battle Ropes', '3x30s', 'Arms, Core'],
    ],
    hybrid: [
      ['Barbell Squat', '3x10', 'Legs'],
      ['Bench Press', '3x10', 'Chest'],
      ['Rowing Machine', '10min', 'Full Body'],
      ['Deadlift', '3x8', 'Back, Legs'],
      ['Treadmill Intervals', '15min', 'Cardio'],
      ['Pull-ups', '3x8', 'Back'],
      ['Plank', '3x45s', 'Core'],
    ],
    functional: [
      ['Kettlebell Swings', '4x15', 'Full Body'],
      ['Box Jumps', '3x10', 'Power'],
      ['Turkish Get-ups', '3x5/side', 'Full Body'],
      ['Medicine Ball Slams', '3x12', 'Power'],
      ['Farmers Walk', '3x40m', 'Grip, Core'],
      ['Battle Ropes', '3x30s', 'Conditioning'],
      ['Goblet Squat', '3x12', 'Legs'],
    ],
    sports: [
      ['Agility Ladder', '5 sets', 'Speed'],
      ['Sprint Intervals', '8x50m', 'Speed, Power'],
      ['Plyometric Jumps', '3x10', 'Power'],
      ['Core Circuit', '3 rounds', 'Stability'],
      ['Sport-Specific Drills', '20min', 'Skill'],
      ['Recovery Stretching', '10min', 'Flexibility'],
    ],
  };

  const style = trainingStyle || 'hybrid';
  const exercises = exerciseTemplates[style] || exerciseTemplates.hybrid;

  // Generate sessions for the week
  const numSessions = parseInt(trainingFrequency) || config.sessionsPerWeek;

  for (let day = 1; day <= numSessions; day++) {
    // Rotate exercises for variety
    const dayExercises = exercises.slice(
      ((day - 1) * 3) % exercises.length,
      ((day - 1) * 3 + 5) % exercises.length + 1
    );
    sessions.push(dayExercises.flat());
  }

  return sessions;
}

// Generate nutrition guidelines
function generateNutritionGuidelines(intake: IntakeData): string[] {
  const guidelines: string[] = [];
  const weight = parseInt(intake.weight) || 150;
  const activityLevel = intake.activityLevel || 'moderate';

  // Calculate base calories (simplified Harris-Benedict)
  const bmr = weight * 10; // Very simplified
  const multipliers: Record<string, number> = {
    sedentary: 1.2,
    light: 1.375,
    moderate: 1.55,
    intense: 1.725,
    'very-intense': 1.9,
  };

  const tdee = Math.round(bmr * (multipliers[activityLevel] || 1.55));

  // Adjust based on goals
  const goal = intake.primaryGoal?.toLowerCase() || '';
  let targetCalories = tdee;

  if (goal.includes('loss') || goal.includes('lean')) {
    targetCalories = Math.round(tdee * 0.85);
    guidelines.push(`Target: ${targetCalories} calories/day for fat loss`);
  } else if (goal.includes('gain') || goal.includes('muscle')) {
    targetCalories = Math.round(tdee * 1.15);
    guidelines.push(`Target: ${targetCalories} calories/day for muscle gain`);
  } else {
    guidelines.push(`Target: ${targetCalories} calories/day for maintenance`);
  }

  // Macros
  const protein = Math.round(weight * 0.8); // 0.8g per lb
  const fat = Math.round((targetCalories * 0.25) / 9);
  const carbs = Math.round((targetCalories - (protein * 4) - (fat * 9)) / 4);

  guidelines.push(`Protein: ${protein}g/day (${Math.round(protein * 4 / targetCalories * 100)}%)`);
  guidelines.push(`Carbohydrates: ${carbs}g/day (${Math.round(carbs * 4 / targetCalories * 100)}%)`);
  guidelines.push(`Fats: ${fat}g/day (${Math.round(fat * 9 / targetCalories * 100)}%)`);

  // Dietary restrictions
  if (intake.dietaryRestrictions) {
    guidelines.push(`\nDietary Considerations: ${intake.dietaryRestrictions}`);
  }

  // Meal timing
  guidelines.push(`\nMeal Timing: ${intake.eatingSchedule || '3 balanced meals with 2 snacks'}`);

  return guidelines;
}

export function generateTrainingPlanPDF(
  clientId: string,
  fullName: string,
  intakeData: IntakeData,
  paymentType: string,
  coachNotes?: string
): Blob {
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4',
  });

  const config = getPlanConfig(paymentType);
  const tierName = getTierName(paymentType);
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin = 20;
  let yPos = margin;

  // Helper function to add a new page if needed
  const checkNewPage = (spaceNeeded: number) => {
    if (yPos + spaceNeeded > pageHeight - margin) {
      doc.addPage();
      yPos = margin;
      return true;
    }
    return false;
  };

  // Helper to draw section header
  const drawSectionHeader = (title: string) => {
    checkNewPage(20);
    doc.setFillColor(COLORS.orange);
    doc.rect(margin, yPos, pageWidth - (margin * 2), 8, 'F');
    doc.setTextColor(COLORS.white);
    doc.setFontSize(12);
    doc.setFont('helvetica', 'bold');
    doc.text(title.toUpperCase(), margin + 3, yPos + 5.5);
    yPos += 12;
    doc.setTextColor(COLORS.dark);
    doc.setFont('helvetica', 'normal');
  };

  // ===== COVER PAGE =====

  // Header bar
  doc.setFillColor(COLORS.orange);
  doc.rect(0, 0, pageWidth, 45, 'F');

  // REPZ Logo/Title
  doc.setTextColor(COLORS.white);
  doc.setFontSize(36);
  doc.setFont('helvetica', 'bold');
  doc.text('REPZ', pageWidth / 2, 25, { align: 'center' });
  doc.setFontSize(14);
  doc.setFont('helvetica', 'normal');
  doc.text('PERSONALIZED TRAINING PLAN', pageWidth / 2, 38, { align: 'center' });

  // Client Name
  yPos = 70;
  doc.setTextColor(COLORS.dark);
  doc.setFontSize(24);
  doc.setFont('helvetica', 'bold');
  doc.text(fullName.toUpperCase(), pageWidth / 2, yPos, { align: 'center' });

  // Plan Type Badge
  yPos += 15;
  doc.setFillColor(COLORS.dark);
  const badgeWidth = 60;
  doc.roundedRect((pageWidth - badgeWidth) / 2, yPos - 5, badgeWidth, 10, 2, 2, 'F');
  doc.setTextColor(COLORS.white);
  doc.setFontSize(10);
  doc.text(tierName.toUpperCase(), pageWidth / 2, yPos + 2, { align: 'center' });

  // Plan Details Box
  yPos += 25;
  doc.setDrawColor(COLORS.lightGray);
  doc.setFillColor('#F9FAFB');
  doc.roundedRect(margin, yPos, pageWidth - (margin * 2), 50, 3, 3, 'FD');

  doc.setTextColor(COLORS.gray);
  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');

  const details = [
    [`Duration: ${config.weeks} Weeks`, `Sessions/Week: ${config.sessionsPerWeek}`],
    [`Training Style: ${intakeData.trainingStyle || 'Hybrid'}`, `Goal: ${intakeData.primaryGoal?.substring(0, 30) || 'General Fitness'}...`],
    [`Generated: ${new Date().toLocaleDateString()}`, `Plan ID: ${clientId.substring(0, 8)}`],
  ];

  let detailY = yPos + 12;
  details.forEach(row => {
    doc.text(row[0], margin + 5, detailY);
    doc.text(row[1], pageWidth / 2 + 5, detailY);
    detailY += 14;
  });

  // Disclaimer
  yPos = pageHeight - 50;
  doc.setFontSize(8);
  doc.setTextColor(COLORS.gray);
  doc.text('This training plan is personalized based on your intake questionnaire.', pageWidth / 2, yPos, { align: 'center' });
  doc.text('Please consult with a healthcare provider before starting any new exercise program.', pageWidth / 2, yPos + 5, { align: 'center' });
  doc.text('© ' + new Date().getFullYear() + ' REPZ. All rights reserved.', pageWidth / 2, yPos + 15, { align: 'center' });

  // ===== PAGE 2: CLIENT PROFILE =====
  doc.addPage();
  yPos = margin;

  drawSectionHeader('Client Profile');

  const profileData = [
    ['Name', fullName],
    ['Age', intakeData.age + ' years'],
    ['Sex', intakeData.sex || 'Not specified'],
    ['Height', (intakeData.height || 'N/A') + '"'],
    ['Weight', (intakeData.weight || 'N/A') + ' lbs'],
    ['Activity Level', intakeData.activityLevel || 'Moderate'],
    ['Training Experience', (intakeData.trainingYears || '0') + ' years'],
  ];

  doc.setFontSize(10);
  profileData.forEach(([label, value]) => {
    doc.setFont('helvetica', 'bold');
    doc.text(label + ':', margin, yPos);
    doc.setFont('helvetica', 'normal');
    doc.text(value, margin + 45, yPos);
    yPos += 7;
  });

  yPos += 10;
  drawSectionHeader('Goals & Objectives');

  doc.setFontSize(10);
  const goals = [
    ['Primary Goal', intakeData.primaryGoal || 'General fitness improvement'],
    ['Secondary Goals', intakeData.secondaryGoals || 'Not specified'],
    ['Timeline', intakeData.timeline || '12 weeks'],
    ['Weekly Commitment', (intakeData.weeklyCommitment || '5') + ' hours'],
  ];

  goals.forEach(([label, value]) => {
    checkNewPage(15);
    doc.setFont('helvetica', 'bold');
    doc.text(label + ':', margin, yPos);
    yPos += 5;
    doc.setFont('helvetica', 'normal');
    const lines = doc.splitTextToSize(value, pageWidth - (margin * 2) - 5);
    doc.text(lines, margin + 2, yPos);
    yPos += lines.length * 5 + 5;
  });

  // Health Considerations
  if (intakeData.healthConditions || intakeData.injuries || intakeData.medications) {
    yPos += 5;
    drawSectionHeader('Health Considerations');

    doc.setFontSize(10);
    doc.setTextColor('#DC2626'); // Red for important health info
    doc.setFont('helvetica', 'bold');
    doc.text('Please review with your healthcare provider', margin, yPos);
    yPos += 8;

    doc.setTextColor(COLORS.dark);
    doc.setFont('helvetica', 'normal');

    if (intakeData.healthConditions) {
      doc.text('Health Conditions: ' + intakeData.healthConditions.substring(0, 100), margin, yPos);
      yPos += 6;
    }
    if (intakeData.injuries) {
      doc.text('Injuries/Limitations: ' + intakeData.injuries.substring(0, 100), margin, yPos);
      yPos += 6;
    }
    if (intakeData.medications) {
      doc.text('Medications: ' + intakeData.medications.substring(0, 100), margin, yPos);
      yPos += 6;
    }
  }

  // ===== PAGE 3: TRAINING PROGRAM =====
  doc.addPage();
  yPos = margin;

  drawSectionHeader('Weekly Training Program');

  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  doc.text(`Your ${config.weeks}-week program with ${config.sessionsPerWeek} sessions per week`, margin, yPos);
  yPos += 10;

  // Generate workouts
  const workouts = generateWorkoutProgram(intakeData, config);
  const dayNames = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

  workouts.forEach((exercises, idx) => {
    checkNewPage(50);

    // Day header
    doc.setFillColor('#F3F4F6');
    doc.rect(margin, yPos, pageWidth - (margin * 2), 8, 'F');
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11);
    doc.text(`Day ${idx + 1} - ${dayNames[idx]}`, margin + 3, yPos + 5.5);
    yPos += 12;

    // Exercise table header
    doc.setFillColor('#E5E7EB');
    doc.rect(margin, yPos, pageWidth - (margin * 2), 6, 'F');
    doc.setFontSize(9);
    doc.setFont('helvetica', 'bold');
    doc.text('Exercise', margin + 2, yPos + 4);
    doc.text('Sets x Reps', margin + 80, yPos + 4);
    doc.text('Target', margin + 120, yPos + 4);
    yPos += 8;

    // Exercises - take 5 exercises per day
    doc.setFont('helvetica', 'normal');
    for (let i = 0; i < exercises.length; i += 3) {
      if (exercises[i]) {
        doc.text(exercises[i], margin + 2, yPos + 4);
        doc.text(exercises[i + 1] || '', margin + 80, yPos + 4);
        doc.text(exercises[i + 2] || '', margin + 120, yPos + 4);
        yPos += 6;
      }
    }

    yPos += 8;
  });

  // Rest day note
  yPos += 5;
  doc.setFont('helvetica', 'italic');
  doc.setFontSize(9);
  doc.setTextColor(COLORS.gray);
  doc.text('Rest days: Active recovery recommended (light walking, stretching, yoga)', margin, yPos);
  doc.setTextColor(COLORS.dark);

  // ===== PAGE 4: NUTRITION (if included) =====
  if (config.includesNutrition) {
    doc.addPage();
    yPos = margin;

    drawSectionHeader('Nutrition Guidelines');

    const nutritionGuidelines = generateNutritionGuidelines(intakeData);

    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');

    nutritionGuidelines.forEach(line => {
      checkNewPage(8);
      if (line.startsWith('\n')) {
        yPos += 5;
        doc.text(line.trim(), margin, yPos);
      } else {
        doc.text('• ' + line, margin, yPos);
      }
      yPos += 6;
    });

    // Hydration
    yPos += 10;
    drawSectionHeader('Hydration');
    doc.setFontSize(10);
    doc.text('• Aim for 0.5-1 oz of water per pound of body weight daily', margin, yPos);
    yPos += 6;
    doc.text('• Increase intake by 16-24 oz for every hour of exercise', margin, yPos);
    yPos += 6;
    doc.text('• Monitor urine color (pale yellow indicates good hydration)', margin, yPos);

    // Supplement recommendations
    yPos += 15;
    drawSectionHeader('Suggested Supplements');

    const supplements = [
      'Protein Powder - for convenient post-workout recovery',
      'Creatine Monohydrate - for strength and power (5g daily)',
      'Vitamin D3 - especially if limited sun exposure',
      'Fish Oil/Omega-3 - for joint health and recovery',
    ];

    doc.setFontSize(10);
    supplements.forEach(supp => {
      checkNewPage(8);
      doc.text('• ' + supp, margin, yPos);
      yPos += 6;
    });

    doc.setFont('helvetica', 'italic');
    doc.setFontSize(9);
    doc.setTextColor(COLORS.gray);
    yPos += 5;
    doc.text('Always consult with a healthcare provider before starting any supplements.', margin, yPos);
    doc.setTextColor(COLORS.dark);
  }

  // ===== FINAL PAGE: PROGRESS TRACKING =====
  doc.addPage();
  yPos = margin;

  drawSectionHeader('Progress Tracking');

  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  doc.text('Track your progress weekly using the metrics below:', margin, yPos);
  yPos += 10;

  // Progress tracking table
  doc.setDrawColor(COLORS.lightGray);

  // Header row
  doc.setFillColor('#F3F4F6');
  doc.rect(margin, yPos, pageWidth - (margin * 2), 8, 'FD');
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9);
  const headers = ['Week', 'Weight', 'Waist', 'Energy (1-10)', 'Notes'];
  const colWidths = [25, 30, 30, 35, 50];
  let xPos = margin;
  headers.forEach((header, i) => {
    doc.text(header, xPos + 2, yPos + 5.5);
    xPos += colWidths[i];
  });
  yPos += 8;

  // Empty rows for tracking
  doc.setFont('helvetica', 'normal');
  for (let week = 1; week <= Math.min(config.weeks, 12); week++) {
    checkNewPage(8);
    xPos = margin;
    doc.rect(margin, yPos, pageWidth - (margin * 2), 8, 'D');
    doc.text(`Week ${week}`, xPos + 2, yPos + 5.5);
    xPos += colWidths[0];
    // Draw vertical lines
    colWidths.slice(0, -1).forEach((width) => {
      doc.line(xPos, yPos, xPos, yPos + 8);
      xPos += width;
    });
    yPos += 8;
  }

  // Success Metrics
  yPos += 15;
  drawSectionHeader('Your Success Metrics');

  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  if (intakeData.successMetrics) {
    const lines = doc.splitTextToSize(intakeData.successMetrics, pageWidth - (margin * 2));
    doc.text(lines, margin, yPos);
    yPos += lines.length * 5 + 10;
  }

  // Coach Notes
  if (coachNotes) {
    yPos += 10;
    drawSectionHeader("Coach's Notes");

    doc.setFontSize(10);
    const lines = doc.splitTextToSize(coachNotes, pageWidth - (margin * 2));
    doc.text(lines, margin, yPos);
  }

  // Footer on last page
  yPos = pageHeight - 30;
  doc.setDrawColor(COLORS.orange);
  doc.line(margin, yPos, pageWidth - margin, yPos);
  yPos += 10;

  doc.setFontSize(9);
  doc.setTextColor(COLORS.gray);
  doc.text('Questions? Contact us at support@getrepz.app', pageWidth / 2, yPos, { align: 'center' });
  yPos += 5;
  doc.text('www.getrepz.app', pageWidth / 2, yPos, { align: 'center' });

  // Return as Blob
  return doc.output('blob');
}

// Export as downloadable file
export function downloadTrainingPlanPDF(
  clientId: string,
  fullName: string,
  intakeData: IntakeData,
  paymentType: string,
  coachNotes?: string
): void {
  const blob = generateTrainingPlanPDF(clientId, fullName, intakeData, paymentType, coachNotes);
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `REPZ-Training-Plan-${fullName.replace(/\s+/g, '-')}.pdf`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

// Export as base64 for email attachment
export async function generatePDFBase64(
  clientId: string,
  fullName: string,
  intakeData: IntakeData,
  paymentType: string,
  coachNotes?: string
): Promise<string> {
  const blob = generateTrainingPlanPDF(clientId, fullName, intakeData, paymentType, coachNotes);
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64 = reader.result as string;
      // Remove the data:application/pdf;base64, prefix
      resolve(base64.split(',')[1]);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}
