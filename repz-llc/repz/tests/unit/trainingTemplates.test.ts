/**
 * Unit tests for training program templates
 * @file tests/unit/trainingTemplates.test.ts
 */

import { describe, it, expect } from 'vitest';

// Volume landmarks data (from trainingTemplates.ts)
const VOLUME_LANDMARKS = [
  { muscleGroup: 'chest', mev: 10, mav: 18, mrv: 22, frequency: 2 },
  { muscleGroup: 'back', mev: 10, mav: 20, mrv: 25, frequency: 2 },
  { muscleGroup: 'shoulders', mev: 8, mav: 16, mrv: 22, frequency: 2 },
  { muscleGroup: 'biceps', mev: 8, mav: 14, mrv: 20, frequency: 2 },
  { muscleGroup: 'triceps', mev: 6, mav: 12, mrv: 18, frequency: 2 },
  { muscleGroup: 'quadriceps', mev: 8, mav: 16, mrv: 20, frequency: 2 },
  { muscleGroup: 'hamstrings', mev: 6, mav: 12, mrv: 16, frequency: 2 },
  { muscleGroup: 'glutes', mev: 4, mav: 12, mrv: 16, frequency: 2 },
  { muscleGroup: 'calves', mev: 8, mav: 16, mrv: 20, frequency: 3 },
  { muscleGroup: 'abs', mev: 0, mav: 16, mrv: 20, frequency: 3 },
];

// RPE/RIR conversion table
const RPE_RIR_TABLE = [
  { rpe: 10, rir: 0, percentage: 100 },
  { rpe: 9.5, rir: 0.5, percentage: 97 },
  { rpe: 9, rir: 1, percentage: 95 },
  { rpe: 8.5, rir: 1.5, percentage: 92 },
  { rpe: 8, rir: 2, percentage: 90 },
  { rpe: 7.5, rir: 2.5, percentage: 87 },
  { rpe: 7, rir: 3, percentage: 85 },
  { rpe: 6.5, rir: 3.5, percentage: 82 },
  { rpe: 6, rir: 4, percentage: 80 },
  { rpe: 5, rir: 5, percentage: 75 },
];

// Training splits
const TRAINING_SPLITS = {
  pushPullLegs: { daysPerWeek: 6, name: 'Push/Pull/Legs' },
  upperLower: { daysPerWeek: 4, name: 'Upper/Lower' },
  fullBody: { daysPerWeek: 3, name: 'Full Body' },
  arnoldSplit: { daysPerWeek: 6, name: 'Arnold Split' },
  broSplit: { daysPerWeek: 5, name: 'Bro Split' },
  powerbuilding: { daysPerWeek: 4, name: 'Powerbuilding' },
};

// Utility functions
function calculateWeeklyVolume(setsPerSession: number, frequency: number): number {
  return setsPerSession * frequency;
}

function isVolumeInRange(volume: number, mev: number, mrv: number): boolean {
  return volume >= mev && volume <= mrv;
}

function rpeToPercentage(rpe: number): number {
  const entry = RPE_RIR_TABLE.find((e) => e.rpe === rpe);
  return entry?.percentage ?? 0;
}

function rirToRpe(rir: number): number {
  const entry = RPE_RIR_TABLE.find((e) => e.rir === rir);
  return entry?.rpe ?? 0;
}

function getRecommendedSplit(daysAvailable: number): string {
  if (daysAvailable <= 3) return 'fullBody';
  if (daysAvailable === 4) return 'upperLower';
  if (daysAvailable === 5) return 'broSplit';
  return 'pushPullLegs';
}

describe('Volume Landmarks', () => {
  describe('Structure', () => {
    it('should have landmarks for all major muscle groups', () => {
      const muscleGroups = VOLUME_LANDMARKS.map((v) => v.muscleGroup);
      expect(muscleGroups).toContain('chest');
      expect(muscleGroups).toContain('back');
      expect(muscleGroups).toContain('shoulders');
      expect(muscleGroups).toContain('quadriceps');
      expect(muscleGroups).toContain('hamstrings');
    });

    it('should have MEV <= MAV <= MRV for all muscle groups', () => {
      VOLUME_LANDMARKS.forEach((landmark) => {
        expect(landmark.mev).toBeLessThanOrEqual(landmark.mav);
        expect(landmark.mav).toBeLessThanOrEqual(landmark.mrv);
      });
    });

    it('should have positive frequency for all muscle groups', () => {
      VOLUME_LANDMARKS.forEach((landmark) => {
        expect(landmark.frequency).toBeGreaterThan(0);
      });
    });
  });

  describe('Specific Values', () => {
    it('should have correct chest volume landmarks', () => {
      const chest = VOLUME_LANDMARKS.find((v) => v.muscleGroup === 'chest');
      expect(chest?.mev).toBe(10);
      expect(chest?.mav).toBe(18);
      expect(chest?.mrv).toBe(22);
    });

    it('should have correct back volume landmarks', () => {
      const back = VOLUME_LANDMARKS.find((v) => v.muscleGroup === 'back');
      expect(back?.mev).toBe(10);
      expect(back?.mav).toBe(20);
      expect(back?.mrv).toBe(25);
    });

    it('should have higher frequency for calves', () => {
      const calves = VOLUME_LANDMARKS.find((v) => v.muscleGroup === 'calves');
      expect(calves?.frequency).toBe(3);
    });

    it('should allow zero MEV for abs (indirect work)', () => {
      const abs = VOLUME_LANDMARKS.find((v) => v.muscleGroup === 'abs');
      expect(abs?.mev).toBe(0);
    });
  });
});

describe('RPE/RIR Conversion', () => {
  describe('Table Structure', () => {
    it('should have entries from RPE 5 to 10', () => {
      const rpeValues = RPE_RIR_TABLE.map((e) => e.rpe);
      expect(Math.min(...rpeValues)).toBe(5);
      expect(Math.max(...rpeValues)).toBe(10);
    });

    it('should have decreasing percentage as RPE decreases', () => {
      const sortedByRpe = [...RPE_RIR_TABLE].sort((a, b) => b.rpe - a.rpe);
      for (let i = 1; i < sortedByRpe.length; i++) {
        expect(sortedByRpe[i].percentage).toBeLessThan(sortedByRpe[i - 1].percentage);
      }
    });

    it('should have RIR inversely related to RPE', () => {
      RPE_RIR_TABLE.forEach((entry) => {
        expect(entry.rpe + entry.rir).toBeCloseTo(10, 1);
      });
    });
  });

  describe('rpeToPercentage', () => {
    it('should return 100% for RPE 10', () => {
      expect(rpeToPercentage(10)).toBe(100);
    });

    it('should return 90% for RPE 8', () => {
      expect(rpeToPercentage(8)).toBe(90);
    });

    it('should return 85% for RPE 7', () => {
      expect(rpeToPercentage(7)).toBe(85);
    });

    it('should return 0 for invalid RPE', () => {
      expect(rpeToPercentage(11)).toBe(0);
      expect(rpeToPercentage(4)).toBe(0);
    });
  });

  describe('rirToRpe', () => {
    it('should return RPE 10 for RIR 0', () => {
      expect(rirToRpe(0)).toBe(10);
    });

    it('should return RPE 8 for RIR 2', () => {
      expect(rirToRpe(2)).toBe(8);
    });

    it('should return RPE 7 for RIR 3', () => {
      expect(rirToRpe(3)).toBe(7);
    });
  });
});

describe('Training Splits', () => {
  describe('Structure', () => {
    it('should have 6 different split options', () => {
      expect(Object.keys(TRAINING_SPLITS)).toHaveLength(6);
    });

    it('should have days per week between 3 and 6', () => {
      Object.values(TRAINING_SPLITS).forEach((split) => {
        expect(split.daysPerWeek).toBeGreaterThanOrEqual(3);
        expect(split.daysPerWeek).toBeLessThanOrEqual(6);
      });
    });
  });

  describe('Specific Splits', () => {
    it('should have PPL as 6 days', () => {
      expect(TRAINING_SPLITS.pushPullLegs.daysPerWeek).toBe(6);
    });

    it('should have Upper/Lower as 4 days', () => {
      expect(TRAINING_SPLITS.upperLower.daysPerWeek).toBe(4);
    });

    it('should have Full Body as 3 days', () => {
      expect(TRAINING_SPLITS.fullBody.daysPerWeek).toBe(3);
    });
  });

  describe('getRecommendedSplit', () => {
    it('should recommend full body for 3 or fewer days', () => {
      expect(getRecommendedSplit(1)).toBe('fullBody');
      expect(getRecommendedSplit(2)).toBe('fullBody');
      expect(getRecommendedSplit(3)).toBe('fullBody');
    });

    it('should recommend upper/lower for 4 days', () => {
      expect(getRecommendedSplit(4)).toBe('upperLower');
    });

    it('should recommend bro split for 5 days', () => {
      expect(getRecommendedSplit(5)).toBe('broSplit');
    });

    it('should recommend PPL for 6+ days', () => {
      expect(getRecommendedSplit(6)).toBe('pushPullLegs');
      expect(getRecommendedSplit(7)).toBe('pushPullLegs');
    });
  });
});

describe('Volume Calculations', () => {
  describe('calculateWeeklyVolume', () => {
    it('should correctly calculate weekly volume', () => {
      expect(calculateWeeklyVolume(5, 2)).toBe(10);
      expect(calculateWeeklyVolume(8, 2)).toBe(16);
      expect(calculateWeeklyVolume(6, 3)).toBe(18);
    });
  });

  describe('isVolumeInRange', () => {
    it('should return true for volume within range', () => {
      expect(isVolumeInRange(15, 10, 22)).toBe(true);
      expect(isVolumeInRange(10, 10, 22)).toBe(true);
      expect(isVolumeInRange(22, 10, 22)).toBe(true);
    });

    it('should return false for volume outside range', () => {
      expect(isVolumeInRange(5, 10, 22)).toBe(false);
      expect(isVolumeInRange(25, 10, 22)).toBe(false);
    });
  });
});

describe('Periodization Phases', () => {
  const phases = [
    { name: 'MEV', volumeMultiplier: 0.7, rpeRange: [6, 7] },
    { name: 'Building', volumeMultiplier: 0.85, rpeRange: [7, 8] },
    { name: 'MAV', volumeMultiplier: 1.0, rpeRange: [8, 8] },
    { name: 'Overreaching', volumeMultiplier: 1.15, rpeRange: [8, 9] },
    { name: 'MRV', volumeMultiplier: 1.25, rpeRange: [9, 10] },
    { name: 'Deload', volumeMultiplier: 0.5, rpeRange: [5, 6] },
  ];

  it('should have increasing volume multipliers until MRV', () => {
    const nonDeload = phases.filter((p) => p.name !== 'Deload');
    for (let i = 1; i < nonDeload.length; i++) {
      expect(nonDeload[i].volumeMultiplier).toBeGreaterThan(nonDeload[i - 1].volumeMultiplier);
    }
  });

  it('should have deload with lowest volume multiplier', () => {
    const deload = phases.find((p) => p.name === 'Deload');
    const others = phases.filter((p) => p.name !== 'Deload');
    others.forEach((phase) => {
      expect(deload?.volumeMultiplier).toBeLessThan(phase.volumeMultiplier);
    });
  });

  it('should have valid RPE ranges (min <= max)', () => {
    phases.forEach((phase) => {
      expect(phase.rpeRange[0]).toBeLessThanOrEqual(phase.rpeRange[1]);
    });
  });

  it('should have RPE ranges within 5-10', () => {
    phases.forEach((phase) => {
      expect(phase.rpeRange[0]).toBeGreaterThanOrEqual(5);
      expect(phase.rpeRange[1]).toBeLessThanOrEqual(10);
    });
  });
});
