export interface Perspective {
  title: string;
  name: string;
  description: string;
  icon: React.ReactNode;
}

const AthleteIcon = () => (
  <svg className="w-12 h-12 text-lii-gold" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M24 8L32 16L24 32L16 16L24 8Z" stroke="currentColor" strokeWidth="1.5" fill="none" />
    <circle cx="24" cy="24" r="12" stroke="currentColor" strokeWidth="1.5" fill="none" />
    <path d="M24 12L24 36" stroke="currentColor" strokeWidth="1.5" />
  </svg>
);

const EntrepreneurIcon = () => (
  <svg className="w-12 h-12 text-lii-gold" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="12" y="12" width="24" height="24" stroke="currentColor" strokeWidth="1.5" fill="none" />
    <path d="M18 18L30 30M30 18L18 30" stroke="currentColor" strokeWidth="1.5" />
    <circle cx="24" cy="24" r="3" fill="currentColor" />
  </svg>
);

const TravelerIcon = () => (
  <svg className="w-12 h-12 text-lii-gold" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="24" cy="24" r="14" stroke="currentColor" strokeWidth="1.5" fill="none" />
    <path d="M10 24L38 24M24 10L24 38" stroke="currentColor" strokeWidth="1.5" />
    <path d="M15 15L33 33M33 15L15 33" stroke="currentColor" strokeWidth="0.5" opacity="0.5" />
  </svg>
);

export const PODCAST_PERSPECTIVES: Perspective[] = [
  {
    title: 'The Athlete',
    name: 'Peak Performance',
    description: 'Discipline, training methodologies, and the relentless pursuit of excellence.',
    icon: <AthleteIcon />,
  },
  {
    title: 'The Entrepreneur',
    name: 'Strategic Vision',
    description: 'Building empires, calculated risks, and the art of execution.',
    icon: <EntrepreneurIcon />,
  },
  {
    title: 'The Traveler',
    name: 'Living Iconically',
    description: 'Experiences over possessions, cultural immersion, and boundless curiosity.',
    icon: <TravelerIcon />,
  },
];
