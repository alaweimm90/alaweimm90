import { ComponentType } from 'react';
import DiamondLogo from '../icons/DiamondLogo';
import VelocityLogo from '../icons/VelocityLogo';
import ElevationLogo from '../icons/ElevationLogo';
import InfinityLogo from '../icons/InfinityLogo';
import PrestigeLogo from '../icons/PrestigeLogo';
import ElegantLogo from '../ElegantLogo';
import AuroraLogo from '../icons/AuroraLogo';
import PhoenixLogo from '../icons/PhoenixLogo';
import ZenithLogo from '../icons/ZenithLogo';
import ApexLogo from '../icons/ApexLogo';
import PulseLogo from '../icons/PulseLogo';
import NexusLogo from '../icons/NexusLogo';
import VortexLogo from '../icons/VortexLogo';
import RadiantLogo from '../icons/RadiantLogo';
import NovaLogo from '../icons/NovaLogo';
import SummitLogo from '../icons/SummitLogo';
import ModernWordmark from '../brandmarks/ModernWordmark';
import MinimalistStack from '../brandmarks/MinimalistStack';
import GeometricInitials from '../brandmarks/GeometricInitials';
import ScriptElegance from '../brandmarks/ScriptElegance';
import BoldStatement from '../brandmarks/BoldStatement';
import InitialsCircle from '../brandmarks/InitialsCircle';
import DiagonalDynamic from '../brandmarks/DiagonalDynamic';
import MonogramShield from '../brandmarks/MonogramShield';
import LuxurySerif from '../brandmarks/LuxurySerif';
import TechMonospace from '../brandmarks/TechMonospace';
import ClassicEmblem from '../brandmarks/ClassicEmblem';
import FuturisticType from '../brandmarks/FuturisticType';
import ArtDecoStyle from '../brandmarks/ArtDecoStyle';
import HandwrittenSignature from '../brandmarks/HandwrittenSignature';
import ArchitecturalBlock from '../brandmarks/ArchitecturalBlock';

export interface ConceptItem {
  name: string;
  component: ComponentType<{ className?: string }>;
  concept: string;
  meaning: string;
}

export const iconConcepts: ConceptItem[] = [
  {
    name: 'Current - Elegant',
    component: ElegantLogo,
    concept: 'Sophisticated geometry with L.I.I initials',
    meaning: 'Precision, luxury, and refined craftsmanship',
  },
  {
    name: 'Diamond',
    component: DiamondLogo,
    concept: 'Multi-faceted diamond symbolizing excellence',
    meaning: 'Brilliance, clarity, and uncompromising quality',
  },
  {
    name: 'Velocity',
    component: VelocityLogo,
    concept: "Dynamic motion lines with iconic 'I' center",
    meaning: 'Performance, speed, and forward momentum',
  },
  {
    name: 'Elevation',
    component: ElevationLogo,
    concept: 'Mountain peaks representing achievement',
    meaning: 'Aspiration, reaching new heights, overcoming limits',
  },
  {
    name: 'Infinity',
    component: InfinityLogo,
    concept: 'Endless loop symbolizing limitless potential',
    meaning: 'Continuous growth, eternal style, boundless possibilities',
  },
  {
    name: 'Prestige',
    component: PrestigeLogo,
    concept: 'Crown-inspired crest with central jewel',
    meaning: 'Luxury, royalty, and prestigious status',
  },
  {
    name: 'Aurora',
    component: AuroraLogo,
    concept: 'Flowing aurora waves with radiant center',
    meaning: 'Natural elegance, ethereal beauty, mesmerizing presence',
  },
  {
    name: 'Phoenix',
    component: PhoenixLogo,
    concept: 'Rising phoenix with majestic wings',
    meaning: 'Rebirth, transformation, unstoppable resilience',
  },
  {
    name: 'Zenith',
    component: ZenithLogo,
    concept: 'Concentric circles reaching peak achievement',
    meaning: 'Ultimate pinnacle, highest point, absolute excellence',
  },
  {
    name: 'Apex',
    component: ApexLogo,
    concept: 'Triangle peak representing summit',
    meaning: 'Top-tier performance, ultimate achievement, leadership',
  },
  {
    name: 'Pulse',
    component: PulseLogo,
    concept: 'Energetic heartbeat with dynamic rhythm',
    meaning: 'Vitality, life force, passionate energy',
  },
  {
    name: 'Nexus',
    component: NexusLogo,
    concept: 'Connected nodes forming central hub',
    meaning: 'Connection, unity, central point of convergence',
  },
  {
    name: 'Vortex',
    component: VortexLogo,
    concept: 'Spiraling vortex of power and motion',
    meaning: 'Unstoppable force, magnetic attraction, dynamic energy',
  },
  {
    name: 'Radiant',
    component: RadiantLogo,
    concept: 'Sun rays emanating from brilliant core',
    meaning: 'Illumination, warmth, radiating excellence',
  },
  {
    name: 'Nova',
    component: NovaLogo,
    concept: 'Explosive starburst of brilliance',
    meaning: 'New beginnings, explosive growth, stellar achievement',
  },
  {
    name: 'Summit',
    component: SummitLogo,
    concept: 'Mountain peak with victory flag',
    meaning: 'Ultimate goal, conquering challenges, triumph',
  },
];

export const brandmarkConcepts: ConceptItem[] = [
  {
    name: 'Modern Wordmark',
    component: ModernWordmark,
    concept: 'Clean typography with balanced weight distribution',
    meaning: 'Contemporary elegance with timeless appeal',
  },
  {
    name: 'Minimalist Stack',
    component: MinimalistStack,
    concept: 'Vertically stacked words with accent bar',
    meaning: 'Sophistication through simplicity and clarity',
  },
  {
    name: 'Geometric Initials',
    component: GeometricInitials,
    concept: 'L.I.I in geometric frames',
    meaning: 'Structured precision with architectural beauty',
  },
  {
    name: 'Script Elegance',
    component: ScriptElegance,
    concept: 'Flowing script with decorative swash',
    meaning: 'Refined grace with artistic flair',
  },
  {
    name: 'Bold Statement',
    component: BoldStatement,
    concept: 'Mixed weights emphasizing key words',
    meaning: 'Confidence and unmistakable presence',
  },
  {
    name: 'Initials Circle',
    component: InitialsCircle,
    concept: 'Circular badge with centered monogram',
    meaning: 'Complete, unified, timeless luxury',
  },
  {
    name: 'Diagonal Dynamic',
    component: DiagonalDynamic,
    concept: 'Slanted type with speed lines',
    meaning: 'Forward momentum and progressive energy',
  },
  {
    name: 'Monogram Shield',
    component: MonogramShield,
    concept: 'Heraldic shield with crowned initials',
    meaning: 'Heritage, protection, noble distinction',
  },
  {
    name: 'Luxury Serif',
    component: LuxurySerif,
    concept: 'Elegant serif with decorative flourishes',
    meaning: 'Opulence, tradition, refined sophistication',
  },
  {
    name: 'Tech Monospace',
    component: TechMonospace,
    concept: 'Monospace type with grid structure',
    meaning: 'Precision, innovation, modern engineering',
  },
  {
    name: 'Classic Emblem',
    component: ClassicEmblem,
    concept: 'Circular emblem with ornate details',
    meaning: 'Timeless prestige, established excellence',
  },
  {
    name: 'Futuristic Type',
    component: FuturisticType,
    concept: 'Angular letterforms with tech accents',
    meaning: 'Forward-thinking, cutting-edge, innovative',
  },
  {
    name: 'Art Deco Style',
    component: ArtDecoStyle,
    concept: 'Geometric frame with period typography',
    meaning: 'Vintage glamour meets modern luxury',
  },
  {
    name: 'Handwritten Signature',
    component: HandwrittenSignature,
    concept: 'Flowing handwritten script with flourish',
    meaning: 'Personal touch, authentic, bespoke craftsmanship',
  },
  {
    name: 'Architectural Block',
    component: ArchitecturalBlock,
    concept: 'Stacked blocks with varying sizes',
    meaning: 'Solid foundation, structured growth, dimensional',
  },
];
